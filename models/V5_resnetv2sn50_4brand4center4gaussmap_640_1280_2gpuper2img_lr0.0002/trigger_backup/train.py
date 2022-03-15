import os
import time
import torch
import json
import torch.optim as optim
import numpy as np
import time
from copy import deepcopy
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose, ColorJitter

from net.loss import *
from net.network_sn_101 import ACSPNet
from config import Config
from dataloader.loader import *
from sys import exit
ticks = time.time()

config = Config()
config.train_path = './data/citypersons'
config.test_path = './data/citypersons'
config.gpu_ids = [0,1]
config.onegpu = 2
# config.size_train = (1024, 2048)
# config.size_train = (224, 448)
config.size_train = (640, 1280)
config.size_test = (1024, 2048)
config.init_lr = 2e-4
config.num_epochs = 150
config.val = False
config.offset = True
config.teacher = True
# dataset
traintransform = Compose(
    [ColorJitter(brightness=0.5), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
traindataset = CityPersons(path=config.train_path, type='train', config=config,
                           transform=traintransform)
trainloader = DataLoader(traindataset, batch_size=config.onegpu*len(config.gpu_ids))


# net
print('Net...')
net = ACSPNet().cuda()

version = 'V5_%s_4brand4center4gaussmap_%d_%d_%dgpuper%dimg_lr%s'%(net.resnetType, config.size_train[0], config.size_train[1],len(config.gpu_ids),config.onegpu,config.init_lr)
# To continue training
resume_from = ''
# resume_from = './models/%s/ckpt/ACSP_8.pth.tea'%version
begin_epoch_num = 0
if len(resume_from)>0:
    net.load_state_dict(torch.load(resume_from))
    begin_epoch_num = resume_from[resume_from.rfind('_')+1:resume_from.rfind('.pth')]
print("begin_epoch_num:", begin_epoch_num)

# position
center = cls_pos().cuda()
height = reg_pos().cuda()
offset = offset_pos().cuda()

# optimizer
params = []
for n, p in net.named_parameters():
    if p.requires_grad:
        params.append({'params': p})
    else:
        print(n)

if config.teacher:
    teacher_dict = net.state_dict()

# net = nn.DataParallel(net, device_ids=config.gpu_ids)
net = nn.DataParallel(net, device_ids=config.gpu_ids)

optimizer = optim.Adam(params, lr=config.init_lr)

batchsize = config.onegpu * len(config.gpu_ids)
train_batches = len(trainloader)

config.print_conf()


def criterion(output, label): #lebel 0(gausshm, mask, hm)  1(logheigh, mask)  2(height-Y offset, width-X offset, mask)
    #torch.Size([1, 1, 160, 320]) torch.Size([1, 1, 160, 320]) torch.Size([1, 2, 160, 320])
    cls_loss = center(output[0], label[0])
    reg_loss = height(output[1], label[1])
    off_loss = offset(output[2], label[2])
    return cls_loss, reg_loss, off_loss


def train():
    print('Training start')
    if not os.path.exists('./models/'+version):
        os.mkdir('./models/'+version)
    if not os.path.exists('./models/'+version+'/ckpt/'):
        os.mkdir('./models/'+version+'/ckpt')
    if not os.path.exists('./models/'+version+'/loss'):
        os.mkdir('./models/'+version+'/loss')
    if not os.path.exists('./models/'+version+'/log'):
        os.mkdir('./models/'+version+'/log')
        os.mkdir('./models/'+version+'/validation_result_log')

    # open log file
    log_floder = './models/'+version+'/log/'
    log_file = log_floder + time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time())) + '.log'
    log = open(log_file, 'w')

    best_loss = np.Inf
    best_loss_epoch = 0

    
    loss_list = []

    for epoch in range(int(begin_epoch_num),150):
        print('----------')
        print('Epoch %d begin' % (epoch + 1))
        t1 = time.time()

        epoch_loss = 0.0
        net.train()

        for i, data in enumerate(trainloader, 0):

            t3 = time.time()
            # get the inputs
            inputs, labels = data
            inputs = inputs.cuda()
            labels = [l.cuda().float() for l in labels]

            # zero the parameter gradients
            optimizer.zero_grad()

            # heat map
            outputs = net(inputs) #torch.Size([1, 1, 160, 320]) torch.Size([1, 1, 160, 320]) torch.Size([1, 2, 160, 320]) input:torch.Size([1, 3, 640, 1280])

            # loss
            cls_loss, reg_loss, off_loss = criterion(outputs, labels)
            loss = cls_loss + reg_loss + off_loss

            # back-prop
            loss.backward()

            # update param
            optimizer.step()
            if config.teacher:
                for k, v in net.module.state_dict().items():
                # for k, v in net.state_dict().items():
                    if k.find('num_batches_tracked') == -1:#？？？
                        #print("Use mean teacher")
                        teacher_dict[k] = config.alpha * teacher_dict[k] + (1 - config.alpha) * v
                    else:
                        #print("Nullify mean teacher")
                        teacher_dict[k] = 1 * v

            # print statistics
            batch_loss = loss.item()
            batch_cls_loss = cls_loss.item()
            batch_reg_loss = reg_loss.item()
            batch_off_loss = off_loss.item()

            t4 = time.time()
            print('\r[Epoch %d/150, Batch %d/%d]$ <Total loss: %.6f> cls: %.6f, reg: %.6f, off: %.6f, Time: %.3f sec        ' %
                  (epoch + 1, i + 1, train_batches, batch_loss, batch_cls_loss, batch_reg_loss, batch_off_loss, t4-t3)),
            log.write('\r[Epoch %d/150, Batch %d/%d]$ <Total loss: %.6f> cls: %.6f, reg: %.6f, off: %.6f, Time: %.3f sec        ' %
                  (epoch + 1, i + 1, train_batches, batch_loss, batch_cls_loss, batch_reg_loss, batch_off_loss, t4-t3))
            epoch_loss += batch_loss
        print('')

        t2 = time.time()
        epoch_loss /= len(trainloader)
        loss_list.append(epoch_loss)
        loss_out = np.array(loss_list)
        name = "./models/"+version+"/loss/loss_" + str(epoch) + ".npy"
        np.save(name,loss_out)
        
        print('Epoch %d end, AvgLoss is %.6f, Time used %.1f sec.' % (epoch+1, epoch_loss, int(t2-t1)))
        log.write('Epoch %d end, AvgLoss is %.6f, Time used %.1f sec.' % (epoch+1, epoch_loss, int(t2-t1)))
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_loss_epoch = epoch + 1
        print('Epoch %d has lowest loss: %.7f' % (best_loss_epoch, best_loss))

        
        log.write('%d %.7f\n' % (epoch+1, epoch_loss))
            
        print('Save checkpoint...')
        filename = './models/%s/ckpt/%s_%d.pth' % (version,'ACSP',epoch+1)

        # torch.save(net.module.state_dict(), filename)
        torch.save(net.state_dict(), filename)
        if config.teacher:
            torch.save(teacher_dict, filename+'.tea')

        print('%s saved.' % filename)

    log.close()



if __name__ == '__main__':
    train()
