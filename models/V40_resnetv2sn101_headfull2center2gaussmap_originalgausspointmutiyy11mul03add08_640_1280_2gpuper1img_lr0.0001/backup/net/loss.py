import torch.nn as nn
import torch


class cls_pos(nn.Module):
    def __init__(self):
        super(cls_pos, self).__init__()
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, pos_pred, pos_label):  # 0-gauss 1-mask 2-center
        log_loss = self.bce(pos_pred[:, 0, :, :], pos_label[:, 2, :, :])

        positives = pos_label[:, 2, :, :]
        negatives = pos_label[:, 1, :, :] - pos_label[:, 2, :, :]

        fore_weight = positives * (1.0-pos_pred[:, 0, :, :]) ** 2
        back_weight = negatives * ((1.0-pos_label[:, 0, :, :])**4.0) * (pos_pred[:, 0, :, :]**2.0)

        focal_weight = fore_weight + back_weight
        assigned_box = torch.sum(pos_label[:, 2, :, :])

        cls_loss = 0.01 * torch.sum(focal_weight*log_loss) / max(1.0, assigned_box)

        return cls_loss


class reg_pos(nn.Module):
    def __init__(self):
        super(reg_pos, self).__init__()
        self.l1 = nn.L1Loss(reduce=False, size_average=False)
    def forward(self, h_pred, h_label):
        l1_loss = h_label[:, 1, :, :]*self.l1(h_pred[:, 0, :, :]/(h_label[:, 0, :, :]+1e-10),
                                                    h_label[:, 0, :, :]/(h_label[:, 0, :, :]+1e-10))
        reg_loss = 5 * 0.01 * torch.sum(l1_loss) / max(1.0, torch.sum(h_label[:, 1, :, :]))
        return reg_loss


class offset_pos(nn.Module):
    def __init__(self):
        super(offset_pos, self).__init__()
        self.smoothl1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, offset_pred, offset_label):
        l1_loss = offset_label[:, 2, :, :].unsqueeze(dim=1)*self.smoothl1(offset_pred, offset_label[:, :2, :, :])
        off_loss = 0.1 * torch.sum(l1_loss) / max(1.0, torch.sum(offset_label[:, 2, :, :]))
        return off_loss



# class cls_pos(nn.Module):
#     def __init__(self):
#         super(cls_pos, self).__init__()
#         self.bce = nn.BCELoss(reduction='none')
#
#     def forward(self, pos_pred, pos_label):  # 0-gauss 1-mask 2-center
#         #pos_pred: torch.Size([1, 3, 160, 320])
#         #pos_label: torch.Size([1, 7, 160, 320])
#         #backup
#         # def forward(self, pos_pred, pos_label):  # 0-gauss 1-mask 2-center
#         #     log_loss = self.bce(pos_pred[:, 0, :, :], pos_label[:, 2, :, :])
#         #     positives = pos_label[:, 2, :, :]
#         #     negatives = pos_label[:, 1, :, :] - pos_label[:, 2, :, :]
#         #     fore_weight = positives * (1.0 - pos_pred[:, 0, :, :]) ** 2
#         #     back_weight = negatives * ((1.0 - pos_label[:, 0, :, :]) ** 4.0) * (pos_pred[:, 0, :, :] ** 2.0)
#         #     focal_weight = fore_weight + back_weight
#         #     assigned_box = torch.sum(pos_label[:, 2, :, :])
#         #     cls_loss = 0.01 * torch.sum(focal_weight * log_loss) / max(1.0, assigned_box)
#         #     return cls_loss
#         #end backup
#
#         log_loss1 = self.bce(pos_pred[:, 0, :, :], pos_label[:, 4, :, :])
#         log_loss2 = self.bce(pos_pred[:, 1, :, :], pos_label[:, 5, :, :])
#         log_loss3 = self.bce(pos_pred[:, 2, :, :], pos_label[:, 6, :, :])
#         positives1 = pos_label[:, 4, :, :]
#         positives2 = pos_label[:, 5, :, :]
#         positives3 = pos_label[:, 6, :, :]
#         negatives1 = pos_label[:, 3, :, :] - pos_label[:, 4, :, :]
#         negatives2 = pos_label[:, 3, :, :] - pos_label[:, 5, :, :]
#         negatives3 = pos_label[:, 3, :, :] - pos_label[:, 6, :, :]
#         fore_weight1 = positives1 * (1.0 - pos_pred[:, 0, :, :]) ** 2  # fw = (1-hm*)^2  where center point
#         fore_weight2 = positives2 * (1.0 - pos_pred[:, 1, :, :]) ** 2  # fw = (1-hm*)^2  where center point
#         fore_weight3 = positives3 * (1.0 - pos_pred[:, 2, :, :]) ** 2  # fw = (1-hm*)^2  where center point
#         back_weight1 = negatives1 * ((1.0 - pos_label[:, 0, :, :]) ** 4.0) * (
#                     pos_pred[:, 0, :, :] ** 2.0)  # bw = (1-hm)^4*hm*^2 where center point else
#         back_weight2 = negatives2 * ((1.0 - pos_label[:, 1, :, :]) ** 4.0) * (
#                     pos_pred[:, 1, :, :] ** 2.0)  # bw = (1-hm)^4*hm*^2 where center point else
#         back_weight3 = negatives3 * ((1.0 - pos_label[:, 2, :, :]) ** 4.0) * (
#                     pos_pred[:, 2, :, :] ** 2.0)  # bw = (1-hm)^4*hm*^2 where center point else
#         focal_weight1 = fore_weight1 + back_weight1  # focal loss
#         focal_weight2 = fore_weight2 + back_weight2  # focal loss
#         focal_weight3 = fore_weight3 + back_weight3  # focal loss
#
#         assigned_box1 = torch.sum(pos_label[:, 4, :, :]) #bbox number
#         assigned_box2 = torch.sum(pos_label[:, 5, :, :]) #bbox number
#         assigned_box3 = torch.sum(pos_label[:, 6, :, :]) #bbox number
#
#         cls_loss1 = 0.01 * torch.sum(focal_weight1*log_loss1) / max(1.0, assigned_box1)
#         cls_loss2 = 0.01 * torch.sum(focal_weight2*log_loss2) / max(1.0, assigned_box2)
#         cls_loss3 = 0.01 * torch.sum(focal_weight3*log_loss3) / max(1.0, assigned_box3)
#
#         v_log_loss = self.bce(pos_pred[:, 3, :, :], pos_label[:, 8, :, :])
#         v_positives = pos_label[:, 8, :, :]
#         v_negatives = pos_label[:, 1, :, :] - pos_label[:, 8, :, :]
#         v_fore_weight = v_positives * (1.0 - pos_pred[:, 3, :, :]) ** 2  # fw = (1-hm*)^2  where center point
#         v_back_weight = v_negatives * ((1.0 - pos_label[:, 7, :, :]) ** 4.0) * (
#                 pos_pred[:, 3, :, :] ** 2.0)  # bw = (1-hm)^4*hm*^2 where center point else
#         v_focal_weight = v_fore_weight + v_back_weight  # focal loss
#         v_assigned_box = torch.sum(pos_label[:, 8, :, :])  # bbox number
#         v_cls_loss = 0.01 * torch.sum(v_focal_weight * v_log_loss) / max(1.0, v_assigned_box)
#
#         cls_loss = cls_loss1+cls_loss2+cls_loss3 + v_cls_loss
#         return cls_loss
#
#
# class reg_pos(nn.Module):
#     def __init__(self):
#         super(reg_pos, self).__init__()
#         self.l1 = nn.L1Loss(reduce=False, size_average=False)
#     def forward(self, h_pred, h_label): #h_pre:torch.Size([1, 3, 160, 320])  h_label:torch.Size([1, 6, 160, 320])
#         #backup
#         # l1_loss = h_label[:, 1, :, :] * self.l1(h_pred[:, 0, :, :] / (h_label[:, 0, :, :] + 1e-10),
#         #                                         h_label[:, 0, :, :] / (h_label[:, 0, :, :] + 1e-10))
#         # reg_loss = 5 * 0.01 * torch.sum(l1_loss) / max(1.0, torch.sum(h_label[:, 1, :, :]))
#         # return reg_loss
#         #end backup
#         l1_loss1 = h_label[:, 3, :, :]*self.l1(h_pred[:, 0, :, :]/(h_label[:, 0, :, :]+1e-10),
#                                                     h_label[:, 0, :, :]/(h_label[:, 0, :, :]+1e-10))
#         reg_loss1 = 5 * 0.01 * torch.sum(l1_loss1) / max(1.0, torch.sum(h_label[:, 3, :, :]))
#
#         l1_loss2 = h_label[:, 4, :, :]*self.l1(h_pred[:, 1, :, :]/(h_label[:, 1, :, :]+1e-10),
#                                                     h_label[:, 1, :, :]/(h_label[:, 1, :, :]+1e-10))
#         reg_loss2 = 5 * 0.01 * torch.sum(l1_loss2) / max(1.0, torch.sum(h_label[:, 4, :, :]))
#
#         l1_loss3 = h_label[:, 5, :, :]*self.l1(h_pred[:, 2, :, :]/(h_label[:, 2, :, :]+1e-10),
#                                                     h_label[:, 2, :, :]/(h_label[:, 2, :, :]+1e-10))
#         reg_loss3 = 5 * 0.01 * torch.sum(l1_loss3) / max(1.0, torch.sum(h_label[:, 5, :, :]))
#
#         reg_loss = reg_loss1+reg_loss2+reg_loss3
#         return reg_loss
#
#
# class offset_pos(nn.Module):
#     def __init__(self):
#         super(offset_pos, self).__init__()
#         self.smoothl1 = nn.SmoothL1Loss(reduction='none')
#
#     def forward(self, offset_pred, offset_label):
#         #backup
#         # l1_loss = offset_label[:, 2, :, :].unsqueeze(dim=1) * self.smoothl1(offset_pred, offset_label[:, :2, :, :])
#         # off_loss = 0.1 * torch.sum(l1_loss) / max(1.0, torch.sum(offset_label[:, 2, :, :]))
#         # return off_loss
#         #end backup
#         # smooth_l1_loss = self.smoothl1(offset_pred, offset_label[:, :6, :, :]) #torch.Size([2, 6, 160, 320])
#         l1_loss1 = offset_label[:, 6, :, :].unsqueeze(dim=1) * self.smoothl1(offset_pred[:, :2, :, :], offset_label[:, :2, :, :])
#         off_loss1 = 0.1 * torch.sum(l1_loss1) / max(1.0, torch.sum(offset_label[:, 6, :, :]))
#         l1_loss2 = offset_label[:, 7, :, :].unsqueeze(dim=1) * self.smoothl1(offset_pred[:, 2:4, :, :], offset_label[:, 2:4, :, :])
#         off_loss2 = 0.1 * torch.sum(l1_loss2) / max(1.0, torch.sum(offset_label[:, 7, :, :]))
#         l1_loss3 = offset_label[:, 8, :, :].unsqueeze(dim=1) * self.smoothl1(offset_pred[:, 4:6, :, :], offset_label[:, 4:6, :, :])
#         off_loss3 = 0.1 * torch.sum(l1_loss3) / max(1.0, torch.sum(offset_label[:, 8, :, :]))
#         off_loss = off_loss1+off_loss2+off_loss3
#         return off_loss
