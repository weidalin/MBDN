#!/bin/bash
cp -vr models/V42*/backup/* ./
python train.py
python test.py
cp -vr models/V43*/backup/* ./
python train.py
python test.py
cp -vr models/V41*/backup/* ./
python train.py
python test.py
cp -vr models/V40*/backup/* ./
python train.py
python test.py


