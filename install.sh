#!/bin/bash

# https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=9_FzH13EjseR
pip install torch torchvision pyyaml==5.1 pycocotools>=2.0.1 opencv-python

# https://detectron2.readthedocs.io/tutorials/install.html
cd ..
CC=clang CXX=clang++ python -m pip install python -m pip install -e detectron2
cd -
