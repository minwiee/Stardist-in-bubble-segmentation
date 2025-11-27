#!/usr/bin/env python3
"""
Demo script để chạy bubble detection trên một ảnh
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

from utils_Segmentation import createLabelUNet, combinedPrediction, load_img, load_MXNet
from csbdeep.utils import normalize
import mxnet as mx
from stardist.models import StarDist2D
import matplotlib.pyplot as plt
import pathlib
import tensorflow as tf
from utils_StarBub import HiddenReco, SaveCSV_List
from tqdm import tqdm
from stardist import random_label_cmap
import matplotlib
matplotlib.rcParams["image.interpolation"] = None

# Cấu hình
base_dir = os.path.abspath('')
Model_dir = base_dir + '/../Models/'
use_gpu = True  # Chuyển thành False nếu không có GPU

# Setup GPU/CPU
if use_gpu:
    ctx = mx.gpu(0)
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
    ctx = mx.cpu(0) 
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

print("Đang tải models...")
# Load models
modelSD = StarDist2D(None, name='stardist', basedir=Model_dir + 'SDmodel')
netMask, netInter = load_MXNet(Model_dir, ctx, "_512")
model = tf.keras.models.load_model(Model_dir + 'RDCModel')
print("Models đã được tải thành công!")

# Cấu hình prediction
Metric = 6.7E-5  # Kích thước pixel trong mét
useRDC = True    # Sử dụng RDC method
boolplot = True  # Hiển thị kết quả

# Đường dẫn ảnh
ImgDir = base_dir + '/Examples/00000.jpg'

print(f"Đang xử lý ảnh: {ImgDir}")

# Load và normalize ảnh
x = load_img(ImgDir)
X = normalize(x if x.ndim == 2 else x[..., 0], 1, 99.8, axis=(0, 1))

# Tạo mask với UNet
imgMask, imgIntersec = createLabelUNet(X, 2, netMask, 512, 300, ctxMask=ctx)

# Prediction với StarDist
labels, _ = combinedPrediction(X, modelSD, imgMask, imgIntersec)

# Hiển thị kết quả
if boolplot:
    lbl_cmap = random_label_cmap()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # UNet mask
    ax1.imshow(X, cmap='gray')
    ax1.imshow(imgMask, cmap='Reds', alpha=0.5)
    ax1.set_axis_off()
    ax1.set_title('UNet mask')
    
    # Segmentation mask
    ax2.imshow(X, cmap='gray')
    ax2.imshow(labels, cmap=lbl_cmap, alpha=0.5)
    ax2.set_axis_off()
    ax2.set_title('Segmentation mask')
    
    # Hidden part reconstruction
    ax3.imshow(X, cmap='gray')
    ax3.set_axis_off() 
    ax3.set_title('Hidden part reconstruction')   
    
    # Reconstruct bubbles
    Bubbles = HiddenReco(labels, Metric, useRDC=useRDC, model=model, boolPlot=boolplot, ax=ax3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Đã phát hiện {len(Bubbles)} bubbles")
    if Bubbles:
        print("Thông tin bubbles:")
        for i, bubble in enumerate(Bubbles[:5]):  # Hiển thị 5 bubbles đầu
            print(f"Bubble {i+1}: {bubble}")
else:
    Bubbles = HiddenReco(labels, Metric, useRDC=useRDC, model=model, boolPlot=False)
    print(f"Đã phát hiện {len(Bubbles)} bubbles")

print("Hoàn thành!")
