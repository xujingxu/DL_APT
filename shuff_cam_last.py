from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2

def make_model():
    resmodel = torch.load(
        './net.pth', map_location={'cuda0:': 'cuda:0'})
#     net = resmodel.cuda(device=device_ids[2])#将模型从CPU发送到GPU,如果没有GPU则删除该行
    return resmodel
model = make_model()


# model = resnet50(pretrained=True)
target_layers = [model.module.conv5]


# 读取测试图像
image_path = '/test.png'


# preprocess = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# # ])
# img1=Image.open(image_path)
# input_tensor_batch = preprocess(img1)
# input_tensor = input_tensor_batch.unsqueeze(0)

img = np.array(Image.open(image_path).convert('RGB'))
img = cv2.resize(img, (224, 224))
img = np.float32(img) / 255
input_tensor = preprocess_image(img)
targets = [ClassifierOutputTarget(0)]

imgbmd = np.array(Image.open('./test_cam.png').convert('RGB'))
imgbmd = cv2.resize(imgbmd, (224, 224))
imgbmd = np.float32(imgbmd) / 255




with GradCAM(model=model, target_layers=target_layers) as cam:
    grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
    cam_image = show_cam_on_image(imgbmd, grayscale_cams[0, :], use_rgb=True)
cam = np.uint8(255*grayscale_cams[0, :])
cam = cv2.merge([cam, cam, cam])
images = np.hstack((np.uint8(255*imgbmd), cam , cam_image))
image= Image.fromarray(images)

image.save('cam.png')