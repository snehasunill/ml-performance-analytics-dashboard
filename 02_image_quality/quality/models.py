"""Classical features plus two PyTorch image classifiers."""
import cv2, numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from .data import LABELS

def features(img):
    a=np.asarray(img.convert('RGB'));gray=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)
    lap=cv2.Laplacian(gray,cv2.CV_64F)
    sobx=cv2.Sobel(gray,cv2.CV_32F,1,0)
    soby=cv2.Sobel(gray,cv2.CV_32F,0,1)
    hsv=cv2.cvtColor(a,cv2.COLOR_RGB2HSV)
    return np.array([lap.var(), np.abs(lap).mean(), gray.std(), gray.mean(),
       np.mean(np.abs(sobx)),np.mean(np.abs(soby)), np.std(hsv[:,:,1]),
       np.mean(np.abs(gray[:,1:].astype(float)-gray[:,:-1].astype(float))),
       np.mean(np.abs(gray[1:,:].astype(float)-gray[:-1,:].astype(float)))],dtype=np.float32)

def classical(seed=42): return RandomForestClassifier(n_estimators=250,class_weight='balanced',random_state=seed,n_jobs=-1)

def neural(kind, pretrained=False):
    import torch
    from torch import nn
    if kind=='cnn':
        return nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(128,4))
    if kind=='transfer':
        from torchvision.models import resnet18, ResNet18_Weights
        model=resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc=nn.Linear(model.fc.in_features,4)
        return model
    raise ValueError('expected cnn or transfer')

def transform(size=224, neural_kind='cnn'):
    from torchvision import transforms
    steps=[transforms.Resize((size,size)),transforms.ToTensor()]
    if neural_kind=='transfer':steps.append(transforms.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225]))
    return transforms.Compose(steps)
