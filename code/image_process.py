from pathlib import Path
import numpy as np
import pandas as pd 
import PIL.Image as Image
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
import torch
import random
import os

# Set random seed
random.seed(42)
torch.manual_seed(42)

# # Load metadata, labels, and splits
# p = Path(os.getcwd())
# metadata = pd.read_csv(p/'data/mimic-cxr-2.0.0-metadata.csv.gz')
# chexpert = pd.read_csv(p/'data/mimic-cxr-2.0.0-chexpert.csv.gz')
# negbio = pd.read_csv(p/'data/mimic-cxr-2.0.0-negbio.csv.gz')
# split = pd.read_csv(p/'data/mimic-cxr-2.0.0-split.csv.gz')

# # 
# example = metadata.iloc[0]
# subID = 'p' + str(example['subject_id'])
# fold = subID[0:3]
# studyID = 's' + str(example['study_id'])
# dicomID = str(example['dicom_id']) + '.jpg'
# dataP = p / 'data/files'
# filePath = dataP / fold / subID / studyID / dicomID

# # Load image
# im = Image.open(filePath)

# # Transforms
# data_transforms = transforms.Compose([
#         transforms.RandomResizedCrop(256),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5], [0.5])
#         ])

# # Apply transforms
# im  = data_transforms(im)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

feature_extract = False

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

model_ft = torch.hub.load('pytorch/vision', 'densenet121', pretrained=True)
set_parameter_requires_grad(model_ft, feature_extract)
num_ftrs = model_ft.classifier.in_features
model_ft.classifier = torch.nn.Linear(num_ftrs, 2)
input_size = 256
model_ft = model_ft.to(device)

