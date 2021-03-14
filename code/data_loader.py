from pathlib import Path
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import PIL.Image as Image

class CXRDataset(Dataset):

    def __init__(self, transform=None):
        self.p = Path(os.getcwd())
        self.metadata = pd.read_csv(self.p/'data/mimic-cxr-2.0.0-metadata.csv.gz')
        self.chexpert = pd.read_csv(self.p/'data/mimic-cxr-2.0.0-chexpert.csv.gz')
        self.negbio = pd.read_csv(self.p/'data/mimic-cxr-2.0.0-negbio.csv.gz')
        self.split = pd.read_csv(self.p/'data/mimic-cxr-2.0.0-split.csv.gz')
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        example = self.metadata.iloc[idx]
        subID = 'p' + str(example['subject_id'])
        fold = subID[0:3]
        studyID = 's' + str(example['study_id'])
        dicomID = str(example['dicom_id']) + '.jpg'
        dataP = self.p / 'data/files'
        filePath = dataP / fold / subID / studyID / dicomID
        im = Image.open(filePath)

        lab = self.chexpert.loc[self.chexpert['study_id'] == example['study_id']]

        if self.transform:
            im = self.transform(im)

        return (im, lab)

data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
        ])


ds = CXRDataset(transform=data_transforms)
print(ds[0])

