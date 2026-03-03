import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class CataractDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def create_dataloader(image_paths, labels, batch_size=32, shuffle=True, transform=None):
    dataset = CataractDataset(image_paths, labels, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)