import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from PIL import Image

# Unified label mapping for all datasets
CLASS_MAPPING = {
    'vehicle': 0,
    'signpost': 1,
    'obstacle': 2,
    'stoplight': 3,
    'pedestrian': 4
}

def _map_labels(dataset_type, original_label):
    """Map dataset-specific labels to unified classes"""
    if dataset_type == 'coco':
        # COCO label mapping logic
        if original_label in [2, 3, 4, 6, 8]:  # Car, motorcycle, bus, truck, bicycle
            return CLASS_MAPPING['vehicle']
        elif original_label == 1:  # Person
            return CLASS_MAPPING['pedestrian']
        elif original_label == 10:  # Traffic light
            return CLASS_MAPPING['stoplight']
        elif original_label == 13:  # Stop sign
            return CLASS_MAPPING['signpost']
        else:
            return CLASS_MAPPING['obstacle']
    elif dataset_type == 'lisa':
        # LISA Traffic Sign Dataset mapping
        if 'stop' in original_label.lower():
            return CLASS_MAPPING['stoplight']
        elif 'pedestrian' in original_label.lower():
            return CLASS_MAPPING['pedestrian']
        elif 'speed' in original_label.lower() or 'yield' in original_label.lower():
            return CLASS_MAPPING['signpost']
        else:
            return CLASS_MAPPING['obstacle']
    elif dataset_type == 'mapillary':
        # Mapillary Traffic Sign Dataset mapping
        if 'regulatory' in original_label.lower():
            return CLASS_MAPPING['signpost']
        elif 'warning' in original_label.lower():
            return CLASS_MAPPING['obstacle']
        elif 'pedestrian' in original_label.lower():
            return CLASS_MAPPING['pedestrian']
        else:
            return CLASS_MAPPING['stoplight']
    elif dataset_type == 'lyft':
        # Lyft L5 Dataset mapping
        if original_label == 'vehicle':
            return CLASS_MAPPING['vehicle']
        elif original_label == 'pedestrian':
            return CLASS_MAPPING['pedestrian']
        elif original_label == 'traffic_light':
            return CLASS_MAPPING['stoplight']
        elif original_label == 'traffic_sign':
            return CLASS_MAPPING['signpost']
        else:
            return CLASS_MAPPING['obstacle']
    elif dataset_type == 'bdd100k':
        # BDD100K Dataset mapping
        if original_label == 'car' or original_label == 'truck' or original_label == 'bus':
            return CLASS_MAPPING['vehicle']
        elif original_label == 'traffic light':
            return CLASS_MAPPING['stoplight']
        elif original_label == 'traffic sign':
            return CLASS_MAPPING['signpost']
        elif original_label == 'person':
            return CLASS_MAPPING['pedestrian']
        else:
            return CLASS_MAPPING['obstacle']
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

class MosaicDataset(Dataset):
    """Mosaic data augmentation for YOLOv9"""
    def __init__(self, base_dataset, size=640, scale=(0.5, 1.5)):
        self.base = base_dataset
        self.size = size
        self.scale = scale
        
    def __len__(self):
        return len(self.base) * 2  # Augmented dataset
    
    def __getitem__(self, idx):
        # Original sample
        if idx < len(self.base):
            return self.base[idx]
            
        # Mosaic augmentation
        indices = [np.random.randint(len(self.base)) for _ in range(4)]
        images = [self.base[i][0] for i in indices]
        labels = [self.base[i][1] for i in indices]
        
        # Create mosaic canvas
        s = self.size
        mosaic = Image.new('RGB', (s*2, s*2))
        positions = [(0,0), (s,0), (0,s), (s,s)]
        for img, pos in zip(images, positions):
            img = transforms.Resize(int(s*np.random.uniform(*self.scale)))(img)
            img = transforms.RandomHorizontalFlip(0.5)(img)
            mosaic.paste(img, pos)
            
        # Random perspective
        mosaic = transforms.RandomPerspective(0.5)(mosaic)
        mosaic = transforms.Resize(s)(mosaic)
        mosaic = transforms.ToTensor()(mosaic)
        
        # Combine labels
        label = torch.stack(labels).max(dim=0)[0]  # Use max class presence
        return mosaic, label

def get_datasets():
    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load COCO 2017 Dataset
    coco_dataset = datasets.CocoDetection(
        root="path/to/coco",
        annFile="path/to/coco/annotations",
        transform=transform,
        target_transform=lambda x: _map_labels('coco', x)
    )

    # Load LISA Traffic Sign Dataset
    lisa_dataset = datasets.ImageFolder(
        root="path/to/lisa",
        transform=transform,
        target_transform=lambda x: _map_labels('lisa', x)
    )

    # Load Mapillary Traffic Sign Dataset
    mapillary_dataset = datasets.ImageFolder(
        root="path/to/mapillary",
        transform=transform,
        target_transform=lambda x: _map_labels('mapillary', x)
    )

    # Load Lyft L5 Dataset
    lyft_dataset = datasets.ImageFolder(
        root="path/to/lyft",
        transform=transform,
        target_transform=lambda x: _map_labels('lyft', x)
    )

    # Load BDD100K Dataset
    bdd100k_dataset = datasets.ImageFolder(
        root="path/to/bdd100k",
        transform=transform,
        target_transform=lambda x: _map_labels('bdd100k', x)
    )

    # Split BDD100K Dataset into train, validation, and test sets
    train_size = int(0.8 * len(bdd100k_dataset))
    val_test_size = len(bdd100k_dataset) - train_size
    val_size = val_test_size // 2
    test_size = val_test_size - val_size

    train_bdd, val_bdd, test_bdd = torch.utils.data.random_split(
        bdd100k_dataset, [train_size, val_size, test_size]
    )

    # Combine all datasets
    combined_train = ConcatDataset([coco_dataset, lisa_dataset, mapillary_dataset, lyft_dataset, train_bdd])
    combined_val = val_bdd
    combined_test = test_bdd

    # Apply Mosaic augmentation to training dataset
    train_dataset = MosaicDataset(combined_train)
    return train_dataset, combined_val, combined_test

def get_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=64):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader