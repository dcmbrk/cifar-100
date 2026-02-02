"""
Feature Extractor Module
Trích xuất đặc trưng từ ảnh sử dụng ResNet-18 pre-trained trên ImageNet.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


class FeatureExtractor:
    """Trích xuất feature vectors từ ảnh sử dụng ResNet-18."""

    def __init__(self, device=None):
        """
        Args:
            device: Device để chạy model (cuda/cpu). Nếu None sẽ tự động chọn.
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Load ResNet-18 pre-trained
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Bỏ layer cuối (fc layer) để lấy feature vector 512 chiều
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Transform cho CIFAR-100 (32x32) -> phù hợp với ImageNet pre-trained
        self.transform = transforms.Compose([
            transforms.Resize(224),  # Resize lên 224x224 cho ResNet
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract_features(self, dataset, batch_size=128):
        """
        Trích xuất features từ toàn bộ dataset.

        Args:
            dataset: PyTorch Dataset chứa ảnh
            batch_size: Batch size cho inference

        Returns:
            features: numpy array shape (n, 512) chứa feature vectors
            labels: numpy array shape (n,) chứa labels
        """
        # Áp dụng transform cho dataset
        original_transform = dataset.transform
        dataset.transform = self.transform

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        features_list = []
        labels_list = []

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Extracting features"):
                images = images.to(self.device)

                # Forward pass
                feats = self.model(images)
                feats = feats.squeeze()  # (batch, 512, 1, 1) -> (batch, 512)

                features_list.append(feats.cpu().numpy())
                labels_list.append(labels.numpy())

        # Khôi phục transform gốc
        dataset.transform = original_transform

        features = np.vstack(features_list)
        labels = np.concatenate(labels_list)

        # Chuẩn hóa L2 cho cosine similarity
        features = features / np.linalg.norm(features, axis=1, keepdims=True)

        return features, labels

    def extract_single(self, image):
        """
        Trích xuất feature từ một ảnh đơn.

        Args:
            image: PIL Image hoặc tensor

        Returns:
            feature: numpy array shape (512,)
        """
        if not torch.is_tensor(image):
            image = self.transform(image)

        image = image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            feature = self.model(image)
            feature = feature.squeeze().cpu().numpy()

        # Chuẩn hóa L2
        feature = feature / np.linalg.norm(feature)

        return feature


def load_or_extract_features(dataset, cache_path=None, device=None, batch_size=128):
    """
    Load features từ cache hoặc trích xuất mới.

    Args:
        dataset: PyTorch Dataset
        cache_path: Đường dẫn file cache (.npz)
        device: Device để chạy
        batch_size: Batch size

    Returns:
        features, labels
    """
    if cache_path is not None:
        try:
            data = np.load(cache_path)
            print(f"Loaded features from cache: {cache_path}")
            return data['features'], data['labels']
        except FileNotFoundError:
            pass

    extractor = FeatureExtractor(device=device)
    features, labels = extractor.extract_features(dataset, batch_size=batch_size)

    if cache_path is not None:
        np.savez(cache_path, features=features, labels=labels)
        print(f"Saved features to cache: {cache_path}")

    return features, labels
