"""
Trainer Module
Huấn luyện mô hình trên coreset được chọn.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import models
import numpy as np
from tqdm import tqdm
import time


class SimpleResNet(nn.Module):
    """ResNet-18 cho CIFAR-100 classification."""

    def __init__(self, num_classes=100, pretrained=True):
        super().__init__()

        if pretrained:
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet18(weights=None)

        # Thay fc layer cho 100 classes
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


class Trainer:
    """Trainer cho CIFAR-100 với coreset selection."""

    def __init__(
        self,
        model=None,
        device=None,
        num_classes=100,
        pretrained=True
    ):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        if model is None:
            self.model = SimpleResNet(num_classes=num_classes, pretrained=pretrained)
        else:
            self.model = model

        self.model = self.model.to(self.device)

    def train(
        self,
        train_dataset,
        selected_indices=None,
        epochs=50,
        batch_size=64,
        lr=0.01,
        momentum=0.9,
        weight_decay=1e-4,
        lr_scheduler='cosine',
        verbose=True
    ):
        """
        Huấn luyện mô hình.

        Args:
            train_dataset: Full training dataset
            selected_indices: List indices của coreset (None = dùng toàn bộ)
            epochs: Số epochs
            batch_size: Batch size
            lr: Learning rate
            momentum: SGD momentum
            weight_decay: Weight decay
            lr_scheduler: 'cosine', 'step', hoặc None
            verbose: In tiến trình

        Returns:
            history: Dict chứa loss và accuracy theo epoch
            runtime: Thời gian huấn luyện (giây)
        """
        start_time = time.time()

        # Tạo subset nếu có selected_indices
        if selected_indices is not None:
            train_data = Subset(train_dataset, selected_indices)
            if verbose:
                print(f"Training on coreset: {len(selected_indices)} samples "
                      f"({100*len(selected_indices)/len(train_dataset):.1f}% of full dataset)")
        else:
            train_data = train_dataset
            if verbose:
                print(f"Training on full dataset: {len(train_dataset)} samples")

        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        # Loss và optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        if lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        elif lr_scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epochs//3, gamma=0.1)
        else:
            scheduler = None

        history = {
            'train_loss': [],
            'train_acc': []
        }

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            if verbose:
                pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            else:
                pbar = train_loader

            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                if verbose:
                    pbar.set_postfix({
                        'loss': running_loss / total,
                        'acc': 100. * correct / total
                    })

            epoch_loss = running_loss / total
            epoch_acc = 100. * correct / total
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc)

            if scheduler:
                scheduler.step()

            if verbose:
                print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%")

        runtime = time.time() - start_time
        return history, runtime

    def evaluate(self, test_dataset, batch_size=128):
        """
        Đánh giá mô hình trên test set.

        Args:
            test_dataset: Test dataset
            batch_size: Batch size

        Returns:
            test_loss: Loss trung bình
            test_acc: Accuracy (%)
            per_class_acc: Dict accuracy từng class
        """
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        criterion = nn.CrossEntropyLoss()

        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        # Per-class statistics
        class_correct = {}
        class_total = {}

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Evaluating"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Per-class accuracy
                for label, pred in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                    if label not in class_correct:
                        class_correct[label] = 0
                        class_total[label] = 0
                    class_total[label] += 1
                    if label == pred:
                        class_correct[label] += 1

        test_loss = running_loss / total
        test_acc = 100. * correct / total

        per_class_acc = {
            cls: 100. * class_correct[cls] / class_total[cls]
            for cls in class_correct
        }

        return test_loss, test_acc, per_class_acc

    def save_checkpoint(self, path, epoch=None, optimizer=None, history=None):
        """Lưu checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
        }
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if history is not None:
            checkpoint['history'] = history

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint


def train_on_coreset(
    train_dataset,
    test_dataset,
    selected_indices,
    epochs=50,
    batch_size=64,
    lr=0.01,
    device=None,
    verbose=True
):
    """
    Helper function để train và evaluate trên coreset.

    Returns:
        results: Dict chứa kết quả training và evaluation
    """
    trainer = Trainer(device=device)

    history, train_time = trainer.train(
        train_dataset,
        selected_indices=selected_indices,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        verbose=verbose
    )

    test_loss, test_acc, per_class_acc = trainer.evaluate(test_dataset)

    results = {
        'train_history': history,
        'train_time': train_time,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'per_class_acc': per_class_acc,
        'coreset_size': len(selected_indices) if selected_indices else len(train_dataset)
    }

    if verbose:
        print(f"\nTest Accuracy: {test_acc:.2f}%")

    return results
