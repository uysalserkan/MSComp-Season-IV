import torch
from torchvision import datasets, transforms
from typing import Optional, Tuple


class STL10DatasetLoader:
    """
    Class for loading STL-10 dataset from PyTorch (torchvision).
    
    This class provides methods to load STL-10 datasets and create data loaders
    with configurable transforms and data loading parameters.
    """
    
    # Default normalization values (ImageNet statistics)
    DEFAULT_MEAN = [0.485, 0.456, 0.406]
    DEFAULT_STD = [0.229, 0.224, 0.225]
    
    def __init__(
        self,
        root: str = "./data",
        download: bool = True,
        default_transform: Optional[transforms.Compose] = None,
    ):
        """
        Initialize STL-10 dataset loader.
        
        Args:
            root: Root directory where the dataset will be stored.
            download: If True, downloads the dataset from the internet when first accessed.
            default_transform: Default transform to apply to images. If None, uses
                             default normalization transform.
        """
        self.root = root
        self.download = download
        
        if default_transform is None:
            self.default_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=self.DEFAULT_MEAN, std=self.DEFAULT_STD)
            ])
        else:
            self.default_transform = default_transform
    
    def get_default_transform(self) -> transforms.Compose:
        """
        Get the default transform used by this loader.
        
        Returns:
            Default transform composition.
        """
        return self.default_transform
    
    @staticmethod
    def get_vit_train_transform(
        image_size: int = 224,
        mean: Optional[list] = None,
        std: Optional[list] = None,
        use_augmentation: bool = True
    ) -> transforms.Compose:
        """
        Get ViT-specific training transforms with data augmentation.
        
        Args:
            image_size: Target image size (default 224x224 for ViT).
            mean: Normalization mean values. If None, uses ImageNet statistics.
            std: Normalization std values. If None, uses ImageNet statistics.
            use_augmentation: If True, applies data augmentation.
        
        Returns:
            Transform composition for training.
        """
        if mean is None:
            mean = STL10DatasetLoader.DEFAULT_MEAN
        if std is None:
            std = STL10DatasetLoader.DEFAULT_STD
        
        if use_augmentation:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.33))
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        
        return transform
    
    @staticmethod
    def get_vit_val_transform(
        image_size: int = 224,
        mean: Optional[list] = None,
        std: Optional[list] = None
    ) -> transforms.Compose:
        """
        Get ViT-specific validation/test transforms (no augmentation).
        
        Args:
            image_size: Target image size (default 224x224 for ViT).
            mean: Normalization mean values. If None, uses ImageNet statistics.
            std: Normalization std values. If None, uses ImageNet statistics.
        
        Returns:
            Transform composition for validation/test.
        """
        if mean is None:
            mean = STL10DatasetLoader.DEFAULT_MEAN
        if std is None:
            std = STL10DatasetLoader.DEFAULT_STD
        
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        return transform
    
    @staticmethod
    def get_contrastive_transform(
        image_size: int = 224,
        mean: Optional[list] = None,
        std: Optional[list] = None,
        use_strong_augmentation: bool = True
    ) -> transforms.Compose:
        """
        Get strong augmentation transforms for contrastive learning.
        
        This transform is designed for self-supervised contrastive learning
        (SimCLR, MoCo, etc.) and includes strong augmentations like:
        - RandomResizedCrop
        - RandomHorizontalFlip
        - ColorJitter
        - GaussianBlur
        - RandomSolarize
        
        Args:
            image_size: Target image size (default 224x224).
            mean: Normalization mean values. If None, uses ImageNet statistics.
            std: Normalization std values. If None, uses ImageNet statistics.
            use_strong_augmentation: If True, applies strong augmentation.
        
        Returns:
            Transform composition for contrastive learning.
        """
        if mean is None:
            mean = STL10DatasetLoader.DEFAULT_MEAN
        if std is None:
            std = STL10DatasetLoader.DEFAULT_STD
        
        if use_strong_augmentation:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.2, 1.0),  # Stronger crop range for contrastive learning
                    ratio=(0.75, 1.33)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([
                    transforms.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.4,
                        hue=0.1
                    )
                ], p=0.8),
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
                ], p=0.5),
                transforms.RandomApply([
                    transforms.RandomSolarize(threshold=128, p=1.0)
                ], p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.33))
            ])
        else:
            # Minimal augmentation for validation/evaluation
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        
        return transform
    
    def get_contrastive_data_loader(
        self,
        batch_size: int = 256,
        num_workers: int = 4,
        image_size: int = 224,
        use_strong_augmentation: bool = True,
        download: Optional[bool] = None,
        shuffle: bool = True,
        pin_memory: bool = True,
        split: str = "unlabeled"
    ) -> torch.utils.data.DataLoader:
        """
        Get data loader for contrastive learning (typically uses unlabeled data).
        
        Args:
            batch_size: Batch size (typically large for contrastive learning, e.g., 256).
            num_workers: Number of worker processes for data loading.
            image_size: Target image size (default 224x224).
            use_strong_augmentation: If True, applies strong augmentation.
            download: If True, downloads the dataset from the internet. If None, uses instance default.
            shuffle: If True, shuffles data.
            pin_memory: If True, pins memory for faster GPU transfer.
            split: Dataset split to use ('unlabeled', 'train', 'train+unlabeled').
        
        Returns:
            DataLoader for contrastive learning.
        """
        # Get contrastive transform
        transform = self.get_contrastive_transform(
            image_size=image_size,
            use_strong_augmentation=use_strong_augmentation
        )
        
        # Load dataset
        if split == "unlabeled":
            dataset = self.get_unlabeled_dataset(transform=transform, download=download)
        else:
            dataset = self.get_dataset(split=split, transform=transform, download=download)
        
        # Create data loader
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True  # Important for contrastive learning to have consistent batch sizes
        )
        
        return loader
    
    def get_vit_data_loaders(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224,
        use_augmentation: bool = True,
        download: Optional[bool] = None,
        shuffle_train: bool = True,
        pin_memory: bool = True,
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        Get STL-10 data loaders with ViT-specific transforms.
        
        Args:
            batch_size: Batch size for data loaders.
            num_workers: Number of worker processes for data loading.
            image_size: Target image size (default 224x224 for ViT).
            use_augmentation: If True, applies data augmentation to training data.
            download: If True, downloads the dataset from the internet. If None, uses instance default.
            shuffle_train: If True, shuffles training data.
            pin_memory: If True, pins memory for faster GPU transfer.
        
        Returns:
            Tuple of (train_loader, test_loader).
        """
        # Get ViT-specific transforms
        train_transform = self.get_vit_train_transform(
            image_size=image_size,
            use_augmentation=use_augmentation
        )
        test_transform = self.get_vit_val_transform(image_size=image_size)
        
        return self.get_data_loaders(
            batch_size=batch_size,
            num_workers=num_workers,
            train_transform=train_transform,
            test_transform=test_transform,
            download=download,
            shuffle_train=shuffle_train,
            pin_memory=pin_memory
        )
    
    def get_dataset(
        self,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        target_transform: Optional[transforms.Compose] = None,
        download: Optional[bool] = None,
    ) -> datasets.STL10:
        """
        Load STL-10 dataset from torchvision.
        
        Args:
            split: One of 'train', 'test', 'unlabeled', 'train+unlabeled', or 'test+unlabeled'.
                   Default is 'train'.
            transform: Optional transform to be applied on images. If None, uses default transform.
            target_transform: Optional transform to be applied on labels.
            download: If True, downloads the dataset from the internet. If None, uses instance default.
        
        Returns:
            STL10 dataset object.
        """
        if transform is None:
            transform = self.default_transform
        
        if download is None:
            download = self.download
        
        dataset = datasets.STL10(
            root=self.root,
            split=split,
            download=download,
            transform=transform,
            target_transform=target_transform
        )
        
        return dataset
    
    def get_train_dataset(
        self,
        transform: Optional[transforms.Compose] = None,
        target_transform: Optional[transforms.Compose] = None,
        download: Optional[bool] = None,
    ) -> datasets.STL10:
        """
        Load STL-10 training dataset.
        
        Args:
            transform: Optional transform to be applied on images. If None, uses default transform.
            target_transform: Optional transform to be applied on labels.
            download: If True, downloads the dataset from the internet. If None, uses instance default.
        
        Returns:
            STL10 training dataset object.
        """
        return self.get_dataset(
            split="train",
            transform=transform,
            target_transform=target_transform,
            download=download
        )
    
    def get_test_dataset(
        self,
        transform: Optional[transforms.Compose] = None,
        target_transform: Optional[transforms.Compose] = None,
        download: Optional[bool] = None,
    ) -> datasets.STL10:
        """
        Load STL-10 test dataset.
        
        Args:
            transform: Optional transform to be applied on images. If None, uses default transform.
            target_transform: Optional transform to be applied on labels.
            download: If True, downloads the dataset from the internet. If None, uses instance default.
        
        Returns:
            STL10 test dataset object.
        """
        return self.get_dataset(
            split="test",
            transform=transform,
            target_transform=target_transform,
            download=download
        )
    
    def get_unlabeled_dataset(
        self,
        transform: Optional[transforms.Compose] = None,
        download: Optional[bool] = None,
    ) -> datasets.STL10:
        """
        Load STL-10 unlabeled dataset.
        
        Args:
            transform: Optional transform to be applied on images. If None, uses default transform.
            download: If True, downloads the dataset from the internet. If None, uses instance default.
        
        Returns:
            STL10 unlabeled dataset object.
        """
        return self.get_dataset(
            split="unlabeled",
            transform=transform,
            download=download
        )
    
    def get_data_loaders(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        train_transform: Optional[transforms.Compose] = None,
        test_transform: Optional[transforms.Compose] = None,
        download: Optional[bool] = None,
        shuffle_train: bool = True,
        pin_memory: bool = True,
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        Get STL-10 train and test data loaders.
        
        Args:
            batch_size: Batch size for data loaders.
            num_workers: Number of worker processes for data loading.
            train_transform: Optional transform for training data. If None, uses default transform.
            test_transform: Optional transform for test data. If None, uses default transform.
            download: If True, downloads the dataset from the internet. If None, uses instance default.
            shuffle_train: If True, shuffles training data.
            pin_memory: If True, pins memory for faster GPU transfer.
        
        Returns:
            Tuple of (train_loader, test_loader).
        """
        if download is None:
            download = self.download
        
        # Load datasets
        train_dataset = self.get_train_dataset(
            transform=train_transform,
            download=download
        )
        
        test_dataset = self.get_test_dataset(
            transform=test_transform,
            download=download
        )
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        return train_loader, test_loader


if __name__ == "__main__":
    # Example usage
    print("Loading STL-10 dataset...")
    
    # Initialize the dataset loader
    loader = STL10DatasetLoader(root="./data", download=True)
    
    # Load train dataset
    train_dataset = loader.get_train_dataset()
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Number of classes: {train_dataset.classes}")
    
    # Load test dataset
    test_dataset = loader.get_test_dataset(download=False)
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Load unlabeled dataset
    unlabeled_dataset = loader.get_unlabeled_dataset(download=False)
    print(f"Unlabeled dataset size: {len(unlabeled_dataset)}")
    
    # Get data loaders
    train_loader, test_loader = loader.get_data_loaders(batch_size=32)
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Print a sample batch
    for images, labels in train_loader:
        print(f"\nSample batch shape: {images.shape}")
        print(f"Sample labels: {labels[:5]}")
        break

