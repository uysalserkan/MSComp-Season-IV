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

