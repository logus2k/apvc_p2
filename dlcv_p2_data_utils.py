"""
Deep Learning for Computer Vision - Project 2
Data Utilities

This module provides utilities for:
- Loading and analyzing the chest X-ray dataset
- Computing class distributions and weights
- Creating train/validation/test splits
- Dataset statistics and inspection
"""

import os
import numpy as np
import cv2
from typing import Tuple, Dict, List, Optional
from sklearn.utils import class_weight as sk_class_weight

from dlcv_p2_config import TRAIN_DIR, TEST_DIR, CLASS_NAMES, SEED, VAL_SPLIT


class DatasetAnalyzer:
    """
    Analyzes the chest X-ray dataset and provides statistics.
    
    This class loads images in their original resolution and provides
    information about:
    - Image sizes and aspect ratios
    - Class distribution
    - Dataset statistics
    """
    
    def __init__(self, train_dir: str = TRAIN_DIR, test_dir: str = TEST_DIR):
        """
        Initialize dataset analyzer.
        
        Args:
            train_dir: Path to training data directory
            test_dir: Path to test data directory
        """
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.class_names = CLASS_NAMES
        
        # Will be populated by analyze methods
        self.train_stats: Optional[Dict] = None
        self.test_stats: Optional[Dict] = None
        self.class_weights: Optional[Dict[int, float]] = None
    
    def _get_image_paths(self, root_dir: str) -> Dict[str, List[str]]:
        """
        Get all image paths organized by class.
        
        Args:
            root_dir: Root directory containing class subdirectories
        
        Returns:
            Dictionary mapping class names to lists of image paths
        """
        image_paths = {class_name: [] for class_name in self.class_names}
        
        for class_name in self.class_names:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory not found: {class_dir}")
                continue
            
            # Get all image files
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths[class_name].append(os.path.join(class_dir, filename))
        
        return image_paths
    
    def _analyze_images(self, image_paths: Dict[str, List[str]]) -> Dict:
        """
        Analyze images and compute statistics.
        
        Args:
            image_paths: Dictionary mapping class names to image paths
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'class_counts': {},
            'image_sizes': [],
            'widths': [],
            'heights': [],
            'aspect_ratios': [],
            'total_images': 0
        }
        
        print("Analyzing images...")
        
        for class_name, paths in image_paths.items():
            stats['class_counts'][class_name] = len(paths)
            stats['total_images'] += len(paths)
            
            print(f"  {class_name}: {len(paths)} images")
            
            # Sample images to get size distribution (analyze all if dataset is small)
            sample_size = min(100, len(paths))
            sample_paths = np.random.choice(paths, sample_size, replace=False)
            
            for img_path in sample_paths:
                # Read image to get dimensions (faster than loading full image)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    h, w = img.shape
                    stats['image_sizes'].append((w, h))
                    stats['widths'].append(w)
                    stats['heights'].append(h)
                    stats['aspect_ratios'].append(w / h)
        
        # Compute summary statistics
        if stats['widths']:
            stats['mean_width'] = np.mean(stats['widths'])
            stats['mean_height'] = np.mean(stats['heights'])
            stats['std_width'] = np.std(stats['widths'])
            stats['std_height'] = np.std(stats['heights'])
            stats['min_width'] = np.min(stats['widths'])
            stats['max_width'] = np.max(stats['widths'])
            stats['min_height'] = np.min(stats['heights'])
            stats['max_height'] = np.max(stats['heights'])
            stats['mean_aspect_ratio'] = np.mean(stats['aspect_ratios'])
        
        return stats
    
    def analyze_train_data(self) -> Dict:
        """
        Analyze training data.
        
        Returns:
            Dictionary with training data statistics
        """
        title = "TRAINING DATA ANALYSIS"        
        print("\n" + "=" * len(title))
        print(title)
        print("=" * len(title))
        
        image_paths = self._get_image_paths(self.train_dir)
        self.train_stats = self._analyze_images(image_paths)
        self.train_stats['image_paths'] = image_paths
        
        return self.train_stats
    
    def analyze_test_data(self) -> Dict:
        """
        Analyze test data.
        
        Returns:
            Dictionary with test data statistics
        """
        title = "TEST DATA ANALYSIS"
        print("\n" + "=" * len(title))
        print(title)
        print("=" * len(title))
        
        image_paths = self._get_image_paths(self.test_dir)
        self.test_stats = self._analyze_images(image_paths)
        self.test_stats['image_paths'] = image_paths
        
        return self.test_stats
    
    def compute_class_weights(self) -> Dict[int, float]:
        """
        Compute balanced class weights from training data.
        
        Returns:
            Dictionary mapping class indices to weights
        """
        if self.train_stats is None:
            self.analyze_train_data()
        
        # Verify stats were loaded
        if self.train_stats is None:
            raise RuntimeError("Failed to load training data statistics")
        
        # Create label array
        labels = []
        for class_idx, class_name in enumerate(self.class_names):
            count = self.train_stats['class_counts'][class_name]
            labels.extend([class_idx] * count)
        
        labels = np.array(labels)
        classes = np.arange(len(self.class_names))
        
        # Compute balanced weights
        weights = sk_class_weight.compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=labels
        )
        
        self.class_weights = {int(cls): float(w) for cls, w in zip(classes, weights)}
        
        title = "CLASS WEIGHTS"
        print("\n" + "=" * len(title))
        print(title)
        print("=" * len(title))
        for class_idx, class_name in enumerate(self.class_names):
            print(f"  {class_name} (class {class_idx}): {self.class_weights[class_idx]:.4f}")
        
        return self.class_weights
    
    def print_summary(self) -> None:
        """Print comprehensive dataset summary."""
        if self.train_stats is None:
            self.analyze_train_data()
        if self.test_stats is None:
            self.analyze_test_data()
        if self.class_weights is None:
            self.compute_class_weights()
        
        # Verify stats were successfully loaded
        if self.train_stats is None or self.test_stats is None:
            print("Error: Could not load dataset statistics")
            return
        
        title = "DATASET SUMMARY"
        print("\n" + "=" * len(title))
        print(title)
        print("=" * len(title))
        
        title = "Class Distribution:"
        print("\n" + title)
        print("-" * len(title))
        total_train = self.train_stats['total_images']
        total_test = self.test_stats['total_images']
        
        for class_name in self.class_names:
            train_count = self.train_stats['class_counts'][class_name]
            test_count = self.test_stats['class_counts'][class_name]
            train_pct = 100 * train_count / total_train
            test_pct = 100 * test_count / total_test
            
            print(f"  {class_name}:")
            print(f"    Train: {train_count:5d} ({train_pct:5.1f}%)")
            print(f"    Test:  {test_count:5d} ({test_pct:5.1f}%)")
        
        print(f"\n  Total:")
        print(f"    Train: {total_train:5d}")
        print(f"    Test:  {total_test:5d}")
        
        title = "Image Size Statistics (sampled):"
        print("\n" + title)
        print("-" * len(title))
        print(f"  Width:  {self.train_stats['mean_width']:.0f} ± {self.train_stats['std_width']:.0f} px")
        print(f"          Range: [{self.train_stats['min_width']:.0f}, {self.train_stats['max_width']:.0f}] px")
        print(f"  Height: {self.train_stats['mean_height']:.0f} ± {self.train_stats['std_height']:.0f} px")
        print(f"          Range: [{self.train_stats['min_height']:.0f}, {self.train_stats['max_height']:.0f}] px")
        print(f"  Mean aspect ratio: {self.train_stats['mean_aspect_ratio']:.2f}")
        
        print("\n" + "=" * len(title))


def load_sample_images(
    dataset_dir: str,
    class_names: List[str] = CLASS_NAMES,
    samples_per_class: int = 2,
    seed: int = SEED
) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """
    Load sample images from each class for visualization.
    
    Args:
        dataset_dir: Root directory containing class subdirectories
        class_names: List of class names
        samples_per_class: Number of samples to load per class
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (images, labels, label_names)
    """
    np.random.seed(seed)
    
    images = []
    labels = []
    label_names = []
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_dir, class_name)
        
        if not os.path.exists(class_dir):
            print(f"Warning: Directory not found: {class_dir}")
            continue
        
        # Get all image files
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Sample random images
        if len(image_files) < samples_per_class:
            selected_files = image_files
        else:
            selected_files = np.random.choice(image_files, samples_per_class, replace=False)
        
        # Load images
        for filename in selected_files:
            img_path = os.path.join(class_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                images.append(img)
                labels.append(class_idx)
                label_names.append(class_name)
    
    return images, labels, label_names


def get_dataset_info(
    train_dir: str = TRAIN_DIR,
    test_dir: str = TEST_DIR
) -> Dict:
    """
    Get comprehensive dataset information.
    
    Args:
        train_dir: Path to training data
        test_dir: Path to test data
    
    Returns:
        Dictionary with all dataset information
    """
    analyzer = DatasetAnalyzer(train_dir, test_dir)
    
    # Analyze both splits
    train_stats = analyzer.analyze_train_data()
    test_stats = analyzer.analyze_test_data()
    class_weights = analyzer.compute_class_weights()
    
    # Print summary
    analyzer.print_summary()
    
    return {
        'train_stats': train_stats,
        'test_stats': test_stats,
        'class_weights': class_weights,
        'class_names': CLASS_NAMES
    }


def create_validation_split(
    image_paths: Dict[str, List[str]],
    val_split: float = VAL_SPLIT,
    seed: int = SEED
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Split training data into train and validation sets.
    
    Args:
        image_paths: Dictionary mapping class names to image paths
        val_split: Fraction of data to use for validation
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_paths, val_paths) dictionaries
    """
    np.random.seed(seed)
    
    train_paths = {class_name: [] for class_name in image_paths.keys()}
    val_paths = {class_name: [] for class_name in image_paths.keys()}
    
    for class_name, paths in image_paths.items():
        # Shuffle paths
        paths_array = np.array(paths)
        np.random.shuffle(paths_array)
        
        # Split
        n_val = int(len(paths_array) * val_split)
        val_paths[class_name] = paths_array[:n_val].tolist()
        train_paths[class_name] = paths_array[n_val:].tolist()
    
    # Print split info
    title = "TRAIN/VALIDATION SPLIT"
    print("\n" + "=" * len(title))
    print(title)
    print("=" * len(title))
    for class_name in image_paths.keys():
        print(f"  {class_name}:")
        print(f"    Train: {len(train_paths[class_name])}")
        print(f"    Val:   {len(val_paths[class_name])}")
    
    return train_paths, val_paths


if __name__ == "__main__":
    # Test data utilities
    print("Testing data utilities...\n")
    
    # Check if data directories exist
    if not os.path.exists(TRAIN_DIR):
        print(f"Training directory not found: {TRAIN_DIR}")
        print("Please ensure the chest_xray dataset is in the correct location.")
    else:
        # Get dataset info
        dataset_info = get_dataset_info()
        
        # Load sample images
        print("\nLoading sample images...")
        images, labels, label_names = load_sample_images(
            TRAIN_DIR,
            samples_per_class=2
        )
        print(f"Loaded {len(images)} sample images")
        
        # Test validation split
        if dataset_info['train_stats']['image_paths']:
            train_paths, val_paths = create_validation_split(
                dataset_info['train_stats']['image_paths']
            )
