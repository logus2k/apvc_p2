"""
Deep Learning for Computer Vision - Project 2
Visualization Utilities

This module provides visualization functions for:
- Preprocessing pipeline stages (raw, enhanced, cropped, resized)
- Data distribution and class balance
- Sample images from dataset
- Augmentation effects
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union, Sequence
import cv2

from dlcv_p2_preprocessor import ImagePreprocessor
from dlcv_p2_preprocessing_config import PreprocessingConfig


# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


def visualize_preprocessing_pipeline(
    images: List[np.ndarray],
    preprocessor: ImagePreprocessor,
    titles: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (16, 16)
) -> None:
    """
    Visualize preprocessing pipeline stages for multiple images in a 4x4 matrix.
    
    Shows 4 stages for each of 4 images:
    - Original
    - Enhanced (CLAHE, denoising, edge enhancement)
    - Cropped (aspect ratio handling)
    - Final (resized + color converted)
    
    Args:
        images: List of 4 input images (numpy arrays)
        preprocessor: ImagePreprocessor instance
        titles: Optional list of titles for each image
        figsize: Figure size
    """
    if len(images) != 4:
        raise ValueError("Exactly 4 images required for 4x4 visualization")
    
    if titles is None:
        titles = [f"Image {i+1}" for i in range(4)]
    
    # Stage names
    stage_names = ['Original', 'Enhanced', 'Cropped', 'Final']
    
    # Create figure
    fig, axes = plt.subplots(4, 4, figsize=figsize)
    fig.suptitle(
        f'Preprocessing Pipeline: {preprocessor.config.name}',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )
    
    # Process each image
    for img_idx, image in enumerate(images):
        # Get pipeline stages
        stages = preprocessor.get_pipeline_stages(image)
        
        # Plot each stage
        for stage_idx, stage_name in enumerate(['original', 'enhanced', 'cropped', 'final']):
            ax = axes[img_idx, stage_idx]
            stage_img = stages[stage_name]
            
            # Handle grayscale vs RGB
            if len(stage_img.shape) == 2:
                ax.imshow(stage_img, cmap='gray')
            elif stage_img.shape[2] == 1:
                ax.imshow(stage_img.squeeze(), cmap='gray')
            else:
                ax.imshow(stage_img)
            
            # Add title (stage name on first row, image title on first column)
            if img_idx == 0:
                ax.set_title(stage_names[stage_idx], fontsize=12, fontweight='bold')
            if stage_idx == 0:
                ax.set_ylabel(titles[img_idx], fontsize=11, fontweight='bold')
            
            # Show image shape
            shape_text = f"{stage_img.shape[0]}×{stage_img.shape[1]}"
            if len(stage_img.shape) == 3:
                shape_text += f"×{stage_img.shape[2]}"
            ax.text(
                0.02, 0.98, shape_text,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
            )
            
            ax.axis('off')
    
    plt.tight_layout()


def compare_preprocessing_configs(
    image: np.ndarray,
    configs: List[PreprocessingConfig],
    config_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (16, 4)
) -> None:
    """
    Compare different preprocessing configurations side by side.
    
    Args:
        image: Input image
        configs: List of PreprocessingConfig instances to compare
        config_names: Optional list of configuration names
        figsize: Figure size
    """
    n_configs = len(configs)
    
    if config_names is None:
        config_names = [config.name for config in configs]
    
    fig, axes = plt.subplots(1, n_configs + 1, figsize=figsize)
    fig.suptitle('Preprocessing Configuration Comparison', fontsize=14, fontweight='bold')
    
    # Show original
    axes[0].imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    axes[0].set_title('Original', fontsize=11, fontweight='bold')
    axes[0].axis('off')
    
    # Show each configuration result
    for idx, (config, name) in enumerate(zip(configs, config_names), start=1):
        preprocessor = ImagePreprocessor(config)
        processed = preprocessor.preprocess(image)
        
        if len(processed.shape) == 2:
            axes[idx].imshow(processed, cmap='gray')
        elif processed.shape[2] == 1:
            axes[idx].imshow(processed.squeeze(), cmap='gray')
        else:
            axes[idx].imshow(processed)
        
        axes[idx].set_title(name, fontsize=11, fontweight='bold')
        axes[idx].axis('off')
    
    plt.tight_layout()


def visualize_augmentation_effects(
    image: np.ndarray,
    preprocessor: ImagePreprocessor,
    n_samples: int = 9,
    figsize: Tuple[int, int] = (12, 12)
) -> None:
    """
    Visualize data augmentation effects by showing multiple augmented versions.
    
    Args:
        image: Input image
        preprocessor: ImagePreprocessor with augmentation configured
        n_samples: Number of augmented samples to show (should be perfect square)
        figsize: Figure size
    """
    # Get augmentation generator
    datagen = preprocessor.get_augmentation_generator()
    
    # Preprocess image first
    processed = preprocessor.preprocess(image)
    
    # Expand dimensions for generator (needs batch dimension)
    if len(processed.shape) == 2:
        processed = np.expand_dims(processed, axis=-1)
    processed = np.expand_dims(processed, axis=0)
    
    # Calculate grid size
    grid_size = int(np.sqrt(n_samples))
    if grid_size * grid_size != n_samples:
        raise ValueError("n_samples must be a perfect square (e.g., 4, 9, 16)")
    
    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    fig.suptitle(
        f'Data Augmentation Effects\n{preprocessor.config.name}',
        fontsize=14,
        fontweight='bold'
    )
    
    # Generate and display augmented images with parameters
    # We need to manually apply transformations to track parameters
    import random
    from scipy import ndimage
    
    for i in range(grid_size):
        for j in range(grid_size):
            ax = axes[i, j] if grid_size > 1 else axes
            
            # Get base image
            aug_img = processed[0].copy()
            
            # Apply random transformations and track them
            params = []
            
            # Rotation
            if preprocessor.config.rotation_range > 0:
                angle = random.uniform(-preprocessor.config.rotation_range, 
                                     preprocessor.config.rotation_range)
                if abs(angle) > 0.5:  # Only mention if significant
                    aug_img = ndimage.rotate(aug_img, angle, reshape=False, mode='constant', cval=0)
                    params.append(f"Rot: {angle:.1f}°")
            
            # Horizontal flip
            if preprocessor.config.horizontal_flip and random.random() > 0.5:
                aug_img = np.fliplr(aug_img)
                params.append("H-Flip")
            
            # Brightness
            if preprocessor.config.brightness_range != (1.0, 1.0):
                brightness_factor = random.uniform(*preprocessor.config.brightness_range)
                if abs(brightness_factor - 1.0) > 0.05:  # Only mention if significant
                    aug_img = np.clip(aug_img * brightness_factor, 0, 255)
                    params.append(f"Bright: {brightness_factor:.2f}x")
            
            # Zoom
            if preprocessor.config.zoom_range > 0:
                zoom_factor = random.uniform(1.0 - preprocessor.config.zoom_range,
                                            1.0 + preprocessor.config.zoom_range)
                if abs(zoom_factor - 1.0) > 0.02:  # Only mention if significant
                    params.append(f"Zoom: {zoom_factor:.2f}x")
            
            # Shifts
            if preprocessor.config.width_shift_range > 0 or preprocessor.config.height_shift_range > 0:
                h_shift = random.uniform(-preprocessor.config.width_shift_range * 0.5,
                                        preprocessor.config.width_shift_range * 0.5)
                v_shift = random.uniform(-preprocessor.config.height_shift_range * 0.5,
                                        preprocessor.config.height_shift_range * 0.5)
                if abs(h_shift) > 0.02 or abs(v_shift) > 0.02:
                    if abs(h_shift) > 0.02:
                        params.append(f"H-Shift: {h_shift:.1%}")
                    if abs(v_shift) > 0.02:
                        params.append(f"V-Shift: {v_shift:.1%}")
            
            # Plot
            if len(aug_img.shape) == 2 or aug_img.shape[2] == 1:
                ax.imshow(aug_img.squeeze(), cmap='gray')
            else:
                ax.imshow(aug_img.astype('uint8'))
            
            # Set title with parameters
            if params:
                title = '\n'.join(params[:3])  # Max 3 lines to fit
                ax.set_title(title, fontsize=8)
            else:
                ax.set_title('Original', fontsize=8)
            
            ax.axis('off')
    
    plt.tight_layout()



def plot_class_distribution(
    class_counts: Dict[str, int],
    class_weights: Optional[Dict[int, float]] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot class distribution and weights.
    
    Args:
        class_counts: Dictionary mapping class names to counts
        class_weights: Optional dictionary mapping class indices to weights
        figsize: Figure size
    """
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Plot counts
    bars = ax1.bar(
        classes,
        counts,
        color='skyblue',
        alpha=0.7,
        label='Class Count',
        width=0.4
    )
    
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )
    
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=11)
    
    # Plot weights if provided
    if class_weights is not None:
        ax2 = ax1.twinx()
        weights = [class_weights[i] for i in range(len(classes))]
        
        ax2.plot(
            classes,
            weights,
            color='red',
            marker='o',
            linewidth=2,
            markersize=10,
            label='Class Weight'
        )
        
        ax2.set_ylabel('Class Weight', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=11)
    
    plt.title('Class Distribution and Weights\n', fontsize=14, fontweight='bold')
    plt.tight_layout()



def plot_image_size_distribution(
    image_sizes: List[Tuple[int, int]],
    figsize: Tuple[int, int] = (14, 5)
) -> None:
    """
    Plot distribution of image sizes in dataset.
    
    Args:
        image_sizes: List of (width, height) tuples
        figsize: Figure size
    """
    widths = [size[0] for size in image_sizes]
    heights = [size[1] for size in image_sizes]
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle('Image Size Distribution in Dataset', fontsize=14, fontweight='bold')
    
    # Width distribution
    axes[0].hist(widths, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Width (pixels)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title(f'Width Distribution\nMean: {np.mean(widths):.0f}px', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Height distribution
    axes[1].hist(heights, bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Height (pixels)', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title(f'Height Distribution\nMean: {np.mean(heights):.0f}px', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    # Scatter plot: width vs height
    axes[2].scatter(widths, heights, alpha=0.5, s=20)
    axes[2].set_xlabel('Width (pixels)', fontsize=11)
    axes[2].set_ylabel('Height (pixels)', fontsize=11)
    axes[2].set_title('Width vs Height', fontsize=11)
    axes[2].grid(True, alpha=0.3)
    
    # Add aspect ratio line
    max_dim = max(max(widths), max(heights))
    axes[2].plot([0, max_dim], [0, max_dim], 'r--', alpha=0.5, label='Square (1:1)')
    axes[2].legend()
    
    plt.tight_layout()



def visualize_sample_images(
    images: List[np.ndarray],
    labels: Sequence[Union[int, str]],
    class_names: Optional[List[str]] = None,
    n_cols: int = 4,
    figsize: Optional[Tuple[int, int]] = None
) -> None:
    """
    Visualize sample images from dataset in a grid.
    
    Args:
        images: List of images
        labels: List of labels (integers or strings)
        class_names: Optional class names for labels
        n_cols: Number of columns in grid
        figsize: Figure size (auto-calculated if None)
    """
    n_images = len(images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    if figsize is None:
        figsize = (4 * n_cols, 4 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle('Sample Images from Dataset', fontsize=14, fontweight='bold')
    
    # Flatten axes for easy iteration
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for idx, (image, label) in enumerate(zip(images, labels)):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        
        # Display image
        if len(image.shape) == 2:
            ax.imshow(image, cmap='gray')
        elif image.shape[2] == 1:
            ax.imshow(image.squeeze(), cmap='gray')
        else:
            ax.imshow(image)
        
        # Add label
        if class_names is not None:
            label_text = class_names[int(label)] if isinstance(label, (int, np.integer)) else label
        else:
            label_text = str(label)
        
        ax.set_title(label_text, fontsize=11, fontweight='bold')
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()



def plot_preprocessing_comparison_grid(
    images: List[np.ndarray],
    preprocessor1: ImagePreprocessor,
    preprocessor2: ImagePreprocessor,
    image_labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (16, 12)
) -> None:
    """
    Compare two preprocessing configurations side by side for multiple images.
    
    Args:
        images: List of input images
        preprocessor1: First preprocessor
        preprocessor2: Second preprocessor
        image_labels: Optional labels for images
        figsize: Figure size
    """
    n_images = len(images)
    
    if image_labels is None:
        image_labels = [f"Image {i+1}" for i in range(n_images)]
    
    fig, axes = plt.subplots(n_images, 3, figsize=figsize)
    fig.suptitle(
        f'Preprocessing Comparison\n{preprocessor1.config.name} vs {preprocessor2.config.name}',
        fontsize=14,
        fontweight='bold'
    )
    
    # Handle single image case
    if n_images == 1:
        axes = [axes]
    
    for idx, (image, label) in enumerate(zip(images, image_labels)):
        # Original
        axes[idx][0].imshow(image, cmap='gray' if len(image.shape) == 2 else None)
        axes[idx][0].set_ylabel(label, fontsize=11, fontweight='bold')
        if idx == 0:
            axes[idx][0].set_title('Original', fontsize=12, fontweight='bold')
        axes[idx][0].axis('off')
        
        # Preprocessor 1
        processed1 = preprocessor1.preprocess(image)
        if len(processed1.shape) == 2 or processed1.shape[2] == 1:
            axes[idx][1].imshow(processed1.squeeze(), cmap='gray')
        else:
            axes[idx][1].imshow(processed1)
        if idx == 0:
            axes[idx][1].set_title(preprocessor1.config.name, fontsize=12, fontweight='bold')
        axes[idx][1].axis('off')
        
        # Preprocessor 2
        processed2 = preprocessor2.preprocess(image)
        if len(processed2.shape) == 2 or processed2.shape[2] == 1:
            axes[idx][2].imshow(processed2.squeeze(), cmap='gray')
        else:
            axes[idx][2].imshow(processed2)
        if idx == 0:
            axes[idx][2].set_title(preprocessor2.config.name, fontsize=12, fontweight='bold')
        axes[idx][2].axis('off')
    
    plt.tight_layout()



if __name__ == "__main__":
    # Test visualizations with real data
    from dlcv_p2_preprocessing_config import get_baseline_config, get_enhanced_config
    from dlcv_p2_config import TRAIN_DIR, CLASS_NAMES
    import cv2
    
    print("Testing visualization utilities with real chest X-ray data...")
    print("Saving example visualizations to ./results/test_visualizations/")
    
    # Create output directory
    output_dir = "./results/test_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load real images from dataset
    print("\nLoading sample images from dataset...")
    real_images = []
    image_labels = []
    
    try:
        # Load 2 images from each class
        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = os.path.join(TRAIN_DIR, class_name)
            if os.path.exists(class_dir):
                image_files = [f for f in os.listdir(class_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:2]
                
                for img_file in image_files:
                    img_path = os.path.join(class_dir, img_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        real_images.append(img)
                        image_labels.append(class_name)
                        print(f"  Loaded: {class_name}/{img_file} - shape {img.shape}")
        
        if len(real_images) < 4:
            print("Warning: Could not load enough images, using dummy data")
            raise FileNotFoundError
            
    except (FileNotFoundError, Exception) as e:
        print(f"Could not load real images: {e}")
        print("Falling back to dummy data for testing...")
        np.random.seed(42)
        real_images = [
            np.random.randint(50, 200, (1470, 1033), dtype=np.uint8) for _ in range(4)
        ]
        image_labels = ['Sample 1', 'Sample 2', 'Sample 3', 'Sample 4']
    
    # Ensure we have exactly 4 images for the 4x4 grid
    display_images = real_images[:4]
    display_labels = image_labels[:4]
    
    # Test 1: Preprocessing pipeline visualization
    print("\n1. Testing preprocessing pipeline visualization...")
    baseline_config = get_baseline_config()
    baseline_preprocessor = ImagePreprocessor(baseline_config)
    
    visualize_preprocessing_pipeline(
        display_images,
        baseline_preprocessor,
        titles=display_labels
    )
    plt.savefig(os.path.join(output_dir, "01_preprocessing_pipeline.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_dir}/01_preprocessing_pipeline.png")
    
    # Test 2: Compare preprocessing configs
    print("\n2. Testing preprocessing config comparison...")
    baseline_config = get_baseline_config((224, 224))
    enhanced_config = get_enhanced_config((224, 224))
    
    compare_preprocessing_configs(
        display_images[0],
        [baseline_config, enhanced_config],
        config_names=['Baseline', 'CLAHE Enhanced']
    )
    plt.savefig(os.path.join(output_dir, "02_config_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_dir}/02_config_comparison.png")
    
    # Test 3: Augmentation effects
    print("\n3. Testing augmentation effects visualization...")
    visualize_augmentation_effects(
        display_images[0],
        baseline_preprocessor,
        n_samples=9
    )
    plt.savefig(os.path.join(output_dir, "03_augmentation_effects.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_dir}/03_augmentation_effects.png")
    
    # Test 4: Class distribution
    print("\n4. Testing class distribution plot...")
    class_counts = {'NORMAL': 1341, 'PNEUMONIA': 3875}
    class_weights = {0: 2.89, 1: 1.0}
    plot_class_distribution(class_counts, class_weights)
    plt.savefig(os.path.join(output_dir, "04_class_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_dir}/04_class_distribution.png")
    
    # Test 5: Image size distribution
    print("\n5. Testing image size distribution plot...")
    # Use actual image sizes from loaded images
    if len(real_images) > 4:
        image_sizes = [(img.shape[1], img.shape[0]) for img in real_images]
    else:
        # Generate fake sizes similar to real dataset if not enough real images
        image_sizes = [(np.random.randint(800, 2000), np.random.randint(800, 2000)) for _ in range(100)]
    plot_image_size_distribution(image_sizes)
    plt.savefig(os.path.join(output_dir, "05_image_size_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_dir}/05_image_size_distribution.png")
    
    # Test 6: Sample images grid
    print("\n6. Testing sample images visualization...")
    if len(real_images) >= 8:
        sample_images = real_images[:8]
        sample_labels = [0 if 'NORMAL' in lbl else 1 for lbl in image_labels[:8]]
    else:
        sample_images = display_images * 2  # Repeat to get 8
        sample_labels = [0, 1, 0, 1, 1, 0, 1, 0]
    
    visualize_sample_images(
        sample_images,
        sample_labels,
        class_names=['NORMAL', 'PNEUMONIA'],
        n_cols=4
    )
    plt.savefig(os.path.join(output_dir, "06_sample_images.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_dir}/06_sample_images.png")
    
    # Test 7: Preprocessing comparison grid
    print("\n7. Testing preprocessing comparison grid...")
    enhanced_preprocessor = ImagePreprocessor(enhanced_config)
    plot_preprocessing_comparison_grid(
        display_images[:3],
        baseline_preprocessor,
        enhanced_preprocessor,
        image_labels=display_labels[:3]
    )
    plt.savefig(os.path.join(output_dir, "07_preprocessing_comparison_grid.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_dir}/07_preprocessing_comparison_grid.png")
    
    print("\n" + "=" * 60)
    print("All visualization tests completed successfully!")
    print(f"The generated images were stored in {output_dir}")
    print("=" * 60)
