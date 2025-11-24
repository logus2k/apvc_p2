"""
Deep Learning for Computer Vision - Project 2
Augmentation Analysis Tool

This module analyzes augmentation effects across different image dimensions.
Creates visualization matrices showing how each transformation affects images
of different dimensions, helping decide optimal augmentation parameters.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
from collections import defaultdict
from scipy import ndimage

from dlcv_p2_config import TRAIN_DIR, CLASS_NAMES


@dataclass
class AugmentationAnalysisConfig:
    """Configuration for augmentation analysis"""
    
    rotation_range: int = 10
    """Maximum rotation angle in degrees (±rotation_range)"""
    
    zoom_range: float = 0.1
    """Zoom range (1.0 ± zoom_range)"""
    
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    """Brightness adjustment range as (min, max) multipliers"""
    
    target_size: Tuple[int, int] = (224, 224)
    """Final target size after all transformations"""
    
    max_images_per_dimension: int = 10
    """Maximum number of images to show per dimension type"""
    
    output_dir: str = "./test_augmentations"
    """Directory to save visualization matrices"""
    
    # Binning configuration
    use_binning: bool = True
    """Whether to use dimension binning (groups similar dimensions)"""
    
    bin_tolerance_percent: float = 15.0
    """Percentage tolerance for dimension binning (only used if use_binning=True)"""
    
    min_images_per_bin: int = 5
    """Minimum images required in a bin to analyze it (only used if use_binning=True)"""
    
    max_bins_to_analyze: Optional[int] = None
    """Maximum number of bins to analyze (None = analyze all that meet min_images_per_bin)"""
    
    # Specific transformation values for visualization
    rotation_angle: int = 10
    """Specific rotation angle to apply (in degrees)"""
    
    zoom_factor: float = 1.1
    """Specific zoom factor to apply (1.0 = no zoom, <1.0 = zoom out, >1.0 = zoom in)"""
    
    brightness_factor: float = 1.15
    """Specific brightness factor to apply"""


class AugmentationAnalyzer:
    """
    Analyzes augmentation effects across different image dimensions.
    """
    
    def __init__(self, config: AugmentationAnalysisConfig):
        """
        Initialize analyzer with configuration.
        
        Args:
            config: AugmentationAnalysisConfig instance
        """
        self.config = config
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    def scan_dataset_dimensions(self) -> Dict[Tuple[int, int], List[str]]:
        """
        Scan dataset and group images by their dimensions.
        
        Returns:
            Dictionary mapping (width, height) to list of image paths
        """
        print("Scanning dataset for image dimensions...")
        
        dimension_groups = defaultdict(list)
        
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(TRAIN_DIR, class_name)
            if not os.path.exists(class_dir):
                continue
            
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_file in image_files:
                img_path = os.path.join(class_dir, img_file)
                
                # Read image to get dimensions
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    h, w = img.shape
                    dimension_groups[(w, h)].append(img_path)
        
        # Apply binning if enabled
        if self.config.use_binning:
            dimension_groups = self._bin_dimensions(dimension_groups)
        else:
            # Sort by frequency (most common dimensions first)
            dimension_groups = dict(sorted(
                dimension_groups.items(),
                key=lambda x: len(x[1]),
                reverse=True
            ))
        
        print(f"\nFound {len(dimension_groups)} dimension groups:")
        for (w, h), paths in list(dimension_groups.items())[:10]:
            print(f"  {w}×{h}: {len(paths)} images")
        if len(dimension_groups) > 10:
            print(f"  ... and {len(dimension_groups) - 10} more groups")
        
        return dimension_groups
    
    def _bin_dimensions(self, dimension_groups: Dict[Tuple[int, int], List[str]]) -> Dict[Tuple[int, int], List[str]]:
        """
        Bin dimensions using tolerance percentage.
        
        Args:
            dimension_groups: Original dimension groups
        
        Returns:
            Binned dimension groups
        """
        print(f"\nApplying binning with {self.config.bin_tolerance_percent}% tolerance...")
        
        # Get all dimensions and their image paths
        all_dimensions = []
        dimension_to_paths = {}
        
        for (w, h), paths in dimension_groups.items():
            for path in paths:
                all_dimensions.append((w, h))
                dimension_to_paths[(w, h)] = dimension_to_paths.get((w, h), []) + [path]
        
        # Create bins
        bins = []
        used_dimensions = set()
        
        # Sort by area for systematic processing
        dimensions_by_area = sorted(set(all_dimensions), key=lambda d: d[0] * d[1])
        
        tolerance_fraction = self.config.bin_tolerance_percent / 100.0
        
        for w_center, h_center in dimensions_by_area:
            if (w_center, h_center) in used_dimensions:
                continue
            
            # Define tolerance
            w_tolerance = w_center * tolerance_fraction
            h_tolerance = h_center * tolerance_fraction
            w_min, w_max = w_center - w_tolerance, w_center + w_tolerance
            h_min, h_max = h_center - h_tolerance, h_center + h_tolerance
            
            # Collect all dimensions in this bin
            bin_paths = []
            for (w, h), paths in dimension_groups.items():
                if (w, h) in used_dimensions:
                    continue
                if w_min <= w <= w_max and h_min <= h <= h_max:
                    bin_paths.extend(paths)
                    used_dimensions.add((w, h))
            
            if len(bin_paths) > 0:
                bins.append({
                    'center': (w_center, h_center),
                    'paths': bin_paths
                })
        
        # Filter by minimum images
        bins = [b for b in bins if len(b['paths']) >= self.config.min_images_per_bin]
        
        # Sort by count (most common first)
        bins = sorted(bins, key=lambda x: len(x['paths']), reverse=True)
        
        # Limit number of bins if specified
        if self.config.max_bins_to_analyze is not None:
            bins = bins[:self.config.max_bins_to_analyze]
        
        # Convert back to dictionary format
        binned_groups = {}
        for bin_info in bins:
            binned_groups[bin_info['center']] = bin_info['paths']
        
        print(f"Created {len(binned_groups)} bins (filtered by min_images_per_bin={self.config.min_images_per_bin})")
        
        return binned_groups
    
    def apply_horizontal_crop(self, image: np.ndarray) -> np.ndarray:
        """
        Apply horizontal center crop (only if width > height).
        
        Args:
            image: Input image
        
        Returns:
            Cropped image (square) or original if already square/portrait
        """
        h, w = image.shape[:2]
        
        if w > h:
            # Crop horizontally to make square
            x_offset = (w - h) // 2
            return image[:, x_offset:x_offset+h]
        else:
            # Return original if square or portrait
            return image
    
    def apply_vertical_crop(self, image: np.ndarray) -> np.ndarray:
        """
        Apply vertical center crop (only if height > width).
        
        Args:
            image: Input image
        
        Returns:
            Cropped image (square) or original if already square/landscape
        """
        h, w = image.shape[:2]
        
        if h > w:
            # Crop vertically to make square
            y_offset = (h - w) // 2
            return image[y_offset:y_offset+w, :]
        else:
            # Return original if square or landscape
            return image
    
    def apply_zoom(self, image: np.ndarray) -> np.ndarray:
        """
        Apply zoom transformation.
        
        Args:
            image: Input image
        
        Returns:
            Zoomed image
        """
        h, w = image.shape[:2]
        zoom_factor = self.config.zoom_factor
        
        # Calculate new dimensions
        new_h = int(h * zoom_factor)
        new_w = int(w * zoom_factor)
        
        # Resize image
        zoomed = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Crop or pad to original size
        if zoom_factor < 1.0:
            # Zoomed out - need to pad
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            result = np.zeros((h, w), dtype=image.dtype)
            result[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = zoomed
            return result
        else:
            # Zoomed in - need to crop
            crop_h = (new_h - h) // 2
            crop_w = (new_w - w) // 2
            return zoomed[crop_h:crop_h+h, crop_w:crop_w+w]
    
    def apply_rotation(self, image: np.ndarray) -> np.ndarray:
        """
        Apply rotation transformation.
        
        Args:
            image: Input image
        
        Returns:
            Rotated image
        """
        angle = self.config.rotation_angle
        rotated = ndimage.rotate(image, angle, reshape=False, mode='constant', cval=0)
        return rotated
    
    def apply_brightness(self, image: np.ndarray) -> np.ndarray:
        """
        Apply brightness transformation.
        
        Args:
            image: Input image
        
        Returns:
            Brightness-adjusted image
        """
        brightness_factor = self.config.brightness_factor
        adjusted = np.clip(image.astype(float) * brightness_factor, 0, 255).astype(image.dtype)
        return adjusted
    
    def apply_all_transformations(self, image: np.ndarray) -> np.ndarray:
        """
        Apply all transformations: crop + zoom + brightness + resize.
        
        Note: Rotation is NOT included in final to avoid black corners obscuring the image.
        
        Args:
            image: Input image
        
        Returns:
            Fully transformed image at target size
        """
        # Step 1: Crop to square
        h, w = image.shape[:2]
        if w > h:
            result = self.apply_horizontal_crop(image)
        elif h > w:
            result = self.apply_vertical_crop(image)
        else:
            result = image.copy()
        
        # Step 2: Apply zoom
        result = self.apply_zoom(result)
        
        # Step 3: Apply brightness
        result = self.apply_brightness(result)
        
        # Step 4: Resize to target size
        result = cv2.resize(result, self.config.target_size, interpolation=cv2.INTER_LANCZOS4)
        
        return result
    
    def create_dimension_matrix(
        self,
        dimension: Tuple[int, int],
        image_paths: List[str]
    ) -> None:
        """
        Create visualization matrix for a specific dimension type.
        
        Args:
            dimension: (width, height) tuple
            image_paths: List of image paths with this dimension
        """
        w, h = dimension
        
        # Select images (up to max_images_per_dimension)
        selected_paths = image_paths[:self.config.max_images_per_dimension]
        n_images = len(selected_paths)
        
        if n_images == 0:
            return
        
        # Load images
        images = []
        for path in selected_paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
        
        if len(images) == 0:
            return
        
        # Create figure: n_images rows × 7 columns
        fig, axes = plt.subplots(len(images), 7, figsize=(21, 3 * len(images)))
        fig.suptitle(
            f'Augmentation Analysis: {w}×{h} ({len(image_paths)} images in dataset)',
            fontsize=16,
            fontweight='bold'
        )
        
        # Handle single image case
        if len(images) == 1:
            axes = [axes]
        
        # Column titles
        col_titles = [
            'Original',
            'H-Crop',
            'V-Crop',
            f'Zoom\n({self.config.zoom_factor:.2f}x)',
            f'Rotation\n({self.config.rotation_angle}°)',
            f'Brightness\n({self.config.brightness_factor:.2f}x)',
            f'Final\n({self.config.target_size[0]}×{self.config.target_size[1]})'
        ]
        
        # Process each image
        for row_idx, img in enumerate(images):
            # Column 0: Original
            axes[row_idx][0].imshow(img, cmap='gray')
            axes[row_idx][0].set_ylabel(f'Image {row_idx+1}', fontsize=10, fontweight='bold')
            if row_idx == 0:
                axes[row_idx][0].set_title(col_titles[0], fontsize=11, fontweight='bold')
            axes[row_idx][0].axis('off')
            
            # Column 1: Horizontal crop
            h_crop = self.apply_horizontal_crop(img)
            axes[row_idx][1].imshow(h_crop, cmap='gray')
            if row_idx == 0:
                axes[row_idx][1].set_title(col_titles[1], fontsize=11, fontweight='bold')
            axes[row_idx][1].axis('off')
            
            # Column 2: Vertical crop
            v_crop = self.apply_vertical_crop(img)
            axes[row_idx][2].imshow(v_crop, cmap='gray')
            if row_idx == 0:
                axes[row_idx][2].set_title(col_titles[2], fontsize=11, fontweight='bold')
            axes[row_idx][2].axis('off')
            
            # Column 3: Zoom
            zoomed = self.apply_zoom(img)
            axes[row_idx][3].imshow(zoomed, cmap='gray')
            if row_idx == 0:
                axes[row_idx][3].set_title(col_titles[3], fontsize=11, fontweight='bold')
            axes[row_idx][3].axis('off')
            
            # Column 4: Rotation
            rotated = self.apply_rotation(img)
            axes[row_idx][4].imshow(rotated, cmap='gray')
            if row_idx == 0:
                axes[row_idx][4].set_title(col_titles[4], fontsize=11, fontweight='bold')
            axes[row_idx][4].axis('off')
            
            # Column 5: Brightness
            brightened = self.apply_brightness(img)
            axes[row_idx][5].imshow(brightened, cmap='gray')
            if row_idx == 0:
                axes[row_idx][5].set_title(col_titles[5], fontsize=11, fontweight='bold')
            axes[row_idx][5].axis('off')
            
            # Column 6: Final (all transformations)
            final = self.apply_all_transformations(img)
            axes[row_idx][6].imshow(final, cmap='gray')
            if row_idx == 0:
                axes[row_idx][6].set_title(col_titles[6], fontsize=11, fontweight='bold')
            axes[row_idx][6].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(
            self.config.output_dir,
            f'dimension_{w}x{h}_matrix.png'
        )
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {output_path}")
    
    def analyze_all_dimensions(self) -> None:
        """
        Analyze augmentations for all dimension types in dataset.
        """
        print("\n" + "=" * 60)
        print("AUGMENTATION ANALYSIS")
        print("=" * 60)
        
        # Scan dataset
        dimension_groups = self.scan_dataset_dimensions()
        
        # Create matrix for each dimension type
        print(f"\nCreating visualization matrices...")
        print(f"Output directory: {self.config.output_dir}")
        print()
        
        for dimension, image_paths in dimension_groups.items():
            self.create_dimension_matrix(dimension, image_paths)
        
        print("\n" + "=" * 60)
        print(f"Analysis complete! Generated {len(dimension_groups)} matrices.")
        print(f"Check {self.config.output_dir}/ for visualizations")
        print("=" * 60)


def analyze_augmentations(config: Optional[AugmentationAnalysisConfig] = None) -> None:
    """
    Main function to analyze augmentations across all dimension types.
    
    Can be called from command line or imported in notebook.
    
    Args:
        config: Optional AugmentationAnalysisConfig. If None, uses defaults.
    """
    if config is None:
        config = AugmentationAnalysisConfig()
    
    analyzer = AugmentationAnalyzer(config)
    analyzer.analyze_all_dimensions()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze augmentation effects across different image dimensions',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--use-binning',
        type=lambda x: x.lower() in ('true', 'yes', '1'),
        default=True,
        help='Whether to use dimension binning'
    )
    
    parser.add_argument(
        '--tolerance',
        type=float,
        default=15.0,
        help='Percentage tolerance for dimension binning'
    )
    
    parser.add_argument(
        '--min-images',
        type=int,
        default=5,
        help='Minimum images required per bin'
    )
    
    parser.add_argument(
        '--max-bins',
        type=int,
        default=0,
        help='Maximum number of bins to analyze (0 = no limit)'
    )
    
    parser.add_argument(
        '--rotation',
        type=int,
        default=10,
        help='Rotation angle to show in visualization (degrees)'
    )
    
    parser.add_argument(
        '--zoom',
        type=float,
        default=1.1,
        help='Zoom factor to show in visualization'
    )
    
    parser.add_argument(
        '--brightness',
        type=float,
        default=1.15,
        help='Brightness factor to show in visualization'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./test_augmentations',
        help='Directory to save visualization matrices'
    )
    
    args = parser.parse_args()
    
    # Create configuration from arguments
    config = AugmentationAnalysisConfig(
        use_binning=args.use_binning,
        bin_tolerance_percent=args.tolerance,
        min_images_per_bin=args.min_images,
        max_bins_to_analyze=args.max_bins if args.max_bins > 0 else None,
        rotation_angle=args.rotation,
        zoom_factor=args.zoom,
        brightness_factor=args.brightness,
        output_dir=args.output_dir
    )
    
    print("Running augmentation analysis with configuration:")
    print(f"  Binning: {config.use_binning}")
    if config.use_binning:
        print(f"  Tolerance: {config.bin_tolerance_percent}%")
        print(f"  Min images per bin: {config.min_images_per_bin}")
        print(f"  Max bins to analyze: {config.max_bins_to_analyze if config.max_bins_to_analyze else 'all'}")
    print(f"  Rotation: {config.rotation_angle}°")
    print(f"  Zoom: {config.zoom_factor:.2f}x")
    print(f"  Brightness: {config.brightness_factor:.2f}x")
    print(f"  Output directory: {config.output_dir}")
    print()
    
    analyze_augmentations(config)
