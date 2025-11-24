"""
Deep Learning for Computer Vision - Project 2
Preprocessing Configuration

This module defines the PreprocessingConfig dataclass that holds all
parameters for image preprocessing including enhancement, resizing,
augmentation, and color conversion.
"""

from dataclasses import dataclass, field
from typing import Tuple, Literal
import cv2


@dataclass
class PreprocessingConfig:
    """
    Configuration for image preprocessing pipeline.
    
    This dataclass holds all parameters that control image preprocessing,
    including enhancement, resizing, augmentation, and color conversion.
    All parameters can be modified for hyperparameter sweeps.
    """
    
    # =========================================================================
    # TARGET RESOLUTION
    # =========================================================================
    target_size: Tuple[int, int] = (224, 224)
    """Target image size after preprocessing (width, height)"""
    
    # =========================================================================
    # ASPECT RATIO HANDLING
    # =========================================================================
    aspect_handling: Literal['center_crop', 'pad'] = 'center_crop'
    """How to handle aspect ratio before resizing:
    - 'center_crop': Crop to square from center (loses edges)
    - 'pad': Pad to square with black borders (preserves all info)
    """
    
    resize_interpolation: int = cv2.INTER_LANCZOS4
    """Interpolation method for resizing. Options:
    - cv2.INTER_LANCZOS4: Best quality (slowest)
    - cv2.INTER_CUBIC: Good quality
    - cv2.INTER_LINEAR: Faster, lower quality
    """
    
    # =========================================================================
    # IMAGE ENHANCEMENT
    # =========================================================================
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    use_clahe: bool = False
    """Whether to apply CLAHE enhancement"""
    
    clahe_clip_limit: float = 2.0
    """CLAHE clip limit (threshold for contrast limiting).
    Higher values = more contrast. Typical range: 1.0-4.0
    """
    
    clahe_tile_size: Tuple[int, int] = (8, 8)
    """CLAHE tile grid size for adaptive histogram equalization.
    Smaller tiles = more local adaptation. Typical: (4,4) to (16,16)
    """
    
    # Denoising
    use_denoising: bool = False
    """Whether to apply denoising"""
    
    denoise_strength: float = 10.0
    """Denoising filter strength.
    Higher values = more smoothing. Typical range: 5.0-15.0
    """
    
    denoise_template_window_size: int = 7
    """Denoising template window size. Should be odd. Typical: 7"""
    
    denoise_search_window_size: int = 21
    """Denoising search window size. Should be odd. Typical: 21"""
    
    # Edge Enhancement
    use_edge_enhancement: bool = False
    """Whether to apply edge enhancement (unsharp masking)"""
    
    edge_enhancement_amount: float = 1.0
    """Edge enhancement strength. Typical range: 0.5-2.0"""
    
    edge_enhancement_radius: float = 1.0
    """Edge enhancement radius (Gaussian blur sigma). Typical: 1.0-2.0"""
    
    edge_enhancement_threshold: float = 0.0
    """Edge enhancement threshold (only enhance above this). Typical: 0.0-0.05"""
    
    # =========================================================================
    # DATA AUGMENTATION (training only)
    # =========================================================================
    rotation_range: int = 10
    """Maximum rotation angle in degrees (±rotation_range)"""
    
    width_shift_range: float = 0.1
    """Horizontal shift as fraction of width (±width_shift_range)"""
    
    height_shift_range: float = 0.1
    """Vertical shift as fraction of height (±height_shift_range)"""
    
    zoom_range: float = 0.1
    """Zoom range (1.0 ± zoom_range)"""
    
    horizontal_flip: bool = True
    """Whether to randomly flip images horizontally"""
    
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    """Random brightness adjustment range as (min, max) multipliers"""
    
    fill_mode: str = 'constant'
    """How to fill points outside boundaries during augmentation.
    Options: 'constant', 'nearest', 'reflect', 'wrap'
    """
    
    cval: float = 0.0
    """Fill value when fill_mode='constant' (0.0 = black)"""
    
    # =========================================================================
    # COLOR CONVERSION
    # =========================================================================
    convert_to_rgb: bool = True
    """Whether to convert grayscale images to RGB (3 channels).
    Set to True for transfer learning models trained on ImageNet.
    Set to False to keep grayscale (1 channel) for custom models.
    """
    
    # =========================================================================
    # METADATA
    # =========================================================================
    name: str = "default"
    """Configuration name for identification in sweeps"""
    
    def __post_init__(self):
        """Validate configuration parameters"""
        # Validate target size
        if len(self.target_size) != 2:
            raise ValueError("target_size must be a tuple of (width, height)")
        if self.target_size[0] <= 0 or self.target_size[1] <= 0:
            raise ValueError("target_size dimensions must be positive")
        
        # Validate CLAHE parameters
        if self.clahe_clip_limit <= 0:
            raise ValueError("clahe_clip_limit must be positive")
        if len(self.clahe_tile_size) != 2:
            raise ValueError("clahe_tile_size must be a tuple of (width, height)")
        
        # Validate augmentation ranges
        if self.rotation_range < 0:
            raise ValueError("rotation_range must be non-negative")
        if self.width_shift_range < 0 or self.width_shift_range > 1:
            raise ValueError("width_shift_range must be in [0, 1]")
        if self.height_shift_range < 0 or self.height_shift_range > 1:
            raise ValueError("height_shift_range must be in [0, 1]")
        if self.zoom_range < 0 or self.zoom_range > 1:
            raise ValueError("zoom_range must be in [0, 1]")
    
    def to_dict(self):
        """Convert configuration to dictionary for logging"""
        return {
            'target_size': self.target_size,
            'aspect_handling': self.aspect_handling,
            'use_clahe': self.use_clahe,
            'clahe_clip_limit': self.clahe_clip_limit,
            'clahe_tile_size': self.clahe_tile_size,
            'use_denoising': self.use_denoising,
            'denoise_strength': self.denoise_strength,
            'use_edge_enhancement': self.use_edge_enhancement,
            'edge_enhancement_amount': self.edge_enhancement_amount,
            'rotation_range': self.rotation_range,
            'width_shift_range': self.width_shift_range,
            'height_shift_range': self.height_shift_range,
            'zoom_range': self.zoom_range,
            'horizontal_flip': self.horizontal_flip,
            'brightness_range': self.brightness_range,
            'convert_to_rgb': self.convert_to_rgb,
            'name': self.name,
        }
    
    def __str__(self):
        """String representation for logging"""
        return f"PreprocessingConfig(name='{self.name}', target_size={self.target_size}, " \
               f"clahe={self.use_clahe}, denoise={self.use_denoising}, " \
               f"edge_enhance={self.use_edge_enhancement})"


# Preset configurations for convenience
def get_baseline_config(target_size: Tuple[int, int] = (224, 224)) -> PreprocessingConfig:
    """
    Get baseline preprocessing configuration (no enhancements).
    
    Args:
        target_size: Target image size
    
    Returns:
        PreprocessingConfig with default parameters, no enhancements
    """
    return PreprocessingConfig(
        target_size=target_size,
        use_clahe=False,
        use_denoising=False,
        use_edge_enhancement=False,
        convert_to_rgb=True,
        name=f"baseline_{target_size[0]}x{target_size[1]}"
    )


def get_enhanced_config(target_size: Tuple[int, int] = (224, 224)) -> PreprocessingConfig:
    """
    Get enhanced preprocessing configuration with CLAHE.
    
    Args:
        target_size: Target image size
    
    Returns:
        PreprocessingConfig with CLAHE enabled
    """
    return PreprocessingConfig(
        target_size=target_size,
        use_clahe=True,
        clahe_clip_limit=2.0,
        clahe_tile_size=(8, 8),
        use_denoising=False,
        use_edge_enhancement=False,
        convert_to_rgb=True,
        name=f"clahe_{target_size[0]}x{target_size[1]}"
    )


if __name__ == "__main__":
    # Test configurations
    print("Baseline Config:")
    baseline = get_baseline_config()
    print(baseline)
    print()
    
    print("Enhanced Config:")
    enhanced = get_enhanced_config()
    print(enhanced)
    print()
    
    print("Config as dict:")
    print(baseline.to_dict())
