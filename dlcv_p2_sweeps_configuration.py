"""
Deep Learning for Computer Vision - Project 2
Sweep Configurations

This module defines comprehensive sweep configurations for hyperparameter tuning.
Includes configurations for:
- Image enhancement (CLAHE, denoising, edge enhancement)
- Image resolutions (224, 384, 512)
- Data augmentation parameters
- Combined configurations for systematic testing
"""

from typing import List, Dict, Any, Optional
from dlcv_p2_preprocessing_config import PreprocessingConfig


# =============================================================================
# IMAGE RESOLUTION SWEEPS
# =============================================================================

IMAGE_SIZES = [
    (224, 224),  # Standard ImageNet size
    (384, 384),  # High resolution
    (512, 512),  # Very high resolution (if VRAM allows)
]


# =============================================================================
# CLAHE ENHANCEMENT SWEEPS
# =============================================================================

CLAHE_SWEEP_CONFIGS = [
    # Baseline: No CLAHE
    {
        'name': 'no_clahe',
        'use_clahe': False,
    },
    
    # CLAHE with different clip limits
    {
        'name': 'clahe_clip_1.0',
        'use_clahe': True,
        'clahe_clip_limit': 1.0,
        'clahe_tile_size': (8, 8),
    },
    {
        'name': 'clahe_clip_2.0',
        'use_clahe': True,
        'clahe_clip_limit': 2.0,
        'clahe_tile_size': (8, 8),
    },
    {
        'name': 'clahe_clip_3.0',
        'use_clahe': True,
        'clahe_clip_limit': 3.0,
        'clahe_tile_size': (8, 8),
    },
    {
        'name': 'clahe_clip_4.0',
        'use_clahe': True,
        'clahe_clip_limit': 4.0,
        'clahe_tile_size': (8, 8),
    },
    
    # CLAHE with different tile sizes (using optimal clip limit)
    {
        'name': 'clahe_tile_4x4',
        'use_clahe': True,
        'clahe_clip_limit': 2.0,
        'clahe_tile_size': (4, 4),
    },
    {
        'name': 'clahe_tile_16x16',
        'use_clahe': True,
        'clahe_clip_limit': 2.0,
        'clahe_tile_size': (16, 16),
    },
]


# =============================================================================
# DENOISING SWEEPS
# =============================================================================

DENOISING_SWEEP_CONFIGS = [
    # No denoising (baseline)
    {
        'name': 'no_denoise',
        'use_denoising': False,
    },
    
    # Different denoising strengths
    {
        'name': 'denoise_light',
        'use_denoising': True,
        'denoise_strength': 5.0,
    },
    {
        'name': 'denoise_medium',
        'use_denoising': True,
        'denoise_strength': 10.0,
    },
    {
        'name': 'denoise_strong',
        'use_denoising': True,
        'denoise_strength': 15.0,
    },
]


# =============================================================================
# EDGE ENHANCEMENT SWEEPS
# =============================================================================

EDGE_ENHANCEMENT_SWEEP_CONFIGS = [
    # No edge enhancement (baseline)
    {
        'name': 'no_edge',
        'use_edge_enhancement': False,
    },
    
    # Different enhancement amounts
    {
        'name': 'edge_light',
        'use_edge_enhancement': True,
        'edge_enhancement_amount': 0.5,
        'edge_enhancement_radius': 1.0,
    },
    {
        'name': 'edge_medium',
        'use_edge_enhancement': True,
        'edge_enhancement_amount': 1.0,
        'edge_enhancement_radius': 1.0,
    },
    {
        'name': 'edge_strong',
        'use_edge_enhancement': True,
        'edge_enhancement_amount': 2.0,
        'edge_enhancement_radius': 1.0,
    },
]


# =============================================================================
# DATA AUGMENTATION SWEEPS
# =============================================================================

AUGMENTATION_SWEEP_CONFIGS = [
    # Baseline augmentation (from design decisions)
    {
        'name': 'aug_baseline',
        'rotation_range': 10,
        'width_shift_range': 0.1,
        'height_shift_range': 0.1,
        'zoom_range': 0.1,
        'horizontal_flip': True,
        'brightness_range': (0.8, 1.2),
    },
    
    # Conservative augmentation
    {
        'name': 'aug_conservative',
        'rotation_range': 5,
        'width_shift_range': 0.05,
        'height_shift_range': 0.05,
        'zoom_range': 0.05,
        'horizontal_flip': True,
        'brightness_range': (0.9, 1.1),
    },
    
    # Aggressive augmentation
    {
        'name': 'aug_aggressive',
        'rotation_range': 15,
        'width_shift_range': 0.15,
        'height_shift_range': 0.15,
        'zoom_range': 0.15,
        'horizontal_flip': True,
        'brightness_range': (0.7, 1.3),
    },
    
    # No augmentation (for comparison)
    {
        'name': 'aug_none',
        'rotation_range': 0,
        'width_shift_range': 0.0,
        'height_shift_range': 0.0,
        'zoom_range': 0.0,
        'horizontal_flip': False,
        'brightness_range': (1.0, 1.0),
    },
]


# =============================================================================
# COMBINED ENHANCEMENT SWEEPS
# =============================================================================

COMBINED_ENHANCEMENT_CONFIGS = [
    # Baseline: No enhancements
    {
        'name': 'baseline_no_enhancement',
        'use_clahe': False,
        'use_denoising': False,
        'use_edge_enhancement': False,
    },
    
    # CLAHE only
    {
        'name': 'clahe_only',
        'use_clahe': True,
        'clahe_clip_limit': 2.0,
        'clahe_tile_size': (8, 8),
        'use_denoising': False,
        'use_edge_enhancement': False,
    },
    
    # CLAHE + Denoising
    {
        'name': 'clahe_denoise',
        'use_clahe': True,
        'clahe_clip_limit': 2.0,
        'clahe_tile_size': (8, 8),
        'use_denoising': True,
        'denoise_strength': 10.0,
        'use_edge_enhancement': False,
    },
    
    # CLAHE + Edge Enhancement
    {
        'name': 'clahe_edge',
        'use_clahe': True,
        'clahe_clip_limit': 2.0,
        'clahe_tile_size': (8, 8),
        'use_denoising': False,
        'use_edge_enhancement': True,
        'edge_enhancement_amount': 1.0,
    },
    
    # All enhancements
    {
        'name': 'all_enhancements',
        'use_clahe': True,
        'clahe_clip_limit': 2.0,
        'clahe_tile_size': (8, 8),
        'use_denoising': True,
        'denoise_strength': 10.0,
        'use_edge_enhancement': True,
        'edge_enhancement_amount': 1.0,
    },
]


# =============================================================================
# SWEEP GENERATION FUNCTIONS
# =============================================================================

def generate_resolution_sweep(
    base_config: Optional[Dict[str, Any]] = None
) -> List[PreprocessingConfig]:
    """
    Generate sweep configurations for different resolutions.
    
    Args:
        base_config: Base configuration dictionary (optional)
    
    Returns:
        List of PreprocessingConfig instances with different resolutions
    """
    if base_config is None:
        base_config = {}
    
    configs = []
    for size in IMAGE_SIZES:
        config_dict = base_config.copy()
        config_dict['target_size'] = size
        config_dict['name'] = f"{base_config.get('name', 'config')}_{size[0]}x{size[1]}"
        configs.append(PreprocessingConfig(**config_dict))
    
    return configs


def generate_clahe_sweep(
    target_size: tuple = (224, 224)
) -> List[PreprocessingConfig]:
    """
    Generate sweep configurations for CLAHE parameters.
    
    Args:
        target_size: Target image size
    
    Returns:
        List of PreprocessingConfig instances with different CLAHE settings
    """
    configs = []
    for clahe_config in CLAHE_SWEEP_CONFIGS:
        config_dict = clahe_config.copy()
        config_dict['target_size'] = target_size
        config_dict['name'] = f"{clahe_config['name']}_{target_size[0]}x{target_size[1]}"
        configs.append(PreprocessingConfig(**config_dict))
    
    return configs


def generate_augmentation_sweep(
    target_size: tuple = (224, 224)
) -> List[PreprocessingConfig]:
    """
    Generate sweep configurations for augmentation parameters.
    
    Args:
        target_size: Target image size
    
    Returns:
        List of PreprocessingConfig instances with different augmentation settings
    """
    configs = []
    for aug_config in AUGMENTATION_SWEEP_CONFIGS:
        config_dict = aug_config.copy()
        config_dict['target_size'] = target_size
        configs.append(PreprocessingConfig(**config_dict))
    
    return configs


def generate_combined_sweep(
    target_size: tuple = (224, 224)
) -> List[PreprocessingConfig]:
    """
    Generate sweep configurations for combined enhancements.
    
    Args:
        target_size: Target image size
    
    Returns:
        List of PreprocessingConfig instances with different enhancement combinations
    """
    configs = []
    for combined_config in COMBINED_ENHANCEMENT_CONFIGS:
        config_dict = combined_config.copy()
        config_dict['target_size'] = target_size
        configs.append(PreprocessingConfig(**config_dict))
    
    return configs


def generate_full_sweep() -> List[PreprocessingConfig]:
    """
    Generate comprehensive sweep covering all parameters.
    
    WARNING: This generates a very large number of configurations.
    Consider using targeted sweeps instead.
    
    Returns:
        List of all possible PreprocessingConfig combinations
    """
    configs = []
    
    # For each resolution
    for size in IMAGE_SIZES:
        # For each enhancement combination
        for enhancement_config in COMBINED_ENHANCEMENT_CONFIGS:
            # For each augmentation strategy
            for aug_config in AUGMENTATION_SWEEP_CONFIGS:
                config_dict = {
                    'target_size': size,
                    'name': f"{enhancement_config['name']}_{aug_config['name']}_{size[0]}x{size[1]}"
                }
                config_dict.update(enhancement_config)
                config_dict.update(aug_config)
                configs.append(PreprocessingConfig(**config_dict))
    
    return configs


# =============================================================================
# RECOMMENDED SWEEP STRATEGIES
# =============================================================================

def get_baseline_sweep() -> List[PreprocessingConfig]:
    """
    Get recommended baseline sweep for initial experiments.
    
    Tests:
    - 3 resolutions (224, 384, 512)
    - No enhancement vs CLAHE
    - Baseline augmentation
    
    Returns:
        List of 6 baseline configurations (3 resolutions × 2 enhancement levels)
    """
    configs = []
    
    for size in IMAGE_SIZES:
        # No enhancement
        configs.append(PreprocessingConfig(
            target_size=size,
            use_clahe=False,
            name=f"baseline_{size[0]}x{size[1]}"
        ))
        
        # With CLAHE
        configs.append(PreprocessingConfig(
            target_size=size,
            use_clahe=True,
            clahe_clip_limit=2.0,
            clahe_tile_size=(8, 8),
            name=f"clahe_{size[0]}x{size[1]}"
        ))
    
    return configs


def get_enhancement_sweep(target_size: tuple = (224, 224)) -> List[PreprocessingConfig]:
    """
    Get detailed enhancement sweep for a specific resolution.
    
    Tests all enhancement combinations at the specified resolution.
    
    Args:
        target_size: Target image size
    
    Returns:
        List of configurations testing different enhancements
    """
    return generate_combined_sweep(target_size)


def get_augmentation_sweep(target_size: tuple = (224, 224)) -> List[PreprocessingConfig]:
    """
    Get augmentation parameter sweep for a specific resolution.
    
    Args:
        target_size: Target image size
    
    Returns:
        List of configurations testing different augmentation strategies
    """
    return generate_augmentation_sweep(target_size)


if __name__ == "__main__":
    # Test sweep generation
    title = "SWEEP CONFIGURATION EXAMPLES"
    print("=" * len(title))
    print(title)
    print("=" * len(title))
    
    print("\n1. Baseline Sweep:")
    baseline_sweep = get_baseline_sweep()
    print(f"   Total configurations: {len(baseline_sweep)}")
    counter = 0
    for config in baseline_sweep:
        counter += 1
        print(f"   {str(counter)}) {config.name}")
    
    print("\n2. CLAHE Sweep (224x224):")
    clahe_sweep = generate_clahe_sweep()
    print(f"   Total configurations: {len(clahe_sweep)}")
    counter = 0
    for config in clahe_sweep:
        counter += 1
        print(f"   {str(counter)}) {config.name}")
    
    print("\n3. Enhancement Sweep (224x224):")
    enhancement_sweep = get_enhancement_sweep()
    print(f"   Total configurations: {len(enhancement_sweep)}")
    counter = 0
    for config in enhancement_sweep:
        counter += 1
        print(f"   {str(counter)}) {config.name}")
    
    print("\n4. Full Sweep (Large):")
    full_sweep = generate_full_sweep()
    print(f"   Total configurations: {len(full_sweep)}")
    print(f"   The full sweep requires {len(full_sweep)} training runs")
