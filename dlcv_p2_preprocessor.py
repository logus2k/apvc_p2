"""
Deep Learning for Computer Vision - Project 2
Image Preprocessor

This module implements the ImagePreprocessor class that applies all
preprocessing operations to images based on a PreprocessingConfig.
"""

import cv2
import numpy as np

# Suppress TensorFlow info messages
# 0=all, 1=filter INFO, 2=filter WARNING, 3=filter ERROR
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from keras.src.legacy.preprocessing.image import ImageDataGenerator
from dlcv_p2_preprocessing_config import PreprocessingConfig


class ImagePreprocessor:
    """
    Handles all image preprocessing operations.
    
    This class applies a configurable preprocessing pipeline to images including:
    - CLAHE enhancement
    - Denoising
    - Edge enhancement
    - Aspect ratio handling (crop or pad)
    - Resizing
    - Color conversion (grayscale to RGB)
    
    Note: Normalization is NOT applied here - it's model-specific and handled
    by each model's preprocess_input function.
    """
    
    def __init__(self, config: PreprocessingConfig):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: PreprocessingConfig instance with all parameters
        """
        self.config = config
        
        # Initialize CLAHE if needed
        if self.config.use_clahe:
            self.clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit,
                tileGridSize=self.config.clahe_tile_size
            )
        else:
            self.clahe = None
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        CLAHE enhances local contrast without over-amplifying noise.
        It's particularly effective for medical imaging.
        
        Args:
            image: Input image (grayscale or RGB)
        
        Returns:
            Enhanced image
        """
        if self.clahe is None:
            return image
        
        # CLAHE works on single channel
        if len(image.shape) == 3 and image.shape[2] == 3:
            # RGB image - convert to LAB, apply CLAHE on L channel
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            # Grayscale image
            if len(image.shape) == 3:
                image = image.squeeze()
            return self.clahe.apply(image)
    
    def apply_denoising(self, image: np.ndarray) -> np.ndarray:
        """
        Apply non-local means denoising.
        
        Reduces noise while preserving edges and details.
        
        Args:
            image: Input image (grayscale or RGB)
        
        Returns:
            Denoised image
        """
        if not self.config.use_denoising:
            return image
        
        # Check if grayscale or RGB
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            # Grayscale
            if len(image.shape) == 3:
                image = image.squeeze()
            
            denoised = cv2.fastNlMeansDenoising(
                image,
                h=self.config.denoise_strength,
                templateWindowSize=self.config.denoise_template_window_size,
                searchWindowSize=self.config.denoise_search_window_size
            )
            return denoised
        else:
            # RGB
            denoised = cv2.fastNlMeansDenoisingColored(
                image,
                h=self.config.denoise_strength,
                hColor=self.config.denoise_strength,
                templateWindowSize=self.config.denoise_template_window_size,
                searchWindowSize=self.config.denoise_search_window_size
            )
            return denoised
    
    def apply_edge_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Apply edge enhancement using unsharp masking.
        
        Enhances edges and fine details in the image.
        
        Args:
            image: Input image (grayscale or RGB)
        
        Returns:
            Edge-enhanced image
        """
        if not self.config.use_edge_enhancement:
            return image
        
        # Create Gaussian blur
        blurred = cv2.GaussianBlur(
            image,
            (0, 0),
            self.config.edge_enhancement_radius
        )
        
        # Unsharp mask: image + amount * (image - blurred)
        enhanced = cv2.addWeighted(
            image,
            1.0 + self.config.edge_enhancement_amount,
            blurred,
            -self.config.edge_enhancement_amount,
            0
        )
        
        # Apply threshold if specified
        if self.config.edge_enhancement_threshold > 0:
            mask = np.abs(image.astype(float) - blurred.astype(float)) > \
                   (self.config.edge_enhancement_threshold * 255)
            enhanced = np.where(mask, enhanced, image)
        
        # Clip values to valid range
        enhanced = np.clip(enhanced, 0, 255).astype(image.dtype)
        
        return enhanced
    
    def handle_aspect_ratio(self, image: np.ndarray) -> np.ndarray:
        """
        Handle aspect ratio before resizing.
        
        Args:
            image: Input image
        
        Returns:
            Square image (either cropped or padded)
        """
        h, w = image.shape[:2]
        
        if self.config.aspect_handling == 'center_crop':
            # Crop to square from center
            min_dim = min(h, w)
            y_offset = (h - min_dim) // 2
            x_offset = (w - min_dim) // 2
            cropped = image[y_offset:y_offset+min_dim, x_offset:x_offset+min_dim]
            return cropped
        
        elif self.config.aspect_handling == 'pad':
            # Pad to square with black borders
            max_dim = max(h, w)
            
            if len(image.shape) == 3:
                padded = np.zeros((max_dim, max_dim, image.shape[2]), dtype=image.dtype)
            else:
                padded = np.zeros((max_dim, max_dim), dtype=image.dtype)
            
            y_offset = (max_dim - h) // 2
            x_offset = (max_dim - w) // 2
            padded[y_offset:y_offset+h, x_offset:x_offset+w] = image
            return padded
        
        else:
            # No aspect handling - return as is
            return image
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image
        
        Returns:
            Resized image
        """
        return cv2.resize(
            image,
            self.config.target_size,
            interpolation=self.config.resize_interpolation
        )
    
    def convert_color(self, image: np.ndarray) -> np.ndarray:
        """
        Convert grayscale to RGB if needed.
        
        Transfer learning models trained on ImageNet expect 3-channel RGB input.
        This function replicates the grayscale channel to create RGB.
        
        Args:
            image: Input image (grayscale or RGB)
        
        Returns:
            RGB image (3 channels) or original if convert_to_rgb=False
        """
        if not self.config.convert_to_rgb:
            return image
        
        # Check if image is grayscale
        if len(image.shape) == 2:
            # Shape (H, W) -> (H, W, 3)
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            # Shape (H, W, 1) -> (H, W, 3)
            return np.repeat(image, 3, axis=2)
        
        # Already RGB or has 3 channels
        return image
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Apply full preprocessing pipeline.
        
        Pipeline order:
        1. Enhancement (CLAHE, denoising, edge enhancement)
        2. Aspect ratio handling (crop or pad to square)
        3. Resize to target size
        4. Color conversion (grayscale to RGB if needed)
        
        Note: Normalization is NOT applied here - it's model-specific
        and handled by the model's preprocess_input function.
        
        Args:
            image: Input image (uint8, 0-255 range)
        
        Returns:
            Preprocessed image (uint8, 0-255 range, ready for normalization)
        """
        # Step 1: Enhancement
        if self.config.use_clahe:
            image = self.apply_clahe(image)
        
        if self.config.use_denoising:
            image = self.apply_denoising(image)
        
        if self.config.use_edge_enhancement:
            image = self.apply_edge_enhancement(image)
        
        # Step 2: Aspect ratio handling
        image = self.handle_aspect_ratio(image)
        
        # Step 3: Resize
        image = self.resize_image(image)
        
        # Step 4: Color conversion
        image = self.convert_color(image)
        
        return image
    
    def get_augmentation_generator(self) -> ImageDataGenerator:
        """
        Create ImageDataGenerator for training augmentation.
        
        Returns:
            Configured ImageDataGenerator
        """
        return ImageDataGenerator(
            rotation_range=self.config.rotation_range,
            width_shift_range=self.config.width_shift_range,
            height_shift_range=self.config.height_shift_range,
            zoom_range=self.config.zoom_range,
            horizontal_flip=self.config.horizontal_flip,
            brightness_range=self.config.brightness_range,
            fill_mode=self.config.fill_mode,
            cval=self.config.cval,
            preprocessing_function=None  # Applied separately per model
        )
    
    def get_pipeline_stages(self, image: np.ndarray) -> dict:
        """
        Get intermediate results from each stage of the pipeline.
        
        Useful for visualization and debugging.
        
        Args:
            image: Input image
        
        Returns:
            Dictionary with keys: 'original', 'enhanced', 'cropped', 'resized', 'final'
        """
        stages = {}
        stages['original'] = image.copy()
        
        # Enhancement stage
        enhanced = image.copy()
        if self.config.use_clahe:
            enhanced = self.apply_clahe(enhanced)
        if self.config.use_denoising:
            enhanced = self.apply_denoising(enhanced)
        if self.config.use_edge_enhancement:
            enhanced = self.apply_edge_enhancement(enhanced)
        stages['enhanced'] = enhanced
        
        # Cropping stage
        cropped = self.handle_aspect_ratio(enhanced)
        stages['cropped'] = cropped
        
        # Resizing stage
        resized = self.resize_image(cropped)
        stages['resized'] = resized
        
        # Final (with color conversion)
        final = self.convert_color(resized)
        stages['final'] = final
        
        return stages


if __name__ == "__main__":
    # Test preprocessor
    from dlcv_p2_preprocessing_config import get_baseline_config, get_enhanced_config
    
    # Create test image
    test_image = np.random.randint(0, 255, (1470, 1033), dtype=np.uint8)
    
    print("Testing Baseline Preprocessor:")
    baseline_config = get_baseline_config(target_size=(224, 224))
    baseline_preprocessor = ImagePreprocessor(baseline_config)
    processed = baseline_preprocessor.preprocess(test_image)
    print(f"Input shape: {test_image.shape}")
    print(f"Output shape: {processed.shape}")
    print()
    
    print("Testing Enhanced Preprocessor:")
    enhanced_config = get_enhanced_config(target_size=(384, 384))
    enhanced_preprocessor = ImagePreprocessor(enhanced_config)
    processed = enhanced_preprocessor.preprocess(test_image)
    print(f"Input shape: {test_image.shape}")
    print(f"Output shape: {processed.shape}")
    print()
    
    print("Pipeline stages:")
    stages = enhanced_preprocessor.get_pipeline_stages(test_image)
    for stage_name, stage_image in stages.items():
        print(f"  {stage_name}: {stage_image.shape}")
