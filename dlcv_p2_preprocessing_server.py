"""
Deep Learning for Computer Vision - Project 2
Preprocessing Configuration Server

FastAPI + Socket.IO server for interactive preprocessing configuration UI.
"""

import os
import json
import cv2
import numpy as np
import socketio
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import Dict, List, Optional

from dlcv_p2_config import TRAIN_DIR, TEST_DIR

import tensorflow as tf
from keras.layers import (
    RandomRotation, RandomTranslation, RandomZoom, RandomFlip,
    RandomBrightness, RandomContrast, GaussianNoise
)
from keras import Sequential


AUGMENTATION_AVAILABLE = True

# Metric display names
METRIC_LABELS = {
    'global_std': 'Global Contrast (std)',
    'dynamic_range': 'Dynamic Range',
    'hist_entropy': 'Histogram Entropy',
    'kurtosis': 'Kurtosis',
    'skewness': 'Skewness',
    'local_std_mean': 'Local Std Mean',
    'local_std_var': 'Local Std Variance',
    'high_freq_energy': 'High Freq Energy',
    'centering_offset_x': 'Horizontal Centering Offset',
    'centering_offset_y': 'Vertical Centering Offset',
    'content_width_ratio': 'Content Width Ratio',
    'content_height_ratio': 'Content Height Ratio',
    'aspect_ratio': 'Aspect Ratio'
}

# Metrics to include in range calculations
METRIC_KEYS = list(METRIC_LABELS.keys())


# Create FastAPI app
app = FastAPI(title="DLCV Preprocessing Configuration")

# Create Socket.IO server
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*',
    logger=True,
    engineio_logger=False
)

# Mount STATIC directory normally
app.mount("/static", StaticFiles(directory="static"), name="static")

# Dynamically find first available JSON config
def get_default_config_path():
    """Get the first available JSON config file in static folder"""
    static_dir = "static"
    if os.path.exists(static_dir):
        json_files = sorted([f for f in os.listdir(static_dir) 
                           if f.endswith('.json') and f != 'du_metrics.json'])
        if json_files:
            return os.path.join(static_dir, json_files[0])
    return None

JSON_CONFIG_PATH = get_default_config_path()

# Wrap with ASGI app (NO static_files here)
socket_app = socketio.ASGIApp(
    sio,
    app
)

# Global configuration
config_data = None
# Cache: {dataset: {dimension: [paths]}}
dimension_cache = {}


# Load DU metrics into memory
DU_METRICS_PATH = "./static/du_metrics.json"

if os.path.exists(DU_METRICS_PATH):
    with open(DU_METRICS_PATH, "r") as f:
        du_records = json.load(f)
else:
    du_records = []

# Convert list of records into dictionary using image_path as key
DU_LOOKUP = {rec["image_path"]: rec for rec in du_records}

print(f"Loaded {len(DU_LOOKUP)} DU metric records.")


def get_du_metrics(image_path):
    """Return the DU metrics for the given image path, or an empty dict if unavailable."""
    # Normalize paths to the same style used in DF_VIEW export
    normalized = image_path.replace("\\", "/")
    return DU_LOOKUP.get(normalized, {})


def compute_metric_ranges(image_paths: List[str], dataset: str) -> Dict:
    """
    Compute min/max/outlier thresholds for each DU metric across given image paths.
    
    Args:
        image_paths: List of image paths to analyze
        dataset: 'train' or 'test'
    
    Returns:
        Dictionary with metric statistics:
        {
            'metric_name': {
                'min': float,
                'max': float,
                'q1': float,
                'q3': float,
                'outlier_low': float,  # Q1 - 1.5*IQR
                'outlier_high': float, # Q3 + 1.5*IQR
                'label': str,
                'precision': int,
                'total_count': int     # Number of images with this metric
            }
        }
    """
    # Normalize all paths
    normalized_paths = [p.replace("\\", "/") for p in image_paths]
    
    # Collect metric values for each metric
    metric_values = {key: [] for key in METRIC_KEYS}
    
    for path in normalized_paths:
        du_data = DU_LOOKUP.get(path, {})
        if du_data:
            for key in METRIC_KEYS:
                if key in du_data:
                    metric_values[key].append(float(du_data[key]))
    
    # Compute statistics for each metric
    ranges = {}
    for key in METRIC_KEYS:
        values = metric_values[key]
        
        if not values:
            # No data available for this metric
            ranges[key] = {
                'min': 0.0,
                'max': 1.0,
                'q1': 0.0,
                'q3': 1.0,
                'outlier_low': 0.0,
                'outlier_high': 1.0,
                'label': METRIC_LABELS[key],
                'precision': 2,
                'total_count': 0
            }
            continue
        
        values_array = np.array(values)
        
        # Compute percentiles
        min_val = float(np.min(values_array))
        max_val = float(np.max(values_array))
        q1 = float(np.percentile(values_array, 25))
        q3 = float(np.percentile(values_array, 75))
        iqr = q3 - q1
        
        # Outlier thresholds (IQR method)
        outlier_low = q1 - 1.5 * iqr
        outlier_high = q3 + 1.5 * iqr
        
        # Determine precision based on value range
        value_range = max_val - min_val
        if value_range < 1:
            precision = 4
        elif value_range < 10:
            precision = 3
        else:
            precision = 2
        
        ranges[key] = {
            'min': round(min_val, precision),
            'max': round(max_val, precision),
            'q1': round(q1, precision),
            'q3': round(q3, precision),
            'outlier_low': round(outlier_low, precision),
            'outlier_high': round(outlier_high, precision),
            'label': METRIC_LABELS[key],
            'precision': precision,
            'total_count': len(values)
        }
    
    return ranges


def apply_du_filters(image_paths: List[str], du_filters: Dict) -> List[str]:
    """
    Filter image paths based on DU metric ranges.
    
    Args:
        image_paths: List of image paths to filter
        du_filters: Dictionary of filters {metric_name: [min_val, max_val], ...}
    
    Returns:
        Filtered list of image paths
    """
    if not du_filters:
        return image_paths
    
    filtered_paths = []
    
    for path in image_paths:
        normalized_path = path.replace("\\", "/")
        du_data = DU_LOOKUP.get(normalized_path, {})
        
        # If no DU data, skip this image
        if not du_data:
            continue
        
        # Check all filters (AND condition)
        passes_all = True
        for metric_name, (min_val, max_val) in du_filters.items():
            if metric_name not in du_data:
                passes_all = False
                break
            
            metric_val = float(du_data[metric_name])
            if not (min_val <= metric_val <= max_val):
                passes_all = False
                break
        
        if passes_all:
            filtered_paths.append(path)
    
    return filtered_paths


def build_dimension_cache(dataset: str = 'train'):
    """
    Build cache of all images and their dimensions.
    This is done once to avoid repeatedly reading images.
    """
    global dimension_cache
    
    if dataset in dimension_cache:
        return dimension_cache[dataset]
    
    print(f"Building dimension cache for {dataset} dataset...")
    
    base_dir = TRAIN_DIR if dataset == 'train' else TEST_DIR
    cache = {}  # {(w, h): [paths]}
    
    for class_name in ['NORMAL', 'PNEUMONIA']:
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        
        files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for idx, filename in enumerate(files):
            filepath = os.path.join(class_dir, filename)
            
            # Read image dimensions
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                h, w = img.shape
                key = (w, h)
                if key not in cache:
                    cache[key] = []
                cache[key].append(filepath)
            
            # Progress indicator
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1} images from {class_name}...")
    
    dimension_cache[dataset] = cache
    print(f"Cache built: {len(cache)} unique dimensions, {sum(len(v) for v in cache.values())} total images")
    
    return cache


def load_config_file():
    """Load configuration from JSON file"""
    global config_data
    
    if not JSON_CONFIG_PATH:
        print("No config file available")
        return None
    
    try:
        with open(JSON_CONFIG_PATH, 'r') as f:
            config_data = json.load(f)
        return config_data
    except FileNotFoundError:
        print("File not found")
        return None


def save_config_file(data: dict):
    """Save configuration to JSON file"""
    global config_data
    
    if not JSON_CONFIG_PATH:
        raise ValueError("No config file path available")
    
    with open(JSON_CONFIG_PATH, 'w') as f:
        json.dump(data, f, indent=2)
    config_data = data


def get_image_paths_for_dimension(dimension: str, dataset: str = 'train') -> List[str]:
    """
    Get image paths for a specific dimension using tolerance from config.
    
    Args:
        dimension: Dimension string like "1470x1033"
        dataset: 'train' or 'test'
    
    Returns:
        List of all matching image paths
    """
    # Get tolerance from config
    config = load_config_file()
    tolerance_percent = 15.0  # default fallback
    if config and 'global_settings' in config:
        tolerance_percent = config['global_settings'].get('allowed_variation', 15.0)
    
    w_target, h_target = map(int, dimension.split('x'))
    
    # Calculate tolerance
    w_tolerance = w_target * (tolerance_percent / 100.0)
    h_tolerance = h_target * (tolerance_percent / 100.0)
    w_min, w_max = w_target - w_tolerance, w_target + w_tolerance
    h_min, h_max = h_target - h_tolerance, h_target + h_tolerance
    
    # Build cache if not exists
    cache = build_dimension_cache(dataset)
    
    # Find all matching images from cache
    image_paths = []
    for (w, h), paths in cache.items():
        if w_min <= w <= w_max and h_min <= h <= h_max:
            image_paths.extend(paths)
    
    return image_paths


def apply_preprocessing(image: np.ndarray, params: dict) -> np.ndarray:
    """
    Apply preprocessing transformations to an image.
    
    Args:
        image: Input grayscale image
        params: Preprocessing parameters
    
    Returns:
        Preprocessed image
    """
    result = image.copy()
    
    # Apply horizontal crop (crop from left and right)
    h_crop = int(params.get('h_crop', 0))
    if h_crop > 0:
        # Ensure we don't crop more than available
        max_h_crop = result.shape[1] // 2
        h_crop = min(h_crop, max_h_crop - 1)
        if h_crop > 0:
            result = result[:, h_crop:-h_crop]
    
    # Apply vertical crop (crop from top and bottom)
    v_crop = int(params.get('v_crop', 0))
    if v_crop > 0:
        # Ensure we don't crop more than available
        max_v_crop = result.shape[0] // 2
        v_crop = min(v_crop, max_v_crop - 1)
        if v_crop > 0:
            result = result[v_crop:-v_crop, :]
    
    # Apply CLAHE
    if params.get('clahe', 0) > 0:
        clahe = cv2.createCLAHE(
            clipLimit=float(params['clahe']),
            tileGridSize=(8, 8)
        )
        result = clahe.apply(result)
    
    # Apply zoom
    if params.get('zoom_in', 0) > 1.0:
        zoom_factor = float(params['zoom_in'])
        h, w = result.shape
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
        
        # Resize and crop to original size
        zoomed = cv2.resize(result, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        crop_h = (new_h - h) // 2
        crop_w = (new_w - w) // 2
        result = zoomed[crop_h:crop_h+h, crop_w:crop_w+w]
    
    return result


def apply_augmentation(image: np.ndarray, aug_params: dict) -> np.ndarray:
    """
    Apply Keras-compatible data augmentation to an image using modern Keras Preprocessing Layers.
    
    This replaces the deprecated ImageDataGenerator and all custom NumPy/OpenCV post-processing
    to ensure the augmentation is 100% Keras-native.
    
    Args:
        image: Input grayscale image (already preprocessed)
        aug_params: Augmentation parameters
    
    Returns:
        Augmented image
    """
    # Check if any augmentation is active
    if not AUGMENTATION_AVAILABLE:
        return image
    
    if not aug_params:
        return image
    
    # --- 1. Parameter Extraction ---
    rotation = float(aug_params.get('rotation', 0))
    brightness_var = float(aug_params.get('brightness_var', 0))
    zoom = float(aug_params.get('zoom', 0))
    width_shift = float(aug_params.get('width_shift', 0))
    height_shift = float(aug_params.get('height_shift', 0))
    contrast_var = float(aug_params.get('contrast_var', 0))
    horizontal_flip = bool(aug_params.get('horizontal_flip', False))
    vertical_flip = bool(aug_params.get('vertical_flip', False))
    gaussian_noise = float(aug_params.get('gaussian_noise', 0))
    
    # Check if any augmentation parameter is actually set
    if (rotation == 0 and brightness_var == 0 and zoom == 0 and 
        width_shift == 0 and height_shift == 0 and contrast_var == 0 and 
        not horizontal_flip and not vertical_flip and gaussian_noise == 0):
        return image

    # --- 2. Input/Output Setup ---
    # Convert image (H, W) to Tensor (1, H, W, 1) and change dtype to float32
    # FIX: Normalize input to [0, 1] for GaussianNoise/Brightness/Contrast layers
    input_tensor = tf.convert_to_tensor(image / 255.0, dtype=tf.float32)
    input_tensor = tf.expand_dims(input_tensor, axis=0)  # Add batch dimension
    input_tensor = tf.expand_dims(input_tensor, axis=-1) # Add channel dimension (1 for grayscale)
    
    # --- 3. Build Keras Sequential Model (Dynamic Augmentation Pipeline) ---
    aug_layers = []

    # Random Rotations 
    if rotation > 0:
        # factor = rotation_range / 360.0, a fraction of 2*PI
        factor = rotation / 360.0
        aug_layers.append(RandomRotation(factor=factor, fill_mode='constant', fill_value=0.0))

    # Random Translation/Shift (Replaces width_shift_range/height_shift_range)
    if width_shift > 0 or height_shift > 0:
        # PRESERVING OLD LOGIC SWAP:
        # UI's width_shift (horizontal axis slider) controls vertical movement (height_factor).
        # UI's height_shift (vertical axis slider) controls horizontal movement (width_factor).
        height_factor = width_shift 
        width_factor = height_shift

        aug_layers.append(RandomTranslation(
            height_factor=height_factor,
            width_factor=width_factor,
            fill_mode='constant',
            fill_value=0.0
        ))

    # Random Zoom (Replaces zoom_range)
    if zoom > 0:
        # Use single factor for symmetric zoom
        aug_layers.append(RandomZoom(height_factor=zoom, fill_mode='constant', fill_value=0.0))

    # Random Flip (Replaces horizontal_flip/vertical_flip)
    flip_mode = None
    if horizontal_flip and vertical_flip:
        flip_mode = 'horizontal_and_vertical'
    elif horizontal_flip:
        flip_mode = 'horizontal'
    elif vertical_flip:
        flip_mode = 'vertical'

    if flip_mode:
        aug_layers.append(RandomFlip(mode=flip_mode))

    # Random Brightness (Replaces brightness_range)
    if brightness_var > 0:
        # FIX: Change value_range to (0, 1) to match normalized input data
        aug_layers.append(RandomBrightness(factor=brightness_var, value_range=(0, 1)))

    # Random Contrast (Replaces custom NumPy contrast logic)
    if contrast_var > 0:
        # FIX: Change value_range to (0, 1) to match normalized input data
        aug_layers.append(RandomContrast(factor=contrast_var, value_range=(0, 1)))

    # Gaussian Noise (Replaces custom NumPy noise logic)
    if gaussian_noise > 0:
        # FIX: Use gaussian_noise directly as stddev, since input is normalized [0, 1].
        # The layer requires stddev <= 1.0.
        aug_layers.append(GaussianNoise(stddev=gaussian_noise))

    # --- 4. Execute Augmentation ---
    if not aug_layers:
        return image
        
    augmentation_model = Sequential(aug_layers)
    
    # Call the model. training=True ensures randomness is applied.
    augmented_tensor = augmentation_model(input_tensor, training=True)

    # --- 5. Output Conversion ---
    # Remove batch and channel dimensions (1, H, W, 1) -> (H, W)
    augmented = tf.squeeze(augmented_tensor, axis=[0, 3])
    
    # FIX: Scale back to 0-255 range before clipping and final cast
    augmented_scaled = augmented * 255.0

    # Clip values to 0-255 range and cast to uint8.
    final_tensor = tf.cast(
        tf.clip_by_value(augmented_scaled, 0.0, 255.0), 
        tf.uint8
    )
    
    # Convert to NumPy array. We suppress the non-issue Pylance warning here.
    return final_tensor.numpy() # type: ignore


# Socket.IO event handlers

@sio.event
async def connect(sid, environ):
    """Handle client connection"""
    print(f"Client connected: {sid}")


@sio.event
async def get_available_configs(sid):
    """Get list of available JSON config files"""
    try:
        static_dir = "static"
        if os.path.exists(static_dir):
            files = [f for f in os.listdir(static_dir) 
                    if f.endswith('.json') and f != 'du_metrics.json']
            files.sort()
            await sio.emit('available_configs', files, room=sid)
        else:
            await sio.emit('available_configs', [], room=sid)
    except Exception as e:
        print(f"Error listing configs: {e}")
        await sio.emit('available_configs', [], room=sid)


@sio.event
async def disconnect(sid):
    """Handle client disconnection"""
    print(f"Client disconnected: {sid}")


@sio.event
async def get_metric_ranges(sid, data):
    """
    Get min/max/outlier ranges for DU metrics for a specific dimension.
    
    Args:
        data: dict with 'config_file', 'dimension', 'dataset'
    
    Returns:
        Emits 'metric_ranges' with ranges for all metrics
    """
    config_file = data.get('config_file')
    dimension = data.get('dimension')
    dataset = data.get('dataset', 'train')
    
    if not config_file:
        await sio.emit('error', {'message': 'No config file specified'}, room=sid)
        return
    
    if not dimension:
        await sio.emit('error', {'message': 'No dimension specified'}, room=sid)
        return
    
    # Load the specified config file
    config_path = os.path.join("static", config_file)
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        await sio.emit('error', {'message': f'Failed to load config: {str(e)}'}, room=sid)
        return
    
    if 'dimensions' not in config or dimension not in config['dimensions']:
        await sio.emit('error', {'message': f'Dimension {dimension} not found in config'}, room=sid)
        return
    
    dimension_config = config['dimensions'][dimension]
    all_image_paths = dimension_config.get('image_paths', [])
    
    if not all_image_paths:
        await sio.emit('error', {'message': f'No image paths for dimension {dimension}'}, room=sid)
        return
    
    # Filter by dataset
    dataset_dir = 'chest_xray/train' if dataset == 'train' else 'chest_xray/test'
    dataset_paths = [p for p in all_image_paths if dataset_dir in p]
    
    # Compute ranges
    ranges = compute_metric_ranges(dataset_paths, dataset)
    
    await sio.emit('metric_ranges', {
        'dimension': dimension,
        'dataset': dataset,
        'ranges': ranges
    }, room=sid)

@sio.event
async def get_metric_count(sid, data):
    """
    Count images that pass a specific metric filter.
    
    Args:
        data: dict with 'config_file', 'dimension', 'dataset', 'class_filter', 
              'metric_key', 'min_val', 'max_val'
    
    Returns:
        Emits 'metric_count' with count
    """
    config_file = data.get('config_file')
    dimension = data.get('dimension')
    dataset = data.get('dataset', 'train')
    class_filter = data.get('class_filter', 'all')
    metric_key = data.get('metric_key')
    min_val = float(data.get('min_val'))
    max_val = float(data.get('max_val'))
    
    if not all([config_file, dimension, metric_key is not None]):
        await sio.emit('metric_count', {'count': 0, 'metric_key': metric_key}, room=sid)
        return
    
    # Load config
    config_path = os.path.join("static", config_file)
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception:
        await sio.emit('metric_count', {'count': 0, 'metric_key': metric_key}, room=sid)
        return
    
    if 'dimensions' not in config or dimension not in config['dimensions']:
        await sio.emit('metric_count', {'count': 0, 'metric_key': metric_key}, room=sid)
        return
    
    dimension_config = config['dimensions'][dimension]
    all_image_paths = dimension_config.get('image_paths', [])
    
    # Filter by dataset
    dataset_dir = 'chest_xray/train' if dataset == 'train' else 'chest_xray/test'
    all_image_paths = [p for p in all_image_paths if dataset_dir in p]
    
    # Filter by class
    if class_filter != 'all':
        all_image_paths = [p for p in all_image_paths if f'/{class_filter}/' in p]
    
    # Count images within metric range
    count = 0
    for path in all_image_paths:
        normalized_path = path.replace("\\", "/")
        du_data = DU_LOOKUP.get(normalized_path, {})
        
        if metric_key in du_data:
            metric_val = float(du_data[metric_key])
            if min_val <= metric_val <= max_val:
                count += 1
    
    await sio.emit('metric_count', {
        'count': count,
        'metric_key': metric_key
    }, room=sid)


@sio.event
async def load_config(sid, filename=None):
    """Load configuration from JSON file"""
    if filename:
        config_path = os.path.join("static", filename)
    else:
        config_path = JSON_CONFIG_PATH
    
    if not config_path:
        await sio.emit('error', {'message': 'No configuration file available'}, room=sid)
        return
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        await sio.emit('config_loaded', config, room=sid)
    except FileNotFoundError:
        await sio.emit('error', {'message': f'Configuration file {filename or "default"} not found'}, room=sid)
    except Exception as e:
        await sio.emit('error', {'message': f'Error loading config: {str(e)}'}, room=sid)


@sio.event
async def save_config_data(sid, data):
    """Save configuration to JSON file"""
    try:
        filename = data.get('filename')
        config = data.get('config')
        
        if not filename or not config:
            await sio.emit('error', {'message': 'Invalid save request'}, room=sid)
            return
        
        config_path = os.path.join("static", filename)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        await sio.emit('config_saved', {'success': True}, room=sid)
    except Exception as e:
        await sio.emit('error', {'message': f'Failed to save: {str(e)}'}, room=sid)


@sio.event
async def get_dimension_images(sid, data):
    """
    Get images for a specific dimension using exact paths from config.
    
    Args:
        data: dict with 'config_file', 'dimension', 'dataset', 'class_filter',
        'params', 'du_filters', 'aug_params', 'offset', and 'limit'
    """
    config_file = data.get('config_file')
    dimension = data.get('dimension')
    dataset = data.get('dataset', 'train')
    class_filter = data.get('class_filter', 'all')
    params = data.get('params', {})
    du_filters = data.get('du_filters', {})
    aug_params = data.get('aug_params', {})
    offset = data.get('offset', 0)
    limit = data.get('limit', 18)
    
    if not config_file:
        await sio.emit('error', {'message': 'No config file specified'}, room=sid)
        return
    
    if not dimension:
        await sio.emit('error', {'message': 'No dimension specified'}, room=sid)
        return
    
    # Load the specified config file
    config_path = os.path.join("static", config_file)
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        await sio.emit('error', {'message': f'Failed to load config: {str(e)}'}, room=sid)
        return
    
    if 'dimensions' not in config or dimension not in config['dimensions']:
        await sio.emit('error', {'message': f'Dimension {dimension} not found in config {config_file}'}, room=sid)
        return
    
    dimension_config = config['dimensions'][dimension]
    all_image_paths = dimension_config.get('image_paths', [])
    
    if not all_image_paths:
        await sio.emit('error', {'message': f'No image paths stored for dimension {dimension}'}, room=sid)
        return
    
    # Filter by dataset (train or test)
    dataset_dir = 'chest_xray/train' if dataset == 'train' else 'chest_xray/test'
    all_image_paths = [p for p in all_image_paths if dataset_dir in p]
    
    # Store total count before class filtering
    total_before_class_filter = len(all_image_paths)
    
    # Filter by class if needed
    if class_filter != 'all':
        all_image_paths = [p for p in all_image_paths if f'/{class_filter}/' in p]
    
    # Store total count before DU filtering
    total_before_du_filter = len(all_image_paths)
    
    # Apply DU metric filters
    if du_filters:
        all_image_paths = apply_du_filters(all_image_paths, du_filters)
    
    # Store total count after all filtering
    total_filtered = len(all_image_paths)
    
    # Check if augmentation is active
    aug_active = False
    if aug_params:
        # Check if any augmentation parameter is set
        aug_active = any(
            (k in ['horizontal_flip', 'vertical_flip'] and v) or 
            (k not in ['horizontal_flip', 'vertical_flip'] and v != 0)
            for k, v in aug_params.items()
        )
    
    # If augmentation is active, adjust pagination
    # Show 3 originals + 5 variations each = 18 cells
    if aug_active:
        num_originals = 3
        paginated_paths = all_image_paths[offset:offset + num_originals]
    else:
        paginated_paths = all_image_paths[offset:offset + limit]
    
    # Handle zero images edge case
    if len(paginated_paths) == 0:
        await sio.emit('images_loaded', {
            'dimension': dimension,
            'dataset': dataset,
            'count': 0,
            'total_available': total_before_du_filter,
            'filtered_count': total_filtered,
            'aug_active': aug_active
        }, room=sid)
        return
    
    # Load and process images
    images_data = []
    cell_idx = 0
    
    for path in paginated_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        # Apply preprocessing
        processed = apply_preprocessing(img, params)
        
        # Retrieve DU metrics for this image
        normalized_path = path.replace("\\", "/")
        du_info = get_du_metrics(normalized_path)
        filename = os.path.basename(path)
        
        if aug_active:
            # Send original first
            success, buffer = cv2.imencode('.png', processed)
            if success:
                images_data.append({
                    'index': cell_idx,
                    'data': buffer.tobytes(),
                    'filename': filename,
                    'path': normalized_path,
                    'du': du_info,
                    'is_augmented': False
                })
                cell_idx += 1
            
            # Generate 5 augmented variations
            # This loop relies on the randomness inside apply_augmentation being stateful (i.e., changing on each call)
            # The seed is effectively controlled by the default random state for each Sequential call.
            for var_idx in range(5):
                augmented = apply_augmentation(processed, aug_params)
                success, buffer = cv2.imencode('.png', augmented)
                if success:
                    images_data.append({
                        'index': cell_idx,
                        'data': buffer.tobytes(),
                        'filename': filename,
                        'path': normalized_path,
                        'du': du_info,
                        'is_augmented': True
                    })
                    cell_idx += 1
        else:
            # No augmentation - send original only
            success, buffer = cv2.imencode('.png', processed)
            if success:
                images_data.append({
                    'index': cell_idx,
                    'data': buffer.tobytes(),
                    'filename': filename,
                    'path': normalized_path,
                    'du': du_info,
                    'is_augmented': False
                })
                cell_idx += 1
    
    # Send info about how many images were loaded
    await sio.emit('images_loaded', {
        'dimension': dimension,
        'dataset': dataset,
        'count': len(images_data),
        'total_available': total_before_du_filter,
        'filtered_count': total_filtered,
        'aug_active': aug_active
    }, room=sid)
    
    # Send the images one by one
    for img_data in images_data:
        await sio.emit('image_data', img_data, room=sid)


@sio.event
async def update_preview(sid, data):
    """
    Update preview with new parameters.
    
    Args:
        data: dict with 'dimension', 'dataset', and 'params'
    """
    await get_dimension_images(sid, data)


# REST API endpoints

@app.get("/")
async def root():
    """Serve the main UI page"""
    return FileResponse("dlcv_p2_preprocessing_ui.html")


@app.get("/api/config")
async def get_config():
    """Get current configuration"""
    config = load_config_file()
    if config:
        return config
    return {"error": "Configuration not found"}, 404


@app.post("/api/config")
async def update_config(data: dict):
    """Update configuration"""
    try:
        save_config_file(data)
        return {"success": True}
    except Exception as e:
        return {"error": str(e)}, 500


@app.get("/api/dimensions")
async def get_dimensions():
    """Get list of all dimensions"""
    config = load_config_file()
    if config and 'dimensions' in config:
        return list(config['dimensions'].keys())
    return []


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 70)
    print("DLCV Preprocessing Configuration Server")
    print("=" * 70)
    print("\nStarting server...")
    print("Open your browser to: http://localhost:8000")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 70)
    
    # Check if config file exists
    if not JSON_CONFIG_PATH or not os.path.exists(JSON_CONFIG_PATH):
        print(f"\nWARNING: No configuration files found in 'static/' folder!")
        print("Please run: python dlcv_p2_analyze_dimensions.py [tolerance]")
        print("Example: python dlcv_p2_analyze_dimensions.py 15")
        print()
    
    uvicorn.run(
        socket_app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
