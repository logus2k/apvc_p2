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
        json_files = sorted([f for f in os.listdir(static_dir) if f.endswith('.json')])
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
dimension_cache = {}  # Cache: {dataset: {dimension: [paths]}}


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
            files = [f for f in os.listdir(static_dir) if f.endswith('.json')]
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
        data: dict with 'config_file', 'dimension', 'dataset', 'class_filter', 'params', 'offset', and 'limit'
    """
    config_file = data.get('config_file')
    dimension = data.get('dimension')
    dataset = data.get('dataset', 'train')
    class_filter = data.get('class_filter', 'all')
    params = data.get('params', {})
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
    
    # Filter by class if needed
    if class_filter != 'all':
        all_image_paths = [p for p in all_image_paths if f'/{class_filter}/' in p]
    
    # Apply pagination
    paginated_paths = all_image_paths[offset:offset + limit]
    
    # Load and process images
    images_data = []
    
    for idx, path in enumerate(paginated_paths):
        # Load image
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        # Apply preprocessing
        processed = apply_preprocessing(img, params)
        
        # Encode as PNG
        success, buffer = cv2.imencode('.png', processed)
        if success:
            images_data.append({
                'index': idx,
                'data': buffer.tobytes(),
                'filename': os.path.basename(path)
            })
    
    # Send images info - use exact count from filtered paths
    await sio.emit('images_loaded', {
        'dimension': dimension,
        'dataset': dataset,
        'count': len(images_data),
        'total_available': len(all_image_paths)
    }, room=sid)
    
    # Send each image separately as binary
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
