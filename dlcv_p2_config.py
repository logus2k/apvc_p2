"""
Deep Learning for Computer Vision - Project 2
Global Configuration and Constants

This module contains global configuration parameters used across the project.
"""

import os
import random
import numpy as np

# Suppress TensorFlow info messages
# 0=all, 1=filter INFO, 2=filter WARNING, 3=filter ERROR
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from keras import mixed_precision


# Random seed for reproducibility
SEED = 42

# Set random seeds across all libraries
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Data paths
DATA_ROOT = "./chest_xray"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")

# Class names
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

# Training configuration
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 50

# Model checkpoint
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Results directory
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Sweep results directory
SWEEP_RESULTS_DIR = os.path.join(RESULTS_DIR, "sweeps")
os.makedirs(SWEEP_RESULTS_DIR, exist_ok=True)

# Validation split ratio
VAL_SPLIT = 0.10

# Device configuration
def get_device_info():
    """Print device and GPU information"""
    print("=" * 60)
    print("DEVICE CONFIGURATION")
    print("=" * 60)
    
    # Check for GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPUs available: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
            # Get GPU memory info if available
            try:
                gpu_details = tf.config.experimental.get_device_details(gpu)
                print(f"    Details: {gpu_details}")
            except:
                pass
    else:
        print("No GPUs available. Using CPU.")
    
    print(f"TensorFlow version: {tf.__version__}")
    print("=" * 60)


# Enable mixed precision for RTX 4090
def enable_mixed_precision():
    """Enable mixed precision training for faster computation on modern GPUs"""
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print("Mixed precision enabled: Compute in float16, variables in float32")


# Memory growth for GPU (prevents TF from allocating all VRAM)
def configure_gpu_memory():
    """Configure GPU memory growth to avoid OOM errors"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled")
        except RuntimeError as e:
            print(f"GPU memory growth configuration error: {e}")


if __name__ == "__main__":
    # Test configuration
    get_device_info()
    configure_gpu_memory()
