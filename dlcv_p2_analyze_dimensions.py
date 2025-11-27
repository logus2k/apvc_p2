"""
Analyze dataset dimensions and propose binning strategy with configurable variation
"""

import os
import json
import cv2
import numpy as np
from typing import Optional
from collections import defaultdict
from dlcv_p2_config import TRAIN_DIR, TEST_DIR, CLASS_NAMES


class DimensionAnalyzer:
    """Analyzes dataset dimensions and creates bins with configurable tolerance"""
    
    def __init__(self, tolerance_percent: float = 5.0):
        """
        Initialize dimension analyzer.
        
        Args:
            tolerance_percent: Percentage tolerance for binning (default: 5.0%)
                              Use 0 to create a single bin with all images.
        """
        self.tolerance_percent = tolerance_percent
        self.dimensions = []
        self.bins = []
        # Include tolerance in filename
        if tolerance_percent == 0:
            tolerance_str = "all"
        else:
            tolerance_str = str(int(tolerance_percent)) if tolerance_percent == int(tolerance_percent) else str(tolerance_percent).replace('.', '_')
        self.json_output_path = f"static/dlcv_p2_dataset_dimensions_{tolerance_str}.json"
    
    def scan_all_dimensions(self):
        """Scan all images from both TRAIN and TEST datasets"""
        print("Scanning all images in TRAIN and TEST datasets...")
        
        self.dimensions = []
        
        # Scan both datasets
        for dataset_name, dataset_dir in [('TRAIN', TRAIN_DIR), ('TEST', TEST_DIR)]:
            print(f"\n  {dataset_name} Dataset:")
            for class_name in CLASS_NAMES:
                class_dir = os.path.join(dataset_dir, class_name)
                if not os.path.exists(class_dir):
                    continue
                
                image_files = [f for f in os.listdir(class_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                print(f"    Scanning {class_name}: {len(image_files)} images")
                
                for img_file in image_files:
                    img_path = os.path.join(class_dir, img_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        h, w = img.shape
                        self.dimensions.append((w, h))
        
        print(f"\nTotal images scanned: {len(self.dimensions)}")
        return self.dimensions
    
    def create_bins(self):
        """
        Create bins with configured tolerance.
        
        Special case: If tolerance_percent is 0, create a single bin containing all images.
        
        Strategy: Group dimensions where both width and height are within 
        tolerance_percent of each other.
        """
        print("\n" + "=" * 70)
        
        if self.tolerance_percent == 0:
            print("CREATING SINGLE BIN WITH ALL IMAGES (tolerance=0)")
        else:
            print(f"CREATING BINS WITH {self.tolerance_percent}% VARIATION")
        
        print("=" * 70)
        
        if not self.dimensions:
            print("No dimensions loaded. Run scan_all_dimensions() first.")
            return []
        
        # Extract widths and heights
        widths = np.array([d[0] for d in self.dimensions])
        heights = np.array([d[1] for d in self.dimensions])
        
        print(f"\nWidth range: {widths.min()} - {widths.max()} (mean: {widths.mean():.0f})")
        print(f"Height range: {heights.min()} - {heights.max()} (mean: {heights.mean():.0f})")
        
        # Special case: tolerance=0 means all images in one bin
        if self.tolerance_percent == 0:
            self.bins = [{
                'center': (int(widths.mean()), int(heights.mean())),
                'count': len(self.dimensions),
                'dimensions': self.dimensions.copy(),
                'indices': list(range(len(self.dimensions)))
            }]
            print(f"\nCreated 1 bin containing all {len(self.dimensions)} images")
            return self.bins
        
        # Create bins using k-means-like clustering with tolerance
        self.bins = []
        used_indices = set()
        
        # Sort by width*height (area) to process systematically
        areas = widths * heights
        sorted_indices = np.argsort(areas)
        
        tolerance_fraction = self.tolerance_percent / 100.0
        
        for idx in sorted_indices:
            if idx in used_indices:
                continue
            
            w_center, h_center = self.dimensions[idx]
            
            # Define tolerance
            w_tolerance = w_center * tolerance_fraction
            h_tolerance = h_center * tolerance_fraction
            
            # Find all dimensions within tolerance of this center
            w_min, w_max = w_center - w_tolerance, w_center + w_tolerance
            h_min, h_max = h_center - h_tolerance, h_center + h_tolerance
            
            # Collect all images in this bin
            bin_indices = []
            for i, (w, h) in enumerate(self.dimensions):
                if i in used_indices:
                    continue
                if w_min <= w <= w_max and h_min <= h <= h_max:
                    bin_indices.append(i)
                    used_indices.add(i)
            
            if len(bin_indices) > 0:
                bin_dimensions = [self.dimensions[i] for i in bin_indices]
                self.bins.append({
                    'center': (w_center, h_center),
                    'count': len(bin_indices),
                    'dimensions': bin_dimensions,
                    'indices': bin_indices
                })
        
        # Sort bins by count (most common first)
        self.bins = sorted(self.bins, key=lambda x: x['count'], reverse=True)
        
        return self.bins
    
    def analyze_bins(self):
        """Analyze and display bin statistics"""
        if not self.bins:
            print("No bins created. Run create_bins() first.")
            return
        
        print(f"\nCreated {len(self.bins)} bins with {self.tolerance_percent}% variation tolerance")
        print("\n" + "=" * 70)
        print("TOP 30 BINS (by image count)")
        print("=" * 70)
        print(f"{'Rank':<6} {'Center (W×H)':<20} {'Count':<8} {'% of Total':<12} {'W Range':<20} {'H Range':<20}")
        print("-" * 70)
        
        total_images = len(self.dimensions)
        cumulative_pct = 0
        
        for rank, bin_info in enumerate(self.bins[:30], 1):
            w_center, h_center = bin_info['center']
            count = bin_info['count']
            pct = 100 * count / total_images
            cumulative_pct += pct
            
            # Calculate actual range in this bin
            bin_widths = [d[0] for d in bin_info['dimensions']]
            bin_heights = [d[1] for d in bin_info['dimensions']]
            
            w_range = f"{min(bin_widths)}-{max(bin_widths)}"
            h_range = f"{min(bin_heights)}-{max(bin_heights)}"
            
            print(f"{rank:<6} {w_center}×{h_center:<13} {count:<8} {pct:>6.2f}%      {w_range:<20} {h_range:<20}")
        
        print("-" * 70)
        print(f"Top 30 bins cover: {cumulative_pct:.1f}% of dataset")
        
        # Analyze bin size distribution
        bin_sizes = [b['count'] for b in self.bins]
        
        print(f"\n" + "=" * 70)
        print("BIN SIZE STATISTICS")
        print("=" * 70)
        print(f"Total bins: {len(self.bins)}")
        print(f"Bins with 1 image: {sum(1 for s in bin_sizes if s == 1)}")
        print(f"Bins with 2-4 images: {sum(1 for s in bin_sizes if 2 <= s <= 4)}")
        print(f"Bins with 5-9 images: {sum(1 for s in bin_sizes if 5 <= s <= 9)}")
        print(f"Bins with 10+ images: {sum(1 for s in bin_sizes if s >= 10)}")
        print(f"Bins with 20+ images: {sum(1 for s in bin_sizes if s >= 20)}")
        
        print(f"\nLargest bin: {max(bin_sizes)} images")
        print(f"Mean bin size: {np.mean(bin_sizes):.1f} images")
        print(f"Median bin size: {np.median(bin_sizes):.0f} images")
    
    def recommend_strategy(self):
        """Recommend analysis strategy based on bin distribution"""
        if not self.bins:
            print("No bins created. Run create_bins() first.")
            return
        
        print(f"\n" + "=" * 70)
        print("RECOMMENDATIONS FOR AUGMENTATION ANALYSIS")
        print("=" * 70)
        
        bins_with_5plus = [b for b in self.bins if b['count'] >= 5]
        bins_with_10plus = [b for b in self.bins if b['count'] >= 10]
        bins_with_20plus = [b for b in self.bins if b['count'] >= 20]
        
        print(f"\nOption 1: Analyze bins with ≥5 images")
        print(f"  → {len(bins_with_5plus)} bins")
        print(f"  → Covers {sum(b['count'] for b in bins_with_5plus)} images")
        
        print(f"\nOption 2: Analyze bins with ≥10 images")
        print(f"  → {len(bins_with_10plus)} bins")
        print(f"  → Covers {sum(b['count'] for b in bins_with_10plus)} images")
        
        print(f"\nOption 3: Analyze bins with ≥20 images")
        print(f"  → {len(bins_with_20plus)} bins")
        print(f"  → Covers {sum(b['count'] for b in bins_with_20plus)} images")
        
        print(f"\nOption 4: Top 50 bins (by count)")
        print(f"  → 50 bins")
        print(f"  → Covers {sum(b['count'] for b in self.bins[:50])} images")
        
        print(f"\nOption 5: Top 20 bins (by count)")
        print(f"  → 20 bins")
        print(f"  → Covers {sum(b['count'] for b in self.bins[:20])} images")
        
        print(f"\n" + "=" * 70)
    
    def export_to_json(self, output_path: Optional[str] = None):
        """
        Export dimension bins to JSON configuration file.
        
        Args:
            output_path: Path to output JSON file (default: dlcv_p2_dataset_dimensions.json)
        """
        if not self.bins:
            print("No bins created. Run create_bins() first.")
            return
        
        if output_path is None:
            output_path = self.json_output_path
        
        # Ensure static directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
        
        print(f"\n" + "=" * 70)
        print("EXPORTING TO JSON")
        print("=" * 70)
        
        # Collect image paths for each bin from both TRAIN and TEST
        all_image_paths = {}  # {(w, h): [paths]}
        
        for dataset_name, dataset_dir in [('TRAIN', TRAIN_DIR), ('TEST', TEST_DIR)]:
            for class_name in ['NORMAL', 'PNEUMONIA']:
                class_dir = os.path.join(dataset_dir, class_name)
                if not os.path.exists(class_dir):
                    continue
                
                for filename in os.listdir(class_dir):
                    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue
                    
                    filepath = os.path.join(class_dir, filename)
                    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        h, w = img.shape
                        key = (w, h)
                        if key not in all_image_paths:
                            all_image_paths[key] = []
                        all_image_paths[key].append(filepath)
        
        # Create JSON structure
        json_data = {
            "global_settings": {
                "allowed_variation": self.tolerance_percent,
                "h_crop": True,
                "v_crop": True,
                "clahe": True,
                "zoom_in": True,
                "target_size": [224, 224]
            },
            "dimensions": {}
        }
        
        # Process each bin - match exact images that were binned together
        for bin_info in self.bins:
            w_center, h_center = bin_info['center']
            bin_dimensions = bin_info['dimensions']  # List of (w, h) tuples in this bin
            
            # Deduplicate dimensions - same dimension can appear multiple times in self.dimensions
            unique_dimensions = list(set(bin_dimensions))
            
            # Collect all image paths for dimensions in this bin
            bin_image_paths = []
            for (w, h) in unique_dimensions:
                if (w, h) in all_image_paths:
                    bin_image_paths.extend(all_image_paths[(w, h)])
            
            if not bin_image_paths:
                continue
            
            # Sort paths to get consistent first and last
            bin_image_paths.sort()
            
            # Create dimension key from first and last image filenames
            first_filename = os.path.basename(bin_image_paths[0]).split('.')[0]
            last_filename = os.path.basename(bin_image_paths[-1]).split('.')[0]
            dimension_key = f"{first_filename}_to_{last_filename}"
            
            # Calculate crop values (to make square) using center dimensions
            if w_center > h_center:
                h_crop = (w_center - h_center) // 2
                v_crop = 0
            elif h_center > w_center:
                h_crop = 0
                v_crop = (h_center - w_center) // 2
            else:
                h_crop = 0
                v_crop = 0
            
            json_data["dimensions"][dimension_key] = {
                "count": len(bin_image_paths),
                "center_dimension": f"{w_center}x{h_center}",
                "image_paths": bin_image_paths,
                "h_crop": 0,
                "v_crop": 0,
                "clahe": 0,
                "zoom_in": 0
            }
        
        # Write to file
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"\nExported {len(json_data['dimensions'])} dimension configurations to: {output_path}")
        print(f"Total images covered: {sum(d['count'] for d in json_data['dimensions'].values())}")
        print("\nYou can now:")
        print(f"  1. Edit {output_path} to adjust preprocessing parameters per dimension")
        print(f"  2. Run preprocessing: python dlcv_p2_preprocess_dataset.py")
        print("=" * 70)
    
    def run_full_analysis(self, export_json: bool = True):
        """
        Run complete analysis pipeline.
        
        Args:
            export_json: Whether to export results to JSON (default: True)
        """
        self.scan_all_dimensions()
        self.create_bins()
        self.analyze_bins()
        self.recommend_strategy()
        
        if export_json:
            self.export_to_json()


def analyze_dimensions(tolerance_percent: float = 5.0, export_json: bool = True):
    """
    Main function to analyze dimensions with configurable tolerance.
    
    Args:
        tolerance_percent: Percentage tolerance for binning (default: 5.0%)
        export_json: Whether to export results to JSON (default: True)
    
    Returns:
        DimensionAnalyzer instance with analysis results
    """
    analyzer = DimensionAnalyzer(tolerance_percent=tolerance_percent)
    analyzer.run_full_analysis(export_json=export_json)
    return analyzer


if __name__ == "__main__":
    import sys
    
    # Check for tolerance argument
    tolerance_percent = 5.0  # default
    output_json = None  # Use class default (static/dlcv_p2_dataset_dimensions.json)
    
    if len(sys.argv) > 1:
        try:
            tolerance_percent = float(sys.argv[1])
            print(f"Using {tolerance_percent}% tolerance (from command line argument)")
        except ValueError:
            print(f"Invalid tolerance value: {sys.argv[1]}")
            print("Usage: python dlcv_p2_analyze_dimensions.py [tolerance_percent] [output_json_path]")
            print("Example: python dlcv_p2_analyze_dimensions.py 10")
            print("Example: python dlcv_p2_analyze_dimensions.py 10 my_config.json")
            sys.exit(1)
    else:
        print(f"Using default {tolerance_percent}% tolerance")
        print("Usage: python dlcv_p2_analyze_dimensions.py [tolerance_percent] [output_json_path]")
        print()
    
    if len(sys.argv) > 2:
        output_json = sys.argv[2]
        print(f"Will save to: {output_json}")
    
    # Run analysis
    analyzer = DimensionAnalyzer(tolerance_percent=tolerance_percent)
    if output_json:
        analyzer.json_output_path = output_json
    analyzer.run_full_analysis(export_json=True)
    
    print("\nAnalysis complete!")
