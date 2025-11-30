#!/usr/bin/env python3
"""
Corrected Efficiency Comparison Script for DL Models

This script properly handles models with high recall and generates accurate visualizations
"""

import os
import re
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import nbformat


def extract_threshold_data_from_notebook(notebook_path):
    """
    Extract threshold analysis report from Jupyter notebook
    """
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
        
        # Look for the threshold analysis report in code cell outputs
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                if hasattr(cell, 'outputs'):
                    for output in cell.outputs:
                        if output.output_type == 'stream' and hasattr(output, 'text'):
                            text_content = ''.join(output.text) if isinstance(output.text, list) else output.text
                            if 'THRESHOLD ANALYSIS' in text_content and 'True Positives' in text_content:
                                return text_content
                        elif output.output_type == 'display_data' and 'text/plain' in output.data:
                            text_content = output.data['text/plain']
                            if isinstance(text_content, str) and 'THRESHOLD ANALYSIS' in text_content:
                                return text_content
        
        print(f"Warning: No threshold analysis found in {notebook_path}")
        return None
        
    except Exception as e:
        print(f"Error reading {notebook_path}: {e}")
        return None

def extract_threshold_data_from_report(report_text):
    """
    Parse threshold analysis report and extract FN/FP data
    """
    if not report_text:
        return None
        
    # Pattern to match threshold sections with FN and FP counts
    pattern = r'Threshold:\s*([0-9.]+)[\s\S]*?False Negatives.*?(\d+)[\s\S]*?False Positives.*?(\d+)'
    matches = re.findall(pattern, report_text)
    
    if not matches:
        print("Warning: Could not extract threshold data from report")
        return None
    
    data = {}
    for match in matches:
        threshold = float(match[0])
        fn = int(match[1])
        fp = int(match[2])
        data[threshold] = {'FN': fn, 'FP': fp}
    
    return data

def process_notebook_list(notebook_list):
    """
    Process a specific list of notebooks
    """
    all_model_data = {}
    
    for notebook_path in notebook_list:
        notebook_path = Path(notebook_path)
        
        if not notebook_path.exists():
            print(f"Warning: Notebook does not exist: {notebook_path}")
            continue
        
        print(f"Processing: {notebook_path.name}")
        
        # Extract report from notebook
        report_text = extract_threshold_data_from_notebook(notebook_path)
        
        if report_text:
            # Extract threshold data
            threshold_data = extract_threshold_data_from_report(report_text)
            
            if threshold_data:
                # Create model name from filename
                model_name = notebook_path.stem
                if model_name.startswith('dl_for_cv_project2_FINAL_v'):
                    model_name = model_name.replace('dl_for_cv_project2_FINAL_v', 'Model_')
                
                all_model_data[model_name] = {
                    'threshold_data': threshold_data,
                    'source_file': str(notebook_path)
                }
                print(f"  - Successfully processed {len(threshold_data)} thresholds")
            else:
                print(f"  - Could not extract threshold data")
        else:
            print(f"  - No threshold analysis found")
    
    return all_model_data

def process_notebooks_in_folder(folder_path, pattern="dl_for_cv_project2_FINAL_v*"):
    """
    Process all notebooks in the folder matching the pattern
    """
    folder = Path(folder_path)
    
    # Find all matching notebooks
    notebooks = list(folder.glob(pattern + ".ipynb"))
    
    if not notebooks:
        # Also try without .ipynb extension
        notebooks = list(folder.glob(pattern))
        notebooks = [nb for nb in notebooks if nb.suffix == '.ipynb']
    
    print(f"Found {len(notebooks)} notebooks to process")
    return process_notebook_list(notebooks)

def plot_efficiency_comparison(all_model_data, output_file=None):
    """
    Create comprehensive efficiency comparison charts
    """
    if not all_model_data:
        print("No model data to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. FN vs FP at Each Threshold (Scatter plot)
    ax1 = axes[0, 0]
    for model_name, data in all_model_data.items():
        threshold_data = data['threshold_data']
        fns = [v['FN'] for v in threshold_data.values()]
        fps = [v['FP'] for v in threshold_data.values()]
        thresholds = list(threshold_data.keys())
        
        # Sort by threshold to ensure proper order
        sorted_data = sorted(zip(thresholds, fns, fps), key=lambda x: x[0])
        sorted_thresholds, sorted_fns, sorted_fps = zip(*sorted_data)
        
        scatter = ax1.scatter(sorted_fps, sorted_fns, label=model_name, s=60, alpha=0.7)
        
        # Add threshold labels for all points
        for fn, fp, thresh in zip(sorted_fns, sorted_fps, sorted_thresholds):
            ax1.annotate(f'{thresh}', (fp, fn), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8)
    
    ax1.set_xlabel('False Positives')
    ax1.set_ylabel('False Negatives')
    ax1.set_title('FN vs FP Trade-off (Lower Left = Better)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()  # Lower FP is better
    ax1.invert_yaxis()  # Lower FN is better
    
    # 2. FP Reduction Analysis (Focus on models that maintain low FNs)
    ax2 = axes[0, 1]
    for model_name, data in all_model_data.items():
        threshold_data = data['threshold_data']
        thresholds = sorted(threshold_data.keys(), reverse=True)
        fns = [threshold_data[t]['FN'] for t in thresholds]
        fps = [threshold_data[t]['FP'] for t in thresholds]
        
        # Calculate FP reduction from threshold 0.1 (first threshold)
        fp_01 = fps[0]  # FP at threshold 0.1
        fp_reduction = [fp_01 - fp for fp in fps]  # Positive values = FP reduction
        
        ax2.plot(fp_reduction, fns, marker='o', label=model_name, linewidth=2)
    
    ax2.set_xlabel('FP Reduction from Threshold 0.1')
    ax2.set_ylabel('False Negatives')
    ax2.set_title('FP Reduction vs FN Maintenance')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 3. Threshold Performance Comparison
    ax3 = axes[1, 0]
    ax3_twin = ax3.twinx()
    
    lines1, labels1 = [], []
    lines2, labels2 = [], []
    
    for model_name, data in all_model_data.items():
        threshold_data = data['threshold_data']
        if not threshold_data:
            continue
            
        thresholds = sorted(threshold_data.keys())
        fns = [threshold_data[t]['FN'] for t in thresholds]
        fps = [threshold_data[t]['FP'] for t in thresholds]
        
        # Plot both FN and FP on same graph with different scales
        line1 = ax3.plot(thresholds, fns, 'o-', label=f'{model_name} FNs', linewidth=2)
        line2 = ax3_twin.plot(thresholds, fps, 's-', label=f'{model_name} FPs', linewidth=2)
        
        # Collect handles and labels for combined legend
        lines1.extend(line1)
        labels1.extend([f'{model_name} FNs'])
        lines2.extend(line2)
        labels2.extend([f'{model_name} FPs'])
    
    ax3.set_xlabel('Threshold')
    ax3.set_ylabel('False Negatives', color='blue')
    ax3_twin.set_ylabel('False Positives', color='red')
    ax3.set_title('FN/FP vs Threshold')
    ax3.grid(True, alpha=0.3)
    
    # Combine legends
    ax3.legend(lines1 + lines2, labels1 + labels2, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 4. Practical Comparison (Focus on your clinical needs)
    ax4 = axes[1, 1]
    for model_name, data in all_model_data.items():
        threshold_data = data['threshold_data']
        if not threshold_data:
            continue
            
        thresholds = sorted(threshold_data.keys(), reverse=True)
        
        # Focus on thresholds where FN <= 2 (maintain high recall)
        valid_data = [(t, threshold_data[t]['FN'], threshold_data[t]['FP']) 
                     for t in thresholds if threshold_data[t]['FN'] <= 2]
        
        if valid_data:
            valid_thresholds, valid_fns, valid_fps = zip(*valid_data)
            ax4.plot(valid_fps, valid_fns, marker='o', label=model_name, linewidth=2)
    
    ax4.set_xlabel('False Positives (FN ≤ 2)')
    ax4.set_ylabel('False Negatives')
    ax4.set_title('Performance with High Recall Maintained')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {output_file}")
    else:
        plt.show()

def create_efficiency_summary_table(all_model_data):
    """
    Create summary table focusing on practical metrics for your use case
    """
    if not all_model_data:
        return pd.DataFrame()
    
    summary_data = []
    
    for model_name, data in all_model_data.items():
        threshold_data = data['threshold_data']
        if not threshold_data:
            continue
        
        # Find best performance where recall is maintained high (FN <= 2)
        best_low_fn = None
        for threshold in sorted(threshold_data.keys(), reverse=True):
            fn, fp = threshold_data[threshold]['FN'], threshold_data[threshold]['FP']
            if fn <= 2:  # High recall maintained
                if best_low_fn is None or fp < best_low_fn['FP']:
                    best_low_fn = {'threshold': threshold, 'FN': fn, 'FP': fp}
        
        # Find overall best FP performance regardless of FN
        min_fp = min(threshold_data.values(), key=lambda x: x['FP'])
        min_fp_threshold = [k for k, v in threshold_data.items() if v == min_fp][0]
        
        # Find overall best FN performance (lowest FN)
        min_fn = min(threshold_data.values(), key=lambda x: x['FN'])
        min_fn_threshold = [k for k, v in threshold_data.items() if v == min_fn][0]
        
        summary_data.append({
            'Model': model_name,
            'Best_High_Recall_Threshold': best_low_fn['threshold'] if best_low_fn else 'N/A',
            'Best_High_Recall_FP': best_low_fn['FP'] if best_low_fn else float('inf'),
            'Best_High_Recall_FN': best_low_fn['FN'] if best_low_fn else float('inf'),
            'Min_FP_Threshold': min_fp_threshold,
            'Min_FP_Count': min_fp['FP'],
            'Min_FP_FN': min_fp['FN'],
            'Min_FN_Threshold': min_fn_threshold,
            'Min_FN_Count': min_fn['FN'],
            'Min_FN_FP': min_fn['FP']
        })
    
    df = pd.DataFrame(summary_data)
    if not df.empty:
        # Sort by best high recall performance (lowest FP when FN <= 2)
        df['sort_key'] = df['Best_High_Recall_FP'].apply(lambda x: x if x != float('inf') else 999999)
        df = df.sort_values('sort_key').drop('sort_key', axis=1)
    return df

def main():
    parser = argparse.ArgumentParser(description='Compare efficiency of DL models from Jupyter notebooks')
    parser.add_argument('--folder', '-f', help='Path to folder containing notebooks (with pattern matching)')
    parser.add_argument('--notebooks', '-n', nargs='+', help='List of specific notebook files to process')
    parser.add_argument('--pattern', default='dl_for_cv_project2_FINAL_v*', 
                       help='Pattern for notebook files (used with --folder)')
    parser.add_argument('--output', '-o', help='Output chart file (e.g., comparison.png)')
    parser.add_argument('--csv', help='Output CSV summary file')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.folder and not args.notebooks:
        print("Error: Must specify either --folder or --notebooks")
        parser.print_help()
        return
    
    if args.folder and args.notebooks:
        print("Error: Cannot specify both --folder and --notebooks")
        parser.print_help()
        return
    
    # Process notebooks based on input type
    all_model_data = {}
    
    if args.folder:
        # Validate folder exists
        if not os.path.exists(args.folder):
            print(f"Error: Folder does not exist: {args.folder}")
            return
        
        print(f"Processing notebooks in: {args.folder} with pattern: {args.pattern}")
        all_model_data = process_notebooks_in_folder(args.folder, args.pattern)
    
    elif args.notebooks:
        print(f"Processing specific notebooks: {args.notebooks}")
        all_model_data = process_notebook_list(args.notebooks)
    
    if not all_model_data:
        print("No valid model data found in notebooks")
        return
    
    print(f"\nProcessed {len(all_model_data)} models successfully")
    
    # Create summary table
    summary_df = create_efficiency_summary_table(all_model_data)
    print("\nModel Efficiency Summary:")
    print(summary_df.to_string(index=False))
    
    # Find best model for your clinical needs (maintain high recall with lowest FPs)
    if not summary_df.empty:
        best_model = summary_df.iloc[0]  # Sorted by best high-recall performance
        print(f"\nBest Model for Clinical Use (High Recall + Low FPs): {best_model['Model']}")
        print(f"Best Threshold for High Recall: {best_model['Best_High_Recall_Threshold']}")
        print(f"FN at this threshold: {best_model['Best_High_Recall_FN']}")
        print(f"FP at this threshold: {best_model['Best_High_Recall_FP']}")
    
    # Create and save chart
    plot_efficiency_comparison(all_model_data, args.output)
    
    # Save CSV if requested
    if args.csv:
        summary_df.to_csv(args.csv, index=False)
        print(f"Summary saved to: {args.csv}")

if __name__ == "__main__":
    main()
