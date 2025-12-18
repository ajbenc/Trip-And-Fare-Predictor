"""
Project Statistics Generator
============================

Generate a quick statistics summary of the project.
"""

import os
from pathlib import Path


def count_lines_in_file(filepath):
    """Count lines in a Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return len(f.readlines())
    except:
        return 0


def analyze_project():
    """Analyze project statistics."""
    
    print("="*70)
    print("NYC TAXI TRIP PREDICTION - PROJECT STATISTICS")
    print("="*70)
    
    # Project root
    root = Path(__file__).parent.parent
    
    # Count Python files and lines
    src_dir = root / 'src'
    python_files = list(src_dir.glob('*.py'))
    python_files = [f for f in python_files if '__pycache__' not in str(f)]
    
    total_lines = sum(count_lines_in_file(f) for f in python_files)
    
    print(f"\n📁 CODE STATISTICS:")
    print(f"   Python Modules: {len(python_files)}")
    print(f"   Total Lines of Code: {total_lines:,}")
    print(f"\n   Files:")
    for f in sorted(python_files):
        lines = count_lines_in_file(f)
        print(f"      • {f.name}: {lines} lines")
    
    # Count models
    models_dir = root / 'models' / 'baseline'
    if models_dir.exists():
        model_files = list(models_dir.glob('*.pkl'))
        csv_files = list(models_dir.glob('*.csv'))
        
        print(f"\n💾 SAVED MODELS:")
        print(f"   Model Files (.pkl): {len(model_files)}")
        print(f"   Metric Files (.csv): {len(csv_files)}")
        
        for f in sorted(model_files):
            size = f.stat().st_size / 1024 / 1024
            print(f"      • {f.name}: {size:.2f} MB")
    
    # Count data files
    data_dir = root / 'data'
    raw_dir = data_dir / 'raw'
    processed_dir = data_dir / 'processed'
    
    print(f"\n📊 DATA FILES:")
    if raw_dir.exists():
        raw_files = list(raw_dir.glob('*.parquet'))
        for f in raw_files:
            size = f.stat().st_size / 1024 / 1024
            print(f"   Raw: {f.name} ({size:.2f} MB)")
    
    if processed_dir.exists():
        processed_files = list(processed_dir.glob('*.parquet'))
        for f in sorted(processed_files):
            size = f.stat().st_size / 1024 / 1024
            print(f"   Processed: {f.name} ({size:.2f} MB)")
    
    # Count visualizations
    plots_dir = root / 'models' / 'plots'
    if plots_dir.exists():
        plot_files = list(plots_dir.glob('*.png'))
        print(f"\n📈 VISUALIZATIONS:")
        print(f"   Plot Files: {len(plot_files)}")
        for f in sorted(plot_files):
            print(f"      • {f.name}")
    
    # Notebooks
    notebooks_dir = root / 'notebooks'
    if notebooks_dir.exists():
        notebooks = list(notebooks_dir.glob('*.ipynb'))
        print(f"\n📓 NOTEBOOKS:")
        print(f"   Jupyter Notebooks: {len(notebooks)}")
        for f in sorted(notebooks):
            print(f"      • {f.name}")
    
    # Documentation
    docs_count = len(list(root.glob('*.md')))
    print(f"\n📝 DOCUMENTATION:")
    print(f"   Markdown Files: {docs_count}")
    for f in sorted(root.glob('*.md')):
        print(f"      • {f.name}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✅ Project Progress: 4/12 core tasks (33%)")
    print(f"✅ Code Modules: {len(python_files)}")
    print(f"✅ Lines of Code: {total_lines:,}")
    print(f"✅ Models Trained: 4")
    print(f"✅ Visualizations: {len(plot_files) if plots_dir.exists() else 0}")
    print(f"✅ Status: Baseline Complete, Ready for Advanced Development")
    print("="*70)


if __name__ == "__main__":
    analyze_project()
