"""
Temporal Splitter for Safe Features
-----------------------------------
Creates a temporal (time-based) split for training/validation/testing.
Train: 2022-01 -> 2022-10 (inclusive)
Validation: 2022-11
Test: 2022-12

This script copies the per-month feature/target parquet files from
`data/processed/features/` into `data/splits/<train|val|test>/`.

It performs file-level copies (fast). It does NOT modify originals.

Outputs:
- data/splits/train/features_2022-01_X.parquet
- data/splits/train/features_2022-01_y_fare.parquet
- data/splits/train/features_2022-01_y_duration.parquet
- data/splits/val/...
- data/splits/test/...

"""

from pathlib import Path
import shutil
import re

SRC_DIR = Path('data/processed/features')
DEST_DIR = Path('data/splits')

DEST_DIR.mkdir(parents=True, exist_ok=True)

pattern = re.compile(r'features_(\d{4}-\d{2})_X.parquet')

splits = {
    'train': set([f'2022-{m:02d}' for m in range(1, 11)]),  # Jan-Oct
    'val': set(['2022-11']),
    'test': set(['2022-12']),
}

# create directories
for s in splits:
    (DEST_DIR / s).mkdir(parents=True, exist_ok=True)

files = sorted(SRC_DIR.glob('features_2022-*_X.parquet'))
if not files:
    print('❌ No feature files found in', SRC_DIR)
    raise SystemExit(1)

copied = { 'train': [], 'val': [], 'test': [] }

for f in files:
    m = pattern.search(f.name)
    if not m:
        print('Skipping unknown file:', f.name)
        continue
    month_tag = m.group(1)  # like '2022-01'

    # Determine split
    split_name = None
    for s, months in splits.items():
        if month_tag in months:
            split_name = s
            break
    if not split_name:
        print(f'No split mapping for {month_tag}, skipping')
        continue

    # Source files
    src_X = f
    src_y_fare = SRC_DIR / f'features_{month_tag}_y_fare.parquet'
    src_y_duration = SRC_DIR / f'features_{month_tag}_y_duration.parquet'

    # Destination
    dst_dir = DEST_DIR / split_name
    dst_X = dst_dir / src_X.name
    dst_y_fare = dst_dir / src_y_fare.name
    dst_y_duration = dst_dir / src_y_duration.name

    # Copy files if they exist
    shutil.copy2(src_X, dst_X)
    if src_y_fare.exists():
        shutil.copy2(src_y_fare, dst_y_fare)
    else:
        print(f'Warning: missing {src_y_fare.name}')
    if src_y_duration.exists():
        shutil.copy2(src_y_duration, dst_y_duration)
    else:
        print(f'Warning: missing {src_y_duration.name}')

    copied[split_name].append(month_tag)
    print(f'Copied {month_tag} -> {split_name}')

# Summary
print('\nCopy summary:')
for s in ['train','val','test']:
    print(f'  {s}: {len(copied[s])} months -> {copied[s]}')

print('\nDone. Files are in data/splits/<train|val|test>/')
