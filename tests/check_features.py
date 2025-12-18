import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.modules.preprocessing_utils import load_processed_data, prepare_features_and_target

train_df = load_processed_data(dataset='train')

# Use fare_amount (lowercase)
X_train, y_train, feature_names = prepare_features_and_target(train_df, 'fare_amount')

print(f"\nNumber of features: {len(feature_names)}")
print("\nFeature names:")
for i, col in enumerate(feature_names, 1):
    print(f"{i}. {col}")
