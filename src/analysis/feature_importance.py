"""
Feature Importance Analysis for XGBoost and LightGBM Models

This script analyzes and visualizes feature importance from trained models
to understand what drives predictions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

def load_model(model_path):
    """Load a trained model from pickle file"""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def get_feature_importance(model, feature_names, model_name, target_name):
    """Extract feature importance from model"""
    
    # Get importance values
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        print(f"⚠️ Model {model_name} doesn't have feature_importances_ attribute")
        return None
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Add percentage
    importance_df['importance_pct'] = (importance_df['importance'] / importance_df['importance'].sum()) * 100
    
    # Add cumulative percentage
    importance_df['cumulative_pct'] = importance_df['importance_pct'].cumsum()
    
    print(f"\n{'='*70}")
    print(f"{model_name.upper()} - {target_name.upper()} FEATURE IMPORTANCE")
    print(f"{'='*70}")
    
    print(f"\n📊 Top 20 Most Important Features:")
    print(f"\n{'Rank':<6} {'Feature':<35} {'Importance':<12} {'%':<8} {'Cumulative %'}")
    print("-" * 70)
    
    for idx, row in importance_df.head(20).iterrows():
        print(f"{importance_df.index.get_loc(idx)+1:<6} {row['feature']:<35} "
              f"{row['importance']:<12.4f} {row['importance_pct']:<8.2f} {row['cumulative_pct']:.2f}%")
    
    # Key insights
    print(f"\n🔍 Key Insights:")
    top_5_features = importance_df.head(5)['feature'].tolist()
    top_5_pct = importance_df.head(5)['importance_pct'].sum()
    print(f"   • Top 5 features contribute: {top_5_pct:.1f}% of total importance")
    print(f"   • Top 5 features: {', '.join(top_5_features[:3])}...")
    
    top_10_pct = importance_df.head(10)['importance_pct'].sum()
    print(f"   • Top 10 features contribute: {top_10_pct:.1f}% of total importance")
    
    # Features above 1% importance
    significant = importance_df[importance_df['importance_pct'] >= 1.0]
    print(f"   • Features with ≥1% importance: {len(significant)}")
    
    # Low importance features
    low_importance = importance_df[importance_df['importance_pct'] < 0.5]
    print(f"   • Features with <0.5% importance: {len(low_importance)}")
    
    return importance_df

def plot_feature_importance(importance_df, model_name, target_name, top_n=20):
    """Create comprehensive feature importance visualizations"""
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Horizontal bar chart - Top N features
    ax1 = fig.add_subplot(gs[0, :])
    top_features = importance_df.head(top_n)
    
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_features)))
    bars = ax1.barh(range(len(top_features)), top_features['importance'], 
                     color=colors, edgecolor='black', linewidth=1.2)
    
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features['feature'], fontsize=10)
    ax1.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
    ax1.set_title(f'Top {top_n} Feature Importance - {model_name} ({target_name})', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top_features['importance'])):
        width = bar.get_width()
        pct = top_features.iloc[i]['importance_pct']
        ax1.text(width + max(top_features['importance'])*0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.4f} ({pct:.1f}%)', va='center', fontsize=9, fontweight='bold')
    
    # 2. Cumulative importance
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(range(1, len(importance_df)+1), importance_df['cumulative_pct'], 
             linewidth=3, color='#2E86AB', marker='o', markersize=4, markevery=5)
    ax2.axhline(y=80, color='red', linestyle='--', linewidth=2, alpha=0.7, label='80% threshold')
    ax2.axhline(y=90, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='90% threshold')
    
    ax2.set_xlabel('Number of Features', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Cumulative Importance (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Cumulative Feature Importance', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='lower right')
    
    # Find how many features for 80% and 90%
    features_80 = (importance_df['cumulative_pct'] <= 80).sum()
    features_90 = (importance_df['cumulative_pct'] <= 90).sum()
    
    ax2.annotate(f'{features_80} features\n(80%)', 
                xy=(features_80, 80), xytext=(features_80+5, 75),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, fontweight='bold', color='red')
    
    ax2.annotate(f'{features_90} features\n(90%)', 
                xy=(features_90, 90), xytext=(features_90+5, 85),
                arrowprops=dict(arrowstyle='->', color='orange', lw=2),
                fontsize=10, fontweight='bold', color='orange')
    
    # 3. Importance distribution
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Create bins
    bins = [0, 0.001, 0.005, 0.01, 0.05, 1.0]
    labels = ['<0.1%', '0.1-0.5%', '0.5-1%', '1-5%', '>5%']
    importance_df['importance_bin'] = pd.cut(importance_df['importance_pct'], 
                                             bins=[0, 0.1, 0.5, 1, 5, 100], 
                                             labels=labels)
    
    bin_counts = importance_df['importance_bin'].value_counts().reindex(labels)
    
    colors_pie = ['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71', '#27ae60']
    wedges, texts, autotexts = ax3.pie(bin_counts, labels=labels, autopct='%1.1f%%',
                                        colors=colors_pie, startangle=90,
                                        textprops={'fontsize': 10, 'fontweight': 'bold'})
    
    ax3.set_title('Feature Importance Distribution', fontsize=12, fontweight='bold')
    
    # Add legend with counts
    legend_labels = [f'{label}: {count} features' for label, count in zip(labels, bin_counts)]
    ax3.legend(legend_labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)
    
    plt.suptitle(f'Feature Importance Analysis - {model_name.upper()} ({target_name.title()})',
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save plot
    output_dir = Path('docs/presentation_plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f'feature_importance_{model_name.lower()}_{target_name.lower()}.png'
    output_path = output_dir / filename
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Plot saved to: {output_path}")
    
    plt.show()

def compare_model_importance(xgb_importance, lgb_importance, target_name):
    """Compare feature importance between XGBoost and LightGBM"""
    
    print(f"\n{'='*70}")
    print(f"FEATURE IMPORTANCE COMPARISON: XGBOOST vs LIGHTGBM ({target_name.upper()})")
    print(f"{'='*70}")
    
    # Merge dataframes
    comparison = xgb_importance[['feature', 'importance_pct']].rename(
        columns={'importance_pct': 'xgb_pct'})
    comparison = comparison.merge(
        lgb_importance[['feature', 'importance_pct']].rename(
            columns={'importance_pct': 'lgb_pct'}),
        on='feature', how='outer').fillna(0)
    
    comparison['avg_pct'] = (comparison['xgb_pct'] + comparison['lgb_pct']) / 2
    comparison['diff_pct'] = abs(comparison['xgb_pct'] - comparison['lgb_pct'])
    comparison = comparison.sort_values('avg_pct', ascending=False)
    
    print(f"\n📊 Top 10 Features by Average Importance:")
    print(f"\n{'Feature':<35} {'XGBoost %':<12} {'LightGBM %':<12} {'Avg %':<10} {'Diff %'}")
    print("-" * 85)
    
    for _, row in comparison.head(10).iterrows():
        print(f"{row['feature']:<35} {row['xgb_pct']:<12.2f} {row['lgb_pct']:<12.2f} "
              f"{row['avg_pct']:<10.2f} {row['diff_pct']:.2f}")
    
    # Agreement analysis
    xgb_top10 = set(xgb_importance.head(10)['feature'])
    lgb_top10 = set(lgb_importance.head(10)['feature'])
    agreement = len(xgb_top10.intersection(lgb_top10))
    
    print(f"\n🤝 Model Agreement:")
    print(f"   • Features in both top 10: {agreement}/10")
    print(f"   • Agreement rate: {agreement*10:.0f}%")
    print(f"   • Common features: {', '.join(list(xgb_top10.intersection(lgb_top10))[:5])}...")
    
    return comparison

def analyze_feature_categories(importance_df, model_name, target_name):
    """Analyze feature importance by category"""
    
    print(f"\n{'='*70}")
    print(f"FEATURE IMPORTANCE BY CATEGORY - {model_name.upper()} ({target_name.upper()})")
    print(f"{'='*70}")
    
    # Categorize features
    def categorize_feature(feature):
        if any(x in feature.lower() for x in ['hour', 'day', 'month', 'weekend', 'rush', 'night', 'holiday']):
            return 'Temporal'
        elif any(x in feature.lower() for x in ['distance', 'duration', 'route', 'typical', 'actual', 'ratio', 'efficiency']):
            return 'Trip Metrics'
        elif any(x in feature.lower() for x in ['airport', 'manhattan', 'borough', 'location', 'popular', 'cross']):
            return 'Geographic'
        elif any(x in feature.lower() for x in ['interaction', '_x_', 'combined']):
            return 'Interactions'
        elif any(x in feature.lower() for x in ['sin', 'cos']):
            return 'Cyclical'
        else:
            return 'Other'
    
    importance_df['category'] = importance_df['feature'].apply(categorize_feature)
    
    category_summary = importance_df.groupby('category').agg({
        'importance_pct': ['sum', 'mean', 'count']
    }).round(2)
    
    category_summary.columns = ['Total %', 'Avg %', 'Count']
    category_summary = category_summary.sort_values('Total %', ascending=False)
    
    print(f"\n{'Category':<20} {'Total %':<12} {'Avg %':<12} {'# Features'}")
    print("-" * 60)
    for cat, row in category_summary.iterrows():
        print(f"{cat:<20} {row['Total %']:<12.2f} {row['Avg %']:<12.2f} {int(row['Count'])}")
    
    print(f"\n🎯 Key Findings:")
    top_category = category_summary.index[0]
    top_pct = category_summary.iloc[0]['Total %']
    print(f"   • Most important category: {top_category} ({top_pct:.1f}% total importance)")
    
    for cat in category_summary.index[:3]:
        top_features = importance_df[importance_df['category'] == cat].head(3)['feature'].tolist()
        print(f"   • Top {cat} features: {', '.join(top_features)}")
    
    return category_summary

def main():
    """Main execution function"""
    
    print("\n" + "="*70)
    print("🎯 FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    # Define paths
    base_path = Path('src/models')
    
    models = {
        'XGBoost': {
            'fare': base_path / 'advanced/xgboost_fare_amount.pkl',
            'duration': base_path / 'advanced/xgboost_trip_duration.pkl'
        },
        'LightGBM': {
            'fare': base_path / 'advanced/lightgbm_fare_amount.pkl',
            'duration': base_path / 'advanced/lightgbm_trip_duration.pkl'
        }
    }
    
    # Load feature names from preprocessed data
    print("\n📥 Loading feature names from data...")
    data_path = Path('data/processed/train.parquet')
    
    if data_path.exists():
        df_sample = pd.read_parquet(data_path, engine='pyarrow')
        # Remove target columns
        feature_names = [col for col in df_sample.columns 
                        if col not in ['fare_amount', 'trip_duration']]
        print(f"✅ Loaded {len(feature_names)} features")
    else:
        print("⚠️ Processed data not found. Using default feature names...")
        # Fallback feature names
        feature_names = None
    
    # Analyze each model
    results = {}
    
    for model_name, targets in models.items():
        results[model_name] = {}
        
        for target_name, model_path in targets.items():
            print(f"\n{'='*70}")
            print(f"Analyzing {model_name} - {target_name.title()} Prediction")
            print(f"{'='*70}")
            
            if not model_path.exists():
                print(f"❌ Model not found: {model_path}")
                continue
            
            print(f"📂 Loading model from: {model_path}")
            model = load_model(model_path)
            
            # Get feature names from model if not available
            if feature_names is None:
                if hasattr(model, 'feature_names_in_'):
                    feature_names = model.feature_names_in_
                elif hasattr(model, 'feature_name_'):
                    feature_names = model.feature_name_
                else:
                    print("❌ Cannot determine feature names")
                    continue
            
            # Get importance
            importance_df = get_feature_importance(model, feature_names, model_name, target_name)
            
            if importance_df is not None:
                results[model_name][target_name] = importance_df
                
                # Visualize
                plot_feature_importance(importance_df, model_name, target_name, top_n=20)
                
                # Analyze by category
                analyze_feature_categories(importance_df, model_name, target_name)
                
                # Save to CSV
                output_dir = Path('src/models/advanced')
                csv_path = output_dir / f'{model_name.lower()}_importance_{target_name}.csv'
                importance_df.to_csv(csv_path, index=False)
                print(f"\n✅ Importance data saved to: {csv_path}")
    
    # Compare models
    if 'XGBoost' in results and 'LightGBM' in results:
        for target in ['fare', 'duration']:
            if target in results['XGBoost'] and target in results['LightGBM']:
                comparison = compare_model_importance(
                    results['XGBoost'][target],
                    results['LightGBM'][target],
                    target
                )
                
                # Save comparison
                output_dir = Path('src/models/advanced')
                comp_path = output_dir / f'model_comparison_importance_{target}.csv'
                comparison.to_csv(comp_path, index=False)
                print(f"\n✅ Comparison saved to: {comp_path}")
    
    print("\n" + "="*70)
    print("✅ FEATURE IMPORTANCE ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\n📊 Outputs generated:")
    print(f"   • Feature importance plots (PNG)")
    print(f"   • Importance data (CSV)")
    print(f"   • Model comparison (CSV)")
    print(f"   • Category analysis (displayed)")
    print(f"\n📂 Location: docs/presentation_plots/ and src/models/advanced/")

if __name__ == "__main__":
    main()
