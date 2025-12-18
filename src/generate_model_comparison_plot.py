# Generate Model Comparison Plot for Presentation

import matplotlib.pyplot as plt
import numpy as np

# Model performance data
models = ['Linear\nRegression', 'Decision\nTree', 'LightGBM', 'XGBoost']
fare_r2 = [0.8215, 0.9110, 0.9411, 0.9431]
duration_r2 = [0.6409, 0.7949, 0.8543, 0.8609]

# Professional colors (progression from basic to advanced)
colors = ['#95a5a6', '#3498db', '#f39c12', '#27ae60']

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# ============== Fare Amount R² ==============
bars1 = ax1.bar(models, fare_r2, color=colors, edgecolor='black', linewidth=2, alpha=0.9)
ax1.set_ylabel('R² Score', fontsize=14, fontweight='bold')
ax1.set_title('Fare Amount Prediction Accuracy', fontsize=16, fontweight='bold', pad=15)
ax1.set_ylim(0, 1)
ax1.axhline(y=0.94, color='red', linestyle='--', linewidth=2, alpha=0.4, 
            label='Production Target: 94%')
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.legend(loc='upper left', fontsize=11)

# Add value labels on bars
for bar, score in zip(bars1, fare_r2):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.015,
             f'{score:.2%}', ha='center', va='bottom', 
             fontsize=13, fontweight='bold')
    # Add improvement arrow for XGBoost
    if score == fare_r2[-1]:
        ax1.annotate('BEST!', 
                    xy=(bar.get_x() + bar.get_width()/2., height + 0.05),
                    fontsize=12, ha='center', fontweight='bold', color='green')

# Add baseline reference
ax1.axhline(y=fare_r2[0], color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
ax1.text(0.5, fare_r2[0] - 0.05, 'Baseline', fontsize=10, style='italic', color='gray')

# ============== Trip Duration R² ==============
bars2 = ax2.bar(models, duration_r2, color=colors, edgecolor='black', linewidth=2, alpha=0.9)
ax2.set_ylabel('R² Score', fontsize=14, fontweight='bold')
ax2.set_title('Trip Duration Prediction Accuracy', fontsize=16, fontweight='bold', pad=15)
ax2.set_ylim(0, 1)
ax2.axhline(y=0.86, color='red', linestyle='--', linewidth=2, alpha=0.4, 
            label='Production Target: 86%')
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.legend(loc='upper left', fontsize=11)

# Add value labels on bars
for bar, score in zip(bars2, duration_r2):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.015,
             f'{score:.2%}', ha='center', va='bottom', 
             fontsize=13, fontweight='bold')
    # Add improvement arrow for XGBoost
    if score == duration_r2[-1]:
        ax2.annotate('BEST!', 
                    xy=(bar.get_x() + bar.get_width()/2., height + 0.05),
                    fontsize=12, ha='center', fontweight='bold', color='green')

# Add baseline reference
ax2.axhline(y=duration_r2[0], color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
ax2.text(0.5, duration_r2[0] - 0.05, 'Baseline', fontsize=10, style='italic', color='gray')

# Overall title
fig.suptitle('Model Performance Comparison - XGBoost Achieves Best Results!', 
             fontsize=18, fontweight='bold', y=1.00)

# Add improvement stats box
improvement_fare = ((fare_r2[-1] - fare_r2[0]) / fare_r2[0]) * 100
improvement_dur = ((duration_r2[-1] - duration_r2[0]) / duration_r2[0]) * 100

textstr = f'Improvement vs Baseline:\n  Fare: +{improvement_fare:.1f}%\n  Duration: +{improvement_dur:.1f}%'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
fig.text(0.5, 0.02, textstr, fontsize=11, ha='center',
         bbox=props, family='monospace')

plt.tight_layout(rect=[0, 0.06, 1, 0.96])

# Save with high quality
plt.savefig('docs/presentation_plots/model_comparison.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Model comparison plot saved to: docs/presentation_plots/model_comparison.png")
print(f"\n📊 Performance Summary:")
print(f"   Fare Amount - XGBoost: {fare_r2[-1]:.2%} (Best: {max(fare_r2):.2%})")
print(f"   Trip Duration - XGBoost: {duration_r2[-1]:.2%} (Best: {max(duration_r2):.2%})")
print(f"\n🚀 Improvement over baseline:")
print(f"   Fare: +{improvement_fare:.1f}% (+{(fare_r2[-1] - fare_r2[0]):.4f} R²)")
print(f"   Duration: +{improvement_dur:.1f}% (+{(duration_r2[-1] - duration_r2[0]):.4f} R²)")

plt.show()
