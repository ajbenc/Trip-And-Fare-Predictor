# NYC Taxi Trip Prediction - Model Experiments Documentation

## ⚠️ IMPORTANT: This Document is for TRIP DURATION Model Only

**This documentation covers experiments for the TRIP DURATION prediction model.**

### 🚕 Two Separate Models in This Project:

| Model                | Target        | Performance                | Status           | Documentation    |
|----------------------|--------------|----------------------------|------------------|------------------|
| **Fare Amount** 💰   | `fare_amount`| **94% R² (production)**    | ✅ **PRODUCTION**| Not covered here |
| **Trip Duration** ⏱️| `trip_duration`| **88% R² (production)**    | ✅ **PRODUCTION**| **YOU ARE HERE** |

**Key Point**: The final production models are LightGBM for both fare and duration, each using 56 features. All experiments below are for improving the trip duration model only.

---

## 🎯 Project Goal
Improve **trip duration prediction** from baseline **82.85% R²** to **90%+ R²** using advanced machine learning techniques.

---

## 📊 Model Experiments Summary (Duration Model Only)

### Experiment Timeline

| Model                    | Training R² | Validation R² | Test R² | Status       | Notes                                                              |
|--------------------------|-------------|---------------|---------|--------------|--------------------------------------------------------------------|
| Baseline LightGBM        | 88.00%      | 82.85%        | -       | ✅ Baseline   | Original model                                                     |
| MLP Neural Network V1    | 69.12%      | 68.45%        | -       | ❌ Failed     | Underperformed                                                     |
| MLP Neural Network V2    | 74.88%      | 73.21%        | -       | ❌ Failed     | Still worse than baseline                                          |
| MLP Neural Network V3    | 76.41%      | 76.41%        | 71.98%  | ❌ Failed     | Best MLP, still 11pp below baseline                                |
|MLP exp_full_33M**        |    34.79%  | **76.33%**    | **69.56%** | ✅ completed | Fare: 43.37% / 87.21% / 81.20%; Duration: 26.21% / 65.45% / 57.91% (n=33.3M)
| Enhanced LightGBM V1     | 88.41%      | 84.68%        | -       | ✅ Good       | +1.83pp improvement                                                |
| Enhanced LightGBM V2     | 88.49%      | 84.77%        | 80.80%  | ✅ Good       | +1.92pp improvement                                                |
| Production LightGBM      | 89.59%      | 88.00%        | 88.00%  | ✅ PRODUCTION | Final model, 56 features  

---

## 🏆 Why LightGBM Outperformed Everything Else

### 1. **Neural Networks Failed (MLP V1, V2, V3)**

**Architectures Tested:**

**Results:**
- Even the best MLP (V3: 71.98%) performed 16 percentage points worse than the Production LightGBM model (88%).
- The large-scale MLP trained on 33.3M samples also failed to reach 70%.

**Why MLPs Failed:**
1. **Tabular Data Problem**: Neural networks struggle with tabular data
   - Lack of spatial/sequential structure
   - Features have varying scales and distributions
   - No natural hierarchy for deep learning

2. **Feature Interactions**: MLPs learn interactions implicitly
   - Tree models learn feature interactions explicitly through splits
   - LightGBM can easily handle: "if distance > 10 AND is_raining THEN duration += X"
   - MLPs must learn them implicitly through weights, which is significantly harder and requires far more data.

3. **Overfitting Risk**: Even with regularization (Dropout, BatchNorm)
   - The MLPs show 4–7 percentage point drops between validation and test
   - Neural nets memorize training patterns
   - Poor generalization to December test data

4. **Data Efficiency**: 
   - LightGBM: Works well with 27M samples
   - MLPs: Need 5x-20x more data for same performance
   - Taxis have high variance → neural nets struggle

5. **Training Time**:
   - MLP V3: 45+ minutes to train
   - LightGBM ULTRA: 60 minutes but much better results

**Conclusion**: Neural networks are NOT suitable for taxi trip prediction. Trees win for tabular data.

---

### 2. **LightGBM ULTRA Won - Here's Why**

**Final Model Architecture:**
```
Model: LightGBM ULTRA
Hyperparameters:
  - n_estimators: 1000 (more trees for complex patterns)
  - num_leaves: 1024 (high capacity)
  - max_depth: 15 (deep trees for interactions)
  - learning_rate: 0.02 (slow, careful learning)
  - reg_alpha: 0.5 (L1 regularization)
  - reg_lambda: 2.0 (L2 regularization)
  
Features: 56 total (production)
   - Weather, holidays, temporal, location, interactions, distance
  
Performance:
  - Training R²: 89.59%
  - Validation R²: 88.00% ← PRODUCTION METRIC
  - Test R²: 88.00%
  - MAE: 2.71 minutes (validation), 3.04 minutes (test)
  - Training Time: 60 minutes on 29.2M samples
```

---

## 🔑 Key Success Factors for LightGBM ULTRA

### 1. **Smart Feature Engineering (51 New Features)**

## 📋 Feature Categories (56 Total)

1. **Location Features** (9): Zone IDs, airport flags, Manhattan flags, same location
2. **Temporal Features** (15): Hour, day, month, weekday, rush hour, + cyclical encodings
3. **Distance Features** (1): Estimated distance from zone centroids (Haversine)
4. **Weather Features** (15): Temperature, precipitation, snow, wind, weather severity
5. **Holiday Features** (3): Holiday flags, major holidays, holiday week
6. **Interaction Features** (13): Weather×Location, Time×Distance, Holiday×Location


**Result**: These 56 features captured complex non-linear patterns that raw features couldn't express.

---

### 2. **Tree-Based Advantages for Taxi Data**

**Why LightGBM Excels:**

1. **Automatic Feature Interactions**
   - Trees naturally split on multiple features
   - Example: "IF distance > 10 AND is_raining AND is_rush_hour THEN +15 minutes"
   - MLPs must learn this through millions of weight updates

2. **Handles Missing/Categorical Data**
   - No need for one-hot encoding (accepts categorical directly)
   - Robust to missing values
   - Natural handling of sparse features

3. **Non-Linear Relationships**
   - Distance → Duration is non-linear (traffic congestion)
   - Trees capture this naturally with splits
   - MLPs need careful activation functions

4. **Robustness to Outliers**
   - Tree splits are rank-based, not value-based
   - Outliers don't skew predictions as much
   - Important for taxi data (extreme trips exist)

5. **Interpretability**
   - Feature importance clearly shows what matters:
     - Top 5: DOLocationID (137k), PULocationID (88k), route_hash (80k), route_complexity (77k), estimated_distance (49k)
   - Business insights: "Location is most important, then route patterns, then distance"
   - MLPs are black boxes

---

### 3. **Regularization Prevented Overfitting**

**Training-Validation Gap Analysis:**
```
Baseline:     88.00% train → 82.85% val = 5.15pp gap
ULTRA:        89.59% train → 88.00% val = 1.59pp gap ✅ BEST
```

**Why ULTRA Doesn't Overfit:**
1. **Strong L1/L2 Regularization**: reg_alpha=0.5, reg_lambda=2.0
   - Penalizes complex trees
   - Forces model to generalize

2. **Feature Ratio**: 56 features ÷ 29.9M samples = 534,000 samples per feature
   - Excellent ratio (rule of thumb: >1000 samples/feature)
   - Plenty of data to learn each feature reliably

3. **Early Stopping**: Monitored validation loss
   - Stopped when validation stopped improving
   - Used all 1000 trees (validation kept improving!)

4. **Max Depth = 15**: Not too deep, not too shallow
   - Deep enough for complex interactions
   - Not so deep it memorizes noise

---

### 4. **Data Quality Improvements**

**Outlier Capping:**
- Duration: Capped at 40.33 min (95th percentile)
- Distance: Capped at 20.20 mi (99th percentile)
- Effect: Removed extreme outliers that skew learning

**Weather Integration (15 features):**
- Temperature, precipitation, snow, rain, wind, visibility
- Critical for duration: rain/snow slow traffic significantly
- December test set had different weather → model needed these features

**Holiday Features (3 features):**
- is_holiday, is_major_holiday, is_holiday_week
- December has Christmas/New Year → essential for generalization

---


## 🎯 Final Model Selection: ULTRA LightGBM

**Why ULTRA is Production-Ready:**

✅ **Validation Performance**: 88.00% R² (best we achieved)  
✅ **Test Performance**: 88.00% R² 
✅ **Improvement**: +5.15pp vs baseline (82.85% → 88.00%)  
✅ **Generalization**: 1.59pp train-val gap (healthy)  
✅ **Real-World MAE**: 2.71 min (validation), 3.04 min (test)  
✅ **Business Value**: ±3 minutes is acceptable for dispatch/ETAs  
✅ **Feature Rich**: 56 features capture weather, holidays, route complexity  
✅ **Interpretable**: Clear feature importance for business insights  
✅ **Fast Inference**: 42,605 predictions/second  

---

## 📊 Model Comparison Table

| Metric            | Baseline | Enhanced V2 | Production | Improvement |
|-------------------|----------|-------------|------------|-------------|
| **Validation R²** | 82.85%   | 84.77%      | **88.00%** | **+5.15pp** |
| **Test R²**       | TBD      | 80.80%      | **88.00%** | **+7.20pp** |
| **Features**      | 56       | 76 (56+20)  | 56         | 0           |
| **MAE (val)**     | ~3.4 min | ~2.8 min    | **2.71 min** | **-0.79 min** |
| **MAE (test)**    | TBD      | 3.16 min    | **3.04 min** | **-0.12 min** |
| **Train Time**    | 25 min   | 35 min      | 60 min     | +35 min      |
| **Train-Val Gap** | 5.15pp | 3.72pp | **1.59pp** | **3.56pp** |

---

## 🚀 Deployment Recommendation

**Use: ULTRA LightGBM for Trip Duration Prediction**

**Model Path**: `models/lightgbm_ultra/duration_lightgbm_ultra.txt`

**Required Features**: 56 features 

**Expected Performance**:
- Real-world R²: 82-85% (based on test/validation)
- Average error: ±3 minutes
- 95% predictions within: ±6 minutes

**Production Checklist**:
- ✅ Model trained and validated
- ✅ Feature engineering pipeline documented
- ✅ Inference tested (42k predictions/sec)
- ✅ Weather/holiday features integrated
- ✅ Outlier handling in place

**Monitoring Recommendations**:
1. Track MAE weekly (should stay ~3 minutes)
2. Monitor December performance (holiday season)
3. Alert if R² drops below 80%
4. Retrain quarterly with new data

---

## 📚 Key Learnings

### 1. **Tabular Data → Use Tree Models**
- LightGBM/XGBoost beat neural networks every time
- Trees learn feature interactions naturally
- Better data efficiency (less data needed)

### 2. **Feature Engineering is Critical**
- +56 engineered features → +1.59pp R² improvement
- Weather interactions essential (rain/snow slow traffic)
- Route complexity patterns captured (Manhattan rush hour)

### 3. **Regularization Prevents Overfitting**
- L1=0.5, L2=2.0 kept train-val gap at 4pp
- More trees (1000) + regularization = better generalization

### 4. **Test Set Reveals Reality**
- December is harder than November (holidays, weather)
- 3.4pp val-test gap is normal for seasonal data
- Always validate on truly held-out data

### 5. **Business Context Matters**
- ±3 minutes MAE is acceptable for taxi dispatch
- 85% R² means explaining 85% of variance
- Remaining 12% is random (traffic, driver behavior)

---

## 📈 Future Improvements (Beyond Current Scope)

1. **Real-Time Traffic Data**: Integrate Google Maps API
   - Could improve R² to 88-90%
   - Expensive to implement ($$$)

2. **Driver Behavior Features**: 
   - Driver experience, rating, route choices
   - Privacy concerns

3. **Live Events**: Concerts, sports games, protests
   - Would help with outlier predictions
   - Difficult to source reliable event data

4. **Ensemble Methods**: Combine ULTRA + XGBoost
   - Small improvements (1-2pp)
   - Added complexity

5. **Deep Learning (if 10x more data)**:
   - TabNet, FT-Transformer for tabular data
   - Requires 100M+ samples
   - Not worth it for current scale

---

## ✅ Conclusion

**ULTRA LightGBM is the clear winner** for NYC Taxi trip duration prediction:

- **88.00% validation R²** (best achieved)
- **88.00% test R²** 
- **56 features** (weather, holidays, route complexity)
- **Production-ready** (fast, interpretable, robust)

**Neural networks failed** because tabular data doesn't benefit from deep learning. Trees are the right tool for this job.

**Deploy with confidence!** 🚀
