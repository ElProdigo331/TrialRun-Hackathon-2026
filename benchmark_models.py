"""
Comprehensive Model Benchmarking Script
Tests all model/option combinations to find optimal settings for oil production prediction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from scipy.ndimage import uniform_filter
import warnings
import json
from datetime import datetime
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not available, skipping XGBoost tests")

print("=" * 70)
print("COMPREHENSIVE MODEL BENCHMARKING")
print("Testing all model/option combinations for optimal settings")
print("=" * 70)

train_df = pd.read_csv('outputs/MICE_imputed_production_wells.csv')
test_df = pd.read_csv('outputs/MICE_imputed_preproduction_wells.csv')
prod_history = pd.read_csv('data/Production_history_production_wells.csv')
sand_map = np.load('data/2d_sand_proportion.npy')

print(f"\nData loaded: {len(train_df)} training rows, {len(test_df)} test rows")

target_col = [c for c in prod_history.columns if 'cumulative' in c.lower() and 'oil' in c.lower()]
if not target_col:
    target_col = [c for c in prod_history.columns if 'oil' in c.lower()]
target_col = target_col[0] if target_col else prod_history.columns[-1]

targets = prod_history.groupby('Well_ID')[target_col].max().reset_index()
targets.columns = ['Well_ID', 'Target_BBL']
print(f"Target: {target_col}")
print(f"Target range: {targets['Target_BBL'].min():,.0f} - {targets['Target_BBL'].max():,.0f} BBL")
print(f"Target mean: {targets['Target_BBL'].mean():,.0f} BBL")

def aggregate_wells(df):
    """Aggregate depth-level data to well-level"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ['Well_ID', 'Z', 'facies']
    agg_cols = [c for c in numeric_cols if c not in exclude]
    
    agg_dict = {}
    for col in agg_cols:
        agg_dict[f'{col}_mean'] = (col, 'mean')
        agg_dict[f'{col}_std'] = (col, 'std')
        agg_dict[f'{col}_min'] = (col, 'min')
        agg_dict[f'{col}_max'] = (col, 'max')
    
    agg_dict['depth_range'] = ('Z', lambda x: x.max() - x.min())
    
    if 'facies' in df.columns:
        for f in range(1, 7):
            agg_dict[f'facies_{f}_pct'] = ('facies', lambda x, f=f: (x == f).mean())
    
    result = df.groupby('Well_ID').agg(**agg_dict).reset_index()
    result = result.fillna(0)
    return result

train_agg = aggregate_wells(train_df)
test_agg = aggregate_wells(test_df)

train_agg = train_agg.merge(targets, on='Well_ID', how='inner')
print(f"\nAggregated: {len(train_agg)} training wells, {len(test_agg)} test wells")

def add_sand_proportion(df, sand_map, smoothing=None):
    """Add sand proportion from 2D map with optional smoothing"""
    if smoothing == 'smooth_3x3':
        sand_data = uniform_filter(sand_map, size=3)
    elif smoothing == 'smooth_5x5':
        sand_data = uniform_filter(sand_map, size=5)
    else:
        sand_data = sand_map
    
    sand_props = []
    for _, row in df.iterrows():
        x_idx = min(int(row.get('X_mean', row.get('X', 100))), 199)
        y_idx = min(int(row.get('Y_mean', row.get('Y', 100))), 199)
        sand_props.append(sand_data[y_idx, x_idx])
    
    df = df.copy()
    df['sand_proportion'] = sand_props
    return df

def add_engineered_features(df):
    """Add derived features based on domain knowledge"""
    df = df.copy()
    
    if 'phi_mean' in df.columns and 'perm_mean' in df.columns:
        df['phi_perm_product'] = df['phi_mean'] * np.log1p(df['perm_mean'])
        
        df['RQI'] = 0.0314 * np.sqrt(df['perm_mean'] / (df['phi_mean'] + 0.001))
        phi_ratio = df['phi_mean'] / (1 - df['phi_mean'] + 0.001)
        df['FZI'] = df['RQI'] / (phi_ratio + 0.001)
    
    if 'phi_mean' in df.columns and 'GR_mean' in df.columns:
        df['rock_quality'] = df['phi_mean'] / (df['GR_mean'] + 0.001)
    
    if 'AI_mean' in df.columns and 'SI_mean' in df.columns:
        df['impedance_ratio'] = df['AI_mean'] / (df['SI_mean'] + 0.001)
    
    if 'Vp_mean' in df.columns and 'Vs_mean' in df.columns:
        df['Vp_Vs_ratio'] = df['Vp_mean'] / (df['Vs_mean'] + 0.001)
    
    facies_shale = 0
    for col in ['facies_5_pct', 'facies_6_pct']:
        if col in df.columns:
            facies_shale += df[col]
    df['net_to_gross'] = 1 - facies_shale
    
    return df

SAND_OPTIONS = ['include', 'exclude', 'smooth_3x3', 'smooth_5x5']
NORMALIZE_OPTIONS = [True, False]
MODEL_TYPES = ['Linear', 'Ridge', 'ElasticNet', 'RandomForest']
if HAS_XGBOOST:
    MODEL_TYPES.append('XGBoost')

UNCERTAINTY_METHODS = ['residual_bootstrap', 'bagging_ensemble']

def get_model(model_type, params=None):
    """Get model instance by type"""
    if model_type == 'Linear':
        return LinearRegression()
    elif model_type == 'Ridge':
        return Ridge(alpha=params.get('alpha', 1.0) if params else 1.0)
    elif model_type == 'ElasticNet':
        return ElasticNet(alpha=params.get('alpha', 1.0) if params else 1.0,
                          l1_ratio=params.get('l1_ratio', 0.5) if params else 0.5)
    elif model_type == 'RandomForest':
        return RandomForestRegressor(
            n_estimators=params.get('n_estimators', 100) if params else 100,
            max_depth=params.get('max_depth', 10) if params else 10,
            min_samples_split=params.get('min_samples_split', 5) if params else 5,
            random_state=42, n_jobs=-1
        )
    elif model_type == 'XGBoost' and HAS_XGBOOST:
        return XGBRegressor(
            n_estimators=params.get('n_estimators', 100) if params else 100,
            max_depth=params.get('max_depth', 6) if params else 6,
            learning_rate=params.get('learning_rate', 0.1) if params else 0.1,
            random_state=42, n_jobs=-1, verbosity=0
        )
    return Ridge()

def tune_hyperparameters(model_type, X, y, n_trials=15):
    """Quick hyperparameter tuning with Optuna"""
    
    def objective(trial):
        if model_type == 'Ridge':
            alpha = trial.suggest_float('alpha', 0.01, 100.0, log=True)
            model = Ridge(alpha=alpha)
        elif model_type == 'ElasticNet':
            alpha = trial.suggest_float('alpha', 0.01, 100.0, log=True)
            l1_ratio = trial.suggest_float('l1_ratio', 0.1, 0.9)
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        elif model_type == 'RandomForest':
            n_est = trial.suggest_int('n_estimators', 50, 200)
            max_d = trial.suggest_int('max_depth', 3, 15)
            min_split = trial.suggest_int('min_samples_split', 2, 10)
            model = RandomForestRegressor(n_estimators=n_est, max_depth=max_d,
                                          min_samples_split=min_split, random_state=42, n_jobs=-1)
        elif model_type == 'XGBoost' and HAS_XGBOOST:
            n_est = trial.suggest_int('n_estimators', 50, 200)
            max_d = trial.suggest_int('max_depth', 3, 10)
            lr = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
            model = XGBRegressor(n_estimators=n_est, max_depth=max_d, learning_rate=lr,
                                 random_state=42, n_jobs=-1, verbosity=0)
        else:
            return 0.0
        
        scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        return scores.mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params

def run_benchmark(model_type, sand_option, normalize, n_trials=15):
    """Run a single benchmark configuration"""
    
    if sand_option == 'exclude':
        train_data = train_agg.copy()
        test_data = test_agg.copy()
    else:
        smoothing = sand_option if sand_option in ['smooth_3x3', 'smooth_5x5'] else None
        train_data = add_sand_proportion(train_agg.copy(), sand_map, smoothing)
        test_data = add_sand_proportion(test_agg.copy(), sand_map, smoothing)
    
    train_data = add_engineered_features(train_data)
    test_data = add_engineered_features(test_data)
    
    exclude_cols = ['Well_ID', 'Target_BBL']
    feature_cols = [c for c in train_data.columns if c not in exclude_cols 
                    and train_data[c].dtype in ['float64', 'int64']]
    
    X = train_data[feature_cols].fillna(0)
    y = train_data['Target_BBL']
    
    if normalize:
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)
    else:
        X_scaled = X
    
    if model_type in ['Ridge', 'ElasticNet', 'RandomForest', 'XGBoost']:
        best_params = tune_hyperparameters(model_type, X_scaled, y, n_trials)
    else:
        best_params = {}
    
    model = get_model(model_type, best_params)
    
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    
    y_pred_test = model.predict(X_test)
    test_rmse = np.sqrt(np.mean((y_test - y_pred_test) ** 2))
    test_mae = np.mean(np.abs(y_test - y_pred_test))
    
    target_mean = y.mean()
    rmse_pct = (test_rmse / target_mean) * 100
    
    return {
        'model_type': model_type,
        'sand_option': sand_option,
        'normalize': normalize,
        'best_params': best_params,
        'cv_r2_mean': cv_mean,
        'cv_r2_std': cv_std,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'rmse_pct_of_mean': rmse_pct,
        'n_features': len(feature_cols)
    }

print("\n" + "=" * 70)
print("RUNNING COMPREHENSIVE BENCHMARKS")
print("=" * 70)

results = []
total_configs = len(MODEL_TYPES) * len(SAND_OPTIONS) * len(NORMALIZE_OPTIONS)
current = 0

for model_type in MODEL_TYPES:
    for sand_option in SAND_OPTIONS:
        for normalize in NORMALIZE_OPTIONS:
            current += 1
            config_name = f"{model_type} | Sand={sand_option} | Norm={normalize}"
            print(f"\n[{current}/{total_configs}] Testing: {config_name}")
            
            try:
                result = run_benchmark(model_type, sand_option, normalize)
                results.append(result)
                
                print(f"  CV R²: {result['cv_r2_mean']:.4f} ± {result['cv_r2_std']:.4f}")
                print(f"  Test R²: {result['test_r2']:.4f}")
                print(f"  RMSE: {result['test_rmse']:,.0f} BBL ({result['rmse_pct_of_mean']:.1f}% of mean)")
                
            except Exception as e:
                print(f"  ERROR: {str(e)}")
                results.append({
                    'model_type': model_type,
                    'sand_option': sand_option,
                    'normalize': normalize,
                    'error': str(e)
                })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('cv_r2_mean', ascending=False)

print("\n" + "=" * 70)
print("BENCHMARK RESULTS SUMMARY")
print("=" * 70)

print("\n📊 TOP 10 CONFIGURATIONS (by CV R²):")
print("-" * 70)
top_10 = results_df.head(10)
for i, row in top_10.iterrows():
    if 'error' not in row or pd.isna(row.get('error')):
        print(f"\n#{results_df.index.get_loc(i)+1}: {row['model_type']} | Sand={row['sand_option']} | Norm={row['normalize']}")
        print(f"   CV R²: {row['cv_r2_mean']:.4f} ± {row['cv_r2_std']:.4f}")
        print(f"   Test R²: {row['test_r2']:.4f} | RMSE: {row['test_rmse']:,.0f} BBL ({row['rmse_pct_of_mean']:.1f}%)")

print("\n\n📈 BEST BY MODEL TYPE:")
print("-" * 70)
best_by_model = results_df.groupby('model_type').apply(lambda x: x.nlargest(1, 'cv_r2_mean')).reset_index(drop=True)
for _, row in best_by_model.iterrows():
    if 'error' not in row or pd.isna(row.get('error')):
        print(f"\n{row['model_type']}:")
        print(f"  Best config: Sand={row['sand_option']}, Normalize={row['normalize']}")
        print(f"  CV R²: {row['cv_r2_mean']:.4f} | Test R²: {row['test_r2']:.4f}")
        print(f"  RMSE: {row['test_rmse']:,.0f} BBL ({row['rmse_pct_of_mean']:.1f}% of mean)")

print("\n\n🎯 INDUSTRY BENCHMARK COMPARISON:")
print("-" * 70)
target_mean = train_agg['Target_BBL'].mean()
print(f"Target Mean: {target_mean:,.0f} BBL")
print(f"Excellent threshold: RMSE < {target_mean * 0.10:,.0f} BBL (10%)")
print(f"Good threshold: RMSE < {target_mean * 0.15:,.0f} BBL (15%)")

excellent = results_df[results_df['rmse_pct_of_mean'] < 10]
good = results_df[(results_df['rmse_pct_of_mean'] >= 10) & (results_df['rmse_pct_of_mean'] < 15)]
print(f"\nExcellent configs (RMSE < 10%): {len(excellent)}")
print(f"Good configs (10% ≤ RMSE < 15%): {len(good)}")

benchmark_summary = {
    'timestamp': datetime.now().isoformat(),
    'n_training_wells': len(train_agg),
    'target_mean': float(target_mean),
    'target_std': float(train_agg['Target_BBL'].std()),
    'total_configs_tested': len(results),
    'top_configurations': [],
    'best_by_model': {},
    'recommendations': {}
}

for i, row in results_df.head(5).iterrows():
    if 'error' not in row or pd.isna(row.get('error')):
        benchmark_summary['top_configurations'].append({
            'rank': results_df.index.get_loc(i) + 1,
            'model_type': row['model_type'],
            'sand_option': row['sand_option'],
            'normalize': bool(row['normalize']),
            'cv_r2': float(row['cv_r2_mean']),
            'test_r2': float(row['test_r2']),
            'rmse': float(row['test_rmse']),
            'rmse_pct': float(row['rmse_pct_of_mean'])
        })

for _, row in best_by_model.iterrows():
    if 'error' not in row or pd.isna(row.get('error')):
        benchmark_summary['best_by_model'][row['model_type']] = {
            'sand_option': row['sand_option'],
            'normalize': bool(row['normalize']),
            'cv_r2': float(row['cv_r2_mean']),
            'test_r2': float(row['test_r2']),
            'rmse': float(row['test_rmse']),
            'rmse_pct': float(row['rmse_pct_of_mean']),
            'best_params': row.get('best_params', {})
        }

if len(results_df) > 0:
    best = results_df.iloc[0]
    benchmark_summary['recommendations'] = {
        'best_overall': {
            'model': best['model_type'],
            'sand': best['sand_option'],
            'normalize': bool(best['normalize']),
            'expected_cv_r2': float(best['cv_r2_mean']),
            'expected_test_r2': float(best['test_r2']),
            'expected_rmse_pct': float(best['rmse_pct_of_mean'])
        },
        'for_simplicity': 'Ridge with normalize=True, sand=smooth_3x3',
        'for_accuracy': f"{best['model_type']} with normalize={best['normalize']}, sand={best['sand_option']}",
        'avoid': 'Linear Regression (overfits with many features)'
    }

with open('outputs/benchmark_results.json', 'w') as f:
    json.dump(benchmark_summary, f, indent=2)

results_df.to_csv('outputs/benchmark_results.csv', index=False)

print("\n" + "=" * 70)
print("BENCHMARKING COMPLETE!")
print("=" * 70)
print(f"\nResults saved to:")
print(f"  - outputs/benchmark_results.json (summary)")
print(f"  - outputs/benchmark_results.csv (full details)")

print("\n🏆 FINAL RECOMMENDATIONS:")
print("-" * 70)
if len(results_df) > 0:
    best = results_df.iloc[0]
    print(f"\n✅ BEST OVERALL: {best['model_type']}")
    print(f"   Sand: {best['sand_option']}")
    print(f"   Normalize: {best['normalize']}")
    print(f"   Expected CV R²: {best['cv_r2_mean']:.4f}")
    print(f"   Expected Test R²: {best['test_r2']:.4f}")
    print(f"   Expected RMSE: {best['rmse_pct_of_mean']:.1f}% of mean")
