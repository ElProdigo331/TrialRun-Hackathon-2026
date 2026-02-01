"""
Fast Model Benchmarking - Tests key combinations quickly
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from scipy.ndimage import uniform_filter
import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

print("=" * 70)
print("FAST MODEL BENCHMARKING")
print("=" * 70)

train_df = pd.read_csv('outputs/MICE_imputed_production_wells.csv')
prod_history = pd.read_csv('data/Production_history_production_wells.csv')
sand_map = np.load('data/2d_sand_proportion.npy')

target_col = [c for c in prod_history.columns if 'cumulative' in c.lower() and 'oil' in c.lower()]
if not target_col:
    target_col = [c for c in prod_history.columns if 'oil' in c.lower()]
target_col = target_col[0] if target_col else prod_history.columns[-1]

targets = prod_history.groupby('Well_ID')[target_col].max().reset_index()
targets.columns = ['Well_ID', 'Target_BBL']

def aggregate_wells(df):
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
    return result.fillna(0)

train_agg = aggregate_wells(train_df)
train_agg = train_agg.merge(targets, on='Well_ID', how='inner')
target_mean = train_agg['Target_BBL'].mean()
target_std = train_agg['Target_BBL'].std()

print(f"Training wells: {len(train_agg)}")
print(f"Target mean: {target_mean:,.0f} BBL")

def add_sand_proportion(df, sand_map, smoothing=None):
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

CONFIGS = [
    ('Ridge', 'smooth_3x3', True, {'alpha': 10.0}),
    ('Ridge', 'smooth_3x3', False, {'alpha': 10.0}),
    ('Ridge', 'include', True, {'alpha': 10.0}),
    ('Ridge', 'exclude', True, {'alpha': 10.0}),
    ('ElasticNet', 'smooth_3x3', True, {'alpha': 1.0, 'l1_ratio': 0.5}),
    ('ElasticNet', 'smooth_3x3', False, {'alpha': 1.0, 'l1_ratio': 0.5}),
    ('RandomForest', 'smooth_3x3', True, {'n_estimators': 100, 'max_depth': 8}),
    ('RandomForest', 'smooth_3x3', False, {'n_estimators': 100, 'max_depth': 8}),
    ('RandomForest', 'include', True, {'n_estimators': 100, 'max_depth': 8}),
    ('RandomForest', 'exclude', True, {'n_estimators': 100, 'max_depth': 8}),
    ('RandomForest', 'smooth_3x3', True, {'n_estimators': 150, 'max_depth': 10}),
    ('RandomForest', 'smooth_3x3', True, {'n_estimators': 200, 'max_depth': 12}),
]

if HAS_XGBOOST:
    CONFIGS.extend([
        ('XGBoost', 'smooth_3x3', True, {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1}),
        ('XGBoost', 'smooth_3x3', False, {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1}),
        ('XGBoost', 'include', True, {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1}),
        ('XGBoost', 'smooth_3x3', True, {'n_estimators': 150, 'max_depth': 6, 'learning_rate': 0.05}),
        ('XGBoost', 'smooth_3x3', True, {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.05}),
    ])

def get_model(model_type, params):
    if model_type == 'Ridge':
        return Ridge(alpha=params.get('alpha', 1.0))
    elif model_type == 'ElasticNet':
        return ElasticNet(alpha=params.get('alpha', 1.0), l1_ratio=params.get('l1_ratio', 0.5))
    elif model_type == 'RandomForest':
        return RandomForestRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 10),
            random_state=42, n_jobs=-1
        )
    elif model_type == 'XGBoost' and HAS_XGBOOST:
        return XGBRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 6),
            learning_rate=params.get('learning_rate', 0.1),
            random_state=42, n_jobs=-1, verbosity=0
        )
    return Ridge()

results = []
print(f"\nTesting {len(CONFIGS)} configurations...\n")

for i, (model_type, sand_option, normalize, params) in enumerate(CONFIGS):
    print(f"[{i+1}/{len(CONFIGS)}] {model_type} | Sand={sand_option} | Norm={normalize}")
    
    if sand_option == 'exclude':
        data = train_agg.copy()
    else:
        smoothing = sand_option if sand_option in ['smooth_3x3', 'smooth_5x5'] else None
        data = add_sand_proportion(train_agg.copy(), sand_map, smoothing)
    
    data = add_engineered_features(data)
    
    exclude_cols = ['Well_ID', 'Target_BBL']
    feature_cols = [c for c in data.columns if c not in exclude_cols 
                    and data[c].dtype in ['float64', 'int64']]
    
    X = data[feature_cols].fillna(0)
    y = data['Target_BBL']
    
    if normalize:
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)
    else:
        X_scaled = X
    
    model = get_model(model_type, params)
    
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    
    y_pred_test = model.predict(X_test)
    test_rmse = np.sqrt(np.mean((y_test - y_pred_test) ** 2))
    rmse_pct = (test_rmse / target_mean) * 100
    
    print(f"  CV R²: {cv_mean:.4f} ± {cv_std:.4f} | Test R²: {test_r2:.4f} | RMSE: {rmse_pct:.1f}%")
    
    results.append({
        'model_type': model_type,
        'sand_option': sand_option,
        'normalize': normalize,
        'params': params,
        'cv_r2_mean': cv_mean,
        'cv_r2_std': cv_std,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'rmse_pct': rmse_pct
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('cv_r2_mean', ascending=False)

print("\n" + "=" * 70)
print("TOP 5 CONFIGURATIONS (by CV R²)")
print("=" * 70)

for i, row in results_df.head(5).iterrows():
    rank = results_df.index.get_loc(i) + 1
    print(f"\n#{rank}: {row['model_type']}")
    print(f"   Sand: {row['sand_option']} | Normalize: {row['normalize']}")
    print(f"   Params: {row['params']}")
    print(f"   CV R²: {row['cv_r2_mean']:.4f} ± {row['cv_r2_std']:.4f}")
    print(f"   Test R²: {row['test_r2']:.4f}")
    print(f"   RMSE: {row['test_rmse']:,.0f} BBL ({row['rmse_pct']:.1f}% of mean)")

print("\n" + "=" * 70)
print("BEST BY MODEL TYPE")
print("=" * 70)

for model_type in results_df['model_type'].unique():
    best = results_df[results_df['model_type'] == model_type].iloc[0]
    print(f"\n{model_type}:")
    print(f"  Best: Sand={best['sand_option']}, Normalize={best['normalize']}")
    print(f"  CV R²: {best['cv_r2_mean']:.4f} | Test R²: {best['test_r2']:.4f}")
    print(f"  RMSE: {best['rmse_pct']:.1f}% of mean")

benchmark_summary = {
    'timestamp': datetime.now().isoformat(),
    'n_wells': len(train_agg),
    'target_mean_bbl': float(target_mean),
    'target_std_bbl': float(target_std),
    'configs_tested': len(results),
    'top_5': [],
    'best_by_model': {},
    'recommendations': {}
}

for i, row in results_df.head(5).iterrows():
    benchmark_summary['top_5'].append({
        'rank': results_df.index.get_loc(i) + 1,
        'model': row['model_type'],
        'sand': row['sand_option'],
        'normalize': bool(row['normalize']),
        'params': row['params'],
        'cv_r2': round(float(row['cv_r2_mean']), 4),
        'cv_r2_std': round(float(row['cv_r2_std']), 4),
        'test_r2': round(float(row['test_r2']), 4),
        'rmse_pct': round(float(row['rmse_pct']), 1)
    })

for model_type in results_df['model_type'].unique():
    best = results_df[results_df['model_type'] == model_type].iloc[0]
    benchmark_summary['best_by_model'][model_type] = {
        'sand': best['sand_option'],
        'normalize': bool(best['normalize']),
        'params': best['params'],
        'cv_r2': round(float(best['cv_r2_mean']), 4),
        'test_r2': round(float(best['test_r2']), 4),
        'rmse_pct': round(float(best['rmse_pct']), 1)
    }

best_overall = results_df.iloc[0]
benchmark_summary['recommendations'] = {
    'best_overall': {
        'model': best_overall['model_type'],
        'sand': best_overall['sand_option'],
        'normalize': bool(best_overall['normalize']),
        'params': best_overall['params'],
        'expected_cv_r2': round(float(best_overall['cv_r2_mean']), 4),
        'expected_test_r2': round(float(best_overall['test_r2']), 4),
        'expected_rmse_pct': round(float(best_overall['rmse_pct']), 1)
    },
    'quick_start': 'RandomForest with n_estimators=100, max_depth=8, normalize=True, sand=smooth_3x3',
    'for_accuracy': f"{best_overall['model_type']} with {best_overall['params']}",
    'industry_context': {
        'excellent_threshold_r2': 0.93,
        'good_threshold_r2': 0.85,
        'excellent_rmse_pct': 10,
        'good_rmse_pct': 15
    }
}

with open('outputs/benchmark_results.json', 'w') as f:
    json.dump(benchmark_summary, f, indent=2)

results_df.to_csv('outputs/benchmark_results.csv', index=False)

print("\n" + "=" * 70)
print("BENCHMARKING COMPLETE!")
print("=" * 70)
print("\nFiles saved:")
print("  - outputs/benchmark_results.json")
print("  - outputs/benchmark_results.csv")

print("\n🏆 FINAL RECOMMENDATION:")
print("-" * 70)
print(f"Model: {best_overall['model_type']}")
print(f"Sand Map: {best_overall['sand_option']}")
print(f"Normalize: {best_overall['normalize']}")
print(f"Parameters: {best_overall['params']}")
print(f"Expected CV R²: {best_overall['cv_r2_mean']:.4f}")
print(f"Expected RMSE: {best_overall['rmse_pct']:.1f}% of mean")
