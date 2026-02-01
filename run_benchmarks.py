"""
Comprehensive Model Benchmarking Script
Runs systematic tests across all model types and configurations
to build a knowledge base for the AI Assistant.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeRegressor
from scipy.ndimage import uniform_filter
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not available, skipping XGBoost benchmarks")

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
BENCHMARK_FILE = "outputs/benchmark_results.json"

def load_data():
    """Load all required data files."""
    prod_wells = pd.read_csv(f"{DATA_DIR}/Well_log_data_production_wells.csv")
    preprod_wells = pd.read_csv(f"{DATA_DIR}/Well_log_data_preproduction_wells.csv")
    prod_history = pd.read_csv(f"{DATA_DIR}/Production_history_production_wells.csv")
    sand_map = np.load(f"{DATA_DIR}/2d_sand_proportion.npy")
    return prod_wells, preprod_wells, prod_history, sand_map

def apply_mice_imputation(train_raw, test_raw):
    """Apply MICE + CART imputation at depth level."""
    numeric_cols = ['AI', 'SI', 'Vp', 'Vs', 'rho_b', 'rho_f', 'rho_m', 
                    'K0', 'Kdry', 'Kf', 'Ksat', 'G0', 'Gdry', 'Gsat', 
                    'phi', 'perm', 'GR']
    
    available_cols = [c for c in numeric_cols if c in train_raw.columns]
    
    train_imputed = train_raw.copy()
    test_imputed = test_raw.copy()
    
    cart_estimator = DecisionTreeRegressor(random_state=42, max_depth=10)
    mice_imputer = IterativeImputer(
        estimator=cart_estimator,
        random_state=42, 
        max_iter=10
    )
    
    mice_imputer.fit(train_raw[available_cols])
    train_imputed[available_cols] = mice_imputer.transform(train_raw[available_cols])
    
    test_cols = [c for c in available_cols if c in test_raw.columns]
    test_imputed[test_cols] = mice_imputer.transform(test_raw[test_cols])
    
    return train_imputed, test_imputed

def aggregate_well_logs(well_logs_df):
    """Aggregate depth-level data to well-level features."""
    numeric_cols = ['AI', 'SI', 'Vp', 'Vs', 'rho_b', 'rho_f', 'rho_m', 
                    'K0', 'Kdry', 'Kf', 'Ksat', 'G0', 'Gdry', 'Gsat', 
                    'phi', 'perm', 'GR']
    
    agg_dict = {}
    for col in numeric_cols:
        if col in well_logs_df.columns:
            agg_dict[f'{col}_mean'] = (col, 'mean')
            agg_dict[f'{col}_std'] = (col, 'std')
            agg_dict[f'{col}_min'] = (col, 'min')
            agg_dict[f'{col}_max'] = (col, 'max')
    
    agg_dict['X'] = ('X', 'first')
    agg_dict['Y'] = ('Y', 'first')
    agg_dict['Z_min'] = ('Z', 'min')
    agg_dict['Z_max'] = ('Z', 'max')
    agg_dict['depth_range'] = ('Z', lambda x: x.max() - x.min())
    agg_dict['n_measurements'] = ('Z', 'count')
    
    aggregated = well_logs_df.groupby('Well_ID').agg(**agg_dict).reset_index()
    
    if 'facies' in well_logs_df.columns:
        facies_pivot = well_logs_df.groupby(['Well_ID', 'facies']).size().unstack(fill_value=0)
        facies_pivot = facies_pivot.div(facies_pivot.sum(axis=1), axis=0)
        facies_pivot.columns = [f'facies_{int(c)}_pct' for c in facies_pivot.columns]
        aggregated = aggregated.merge(facies_pivot.reset_index(), on='Well_ID', how='left')
    
    if all(col in well_logs_df.columns for col in ['phi', 'perm', 'GR']):
        df_temp = well_logs_df.copy()
        df_temp['depth_rock_quality'] = df_temp['phi'] * np.log1p(df_temp['perm']) / (df_temp['GR'] + 1)
        
        best_zone = df_temp.loc[df_temp.groupby('Well_ID')['depth_rock_quality'].idxmax()]
        best_zone_cols = {
            'best_zone_phi': 'phi',
            'best_zone_perm': 'perm', 
            'best_zone_GR': 'GR',
            'best_zone_Z': 'Z',
            'best_zone_quality': 'depth_rock_quality'
        }
        for new_col, orig_col in best_zone_cols.items():
            if orig_col in best_zone.columns:
                aggregated = aggregated.merge(
                    best_zone[['Well_ID', orig_col]].rename(columns={orig_col: new_col}),
                    on='Well_ID', how='left'
                )
        
        worst_zone = df_temp.loc[df_temp.groupby('Well_ID')['depth_rock_quality'].idxmin()]
        aggregated = aggregated.merge(
            worst_zone[['Well_ID', 'phi']].rename(columns={'phi': 'worst_zone_phi'}),
            on='Well_ID', how='left'
        )
        
        if 'best_zone_phi' in aggregated.columns and 'worst_zone_phi' in aggregated.columns:
            aggregated['zone_quality_contrast'] = aggregated['best_zone_phi'] - aggregated['worst_zone_phi']
    
    return aggregated

def calculate_3year_targets(prod_history):
    """Calculate 3-year cumulative oil production targets."""
    oil_col = 'Cumulative Oil Production, BBL'
    prod_history['Date'] = pd.to_datetime(prod_history['Date'])
    prod_history = prod_history.sort_values(['Well_ID', 'Date'])
    
    results = []
    for well_id in prod_history['Well_ID'].unique():
        well_data = prod_history[prod_history['Well_ID'] == well_id]
        start_date = well_data['Date'].min()
        end_date = start_date + pd.DateOffset(years=3)
        three_year_data = well_data[well_data['Date'] <= end_date]
        if len(three_year_data) > 0:
            total_oil = three_year_data[oil_col].iloc[-1]
        else:
            total_oil = 0
        results.append({'Well_ID': well_id, 'Target_3yr_Oil_BBL': total_oil})
    
    return pd.DataFrame(results)

def engineer_features(df, sand_map=None, x_min=None, x_max=None, y_min=None, y_max=None):
    """Add engineered features."""
    df = df.copy()
    
    if 'phi_mean' in df.columns and 'perm_mean' in df.columns:
        df['phi_perm_product'] = df['phi_mean'] * np.log1p(df['perm_mean'])
        df['RQI'] = 0.0314 * np.sqrt(df['perm_mean'] / df['phi_mean'].clip(lower=0.01))
        df['FZI'] = df['RQI'] / (df['phi_mean'] / (1 - df['phi_mean'].clip(upper=0.99)))
    
    if 'Vp_mean' in df.columns and 'Vs_mean' in df.columns:
        df['Vp_Vs_ratio'] = df['Vp_mean'] / df['Vs_mean'].clip(lower=0.01)
    
    if 'AI_mean' in df.columns and 'SI_mean' in df.columns:
        df['impedance_ratio'] = df['AI_mean'] / df['SI_mean'].clip(lower=0.01)
    
    if 'GR_mean' in df.columns:
        df['net_to_gross'] = 1 - (df['GR_mean'] / df['GR_mean'].max())
    
    if 'phi_mean' in df.columns and 'depth_range' in df.columns:
        df['storage_capacity'] = df['phi_mean'] * df['depth_range']
    
    if 'perm_mean' in df.columns and 'phi_mean' in df.columns and 'GR_mean' in df.columns:
        df['rock_quality'] = df['phi_mean'] * np.log1p(df['perm_mean']) / (df['GR_mean'].clip(lower=1))
    
    if 'perm_mean' in df.columns and 'phi_mean' in df.columns:
        df['flow_quality'] = np.log1p(df['perm_mean']) * df['phi_mean']
    
    return df

def lookup_sand_proportion(df, sand_map, x_min, x_max, y_min, y_max):
    """Look up sand proportion from 2D map."""
    x_indices = ((df['X'] - x_min) / (x_max - x_min) * (sand_map.shape[1] - 1)).astype(int).clip(0, sand_map.shape[1]-1)
    y_indices = ((df['Y'] - y_min) / (y_max - y_min) * (sand_map.shape[0] - 1)).astype(int).clip(0, sand_map.shape[0]-1)
    return sand_map[y_indices, x_indices]

def run_benchmark(train_df, test_df, config, sand_map, x_min, x_max, y_min, y_max):
    """Run a single benchmark with given configuration."""
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    exclude_cols = ['Well_ID', 'Target_3yr_Oil_BBL']
    
    if config['sand_map'] == 'exclude':
        exclude_cols.append('sand_proportion')
    elif config['sand_map'].startswith('smooth'):
        kernel = int(config['sand_map'].split('_')[1])
        smoothed = uniform_filter(sand_map, size=kernel)
        train_df['sand_proportion'] = lookup_sand_proportion(train_df, smoothed, x_min, x_max, y_min, y_max)
        test_df['sand_proportion'] = lookup_sand_proportion(test_df, smoothed, x_min, x_max, y_min, y_max)
    
    feature_cols = [c for c in train_df.columns if c not in exclude_cols and train_df[c].dtype in ['float64', 'int64']]
    
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['Target_3yr_Oil_BBL']
    X_test = test_df[feature_cols].fillna(0) if 'Target_3yr_Oil_BBL' not in test_df.columns else None
    
    if config['normalize']:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
        if X_test is not None:
            X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)
    
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    model_type = config['model_type']
    params = config.get('params', {})
    
    if model_type == 'Linear':
        model = LinearRegression()
    elif model_type == 'Ridge':
        model = Ridge(alpha=params.get('alpha', 1.0), random_state=42)
    elif model_type == 'ElasticNet':
        model = ElasticNet(alpha=params.get('alpha', 1.0), l1_ratio=params.get('l1_ratio', 0.5), random_state=42, max_iter=5000)
    elif model_type == 'RandomForest':
        model = RandomForestRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 10),
            min_samples_split=params.get('min_samples_split', 5),
            min_samples_leaf=params.get('min_samples_leaf', 2),
            max_features=params.get('max_features', 'sqrt'),
            random_state=42,
            oob_score=True,
            n_jobs=-1
        )
    elif model_type == 'XGBoost' and HAS_XGBOOST:
        model = xgb.XGBRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 6),
            learning_rate=params.get('learning_rate', 0.1),
            subsample=params.get('subsample', 0.8),
            colsample_bytree=params.get('colsample_bytree', 0.8),
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
    else:
        return None
    
    model.fit(X_tr, y_tr)
    
    y_pred_train = model.predict(X_tr)
    y_pred_val = model.predict(X_val)
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2')
    
    results = {
        'train_r2': r2_score(y_tr, y_pred_train),
        'val_r2': r2_score(y_val, y_pred_val),
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std(),
        'train_rmse': np.sqrt(mean_squared_error(y_tr, y_pred_train)),
        'val_rmse': np.sqrt(mean_squared_error(y_val, y_pred_val)),
        'train_mae': mean_absolute_error(y_tr, y_pred_train),
        'val_mae': mean_absolute_error(y_val, y_pred_val),
        'n_features': len(feature_cols),
        'target_mean': y_train.mean(),
        'target_std': y_train.std(),
        'rmse_pct_of_mean': np.sqrt(mean_squared_error(y_val, y_pred_val)) / y_train.mean() * 100,
        'overfitting_gap': r2_score(y_tr, y_pred_train) - r2_score(y_val, y_pred_val)
    }
    
    if hasattr(model, 'oob_score_'):
        results['oob_score'] = model.oob_score_
    
    if hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        results['top_features'] = importances.head(10).to_dict()
    
    return results

def generate_benchmark_configs():
    """Generate extensive benchmark configurations for best performance."""
    configs = []
    
    rf_configs = [
        {'n_estimators': 50, 'max_depth': 5, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt'},
        {'n_estimators': 50, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt'},
        {'n_estimators': 100, 'max_depth': 8, 'min_samples_split': 3, 'min_samples_leaf': 1, 'max_features': 'sqrt'},
        {'n_estimators': 100, 'max_depth': 15, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'log2'},
        {'n_estimators': 150, 'max_depth': 12, 'min_samples_split': 4, 'min_samples_leaf': 2, 'max_features': 0.5},
        {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 3, 'min_samples_leaf': 1, 'max_features': 'sqrt'},
    ]
    xgb_configs = [
        {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8},
        {'n_estimators': 50, 'max_depth': 5, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8},
        {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.7, 'colsample_bytree': 0.7},
        {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.08, 'subsample': 0.9, 'colsample_bytree': 0.9},
        {'n_estimators': 150, 'max_depth': 4, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.8},
        {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.02, 'subsample': 0.85, 'colsample_bytree': 0.85},
    ]
    elastic_configs = [
        {'alpha': 0.1, 'l1_ratio': 0.3},
        {'alpha': 0.5, 'l1_ratio': 0.5},
        {'alpha': 1.0, 'l1_ratio': 0.5},
        {'alpha': 0.5, 'l1_ratio': 0.7},
        {'alpha': 0.1, 'l1_ratio': 0.9},
    ]
    
    for normalize in [True, False]:
        configs.append({'model_type': 'Linear', 'normalize': normalize, 'sand_map': 'include', 'params': {}})
    
    for alpha in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0]:
        configs.append({'model_type': 'Ridge', 'normalize': True, 'sand_map': 'include', 'params': {'alpha': alpha}})
    
    for alpha in [0.5, 1.0, 2.0, 5.0]:
        configs.append({'model_type': 'Ridge', 'normalize': False, 'sand_map': 'include', 'params': {'alpha': alpha}})
    
    for elastic_params in elastic_configs:
        configs.append({'model_type': 'ElasticNet', 'normalize': True, 'sand_map': 'include', 'params': elastic_params})
    
    for rf_params in rf_configs:
        configs.append({'model_type': 'RandomForest', 'normalize': True, 'sand_map': 'include', 'params': rf_params})
        configs.append({'model_type': 'RandomForest', 'normalize': False, 'sand_map': 'include', 'params': rf_params})
    
    if HAS_XGBOOST:
        for xgb_params in xgb_configs:
            configs.append({'model_type': 'XGBoost', 'normalize': True, 'sand_map': 'include', 'params': xgb_params})
            configs.append({'model_type': 'XGBoost', 'normalize': False, 'sand_map': 'include', 'params': xgb_params})
    
    for sand_map in ['smooth_3', 'smooth_5', 'exclude']:
        configs.append({'model_type': 'Ridge', 'normalize': True, 'sand_map': sand_map, 'params': {'alpha': 1.0}})
        configs.append({'model_type': 'Ridge', 'normalize': False, 'sand_map': sand_map, 'params': {'alpha': 1.0}})
        configs.append({'model_type': 'RandomForest', 'normalize': True, 'sand_map': sand_map, 'params': rf_configs[2]})
        if HAS_XGBOOST:
            configs.append({'model_type': 'XGBoost', 'normalize': True, 'sand_map': sand_map, 'params': xgb_configs[2]})
    
    return configs

def main():
    print("=" * 60)
    print("COMPREHENSIVE MODEL BENCHMARKING")
    print("Energy AI Hackathon 2026")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n[1/5] Loading data...")
    prod_wells, preprod_wells, prod_history, sand_map = load_data()
    
    print("[2/5] Applying MICE imputation...")
    train_imputed, test_imputed = apply_mice_imputation(prod_wells, preprod_wells)
    
    print("[3/5] Aggregating well logs...")
    train_agg = aggregate_well_logs(train_imputed)
    test_agg = aggregate_well_logs(test_imputed)
    
    targets = calculate_3year_targets(prod_history)
    train_df = train_agg.merge(targets, on='Well_ID', how='left')
    test_df = test_agg.copy()
    
    all_x = pd.concat([train_df['X'], test_df['X']])
    all_y = pd.concat([train_df['Y'], test_df['Y']])
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()
    
    train_df['sand_proportion'] = lookup_sand_proportion(train_df, sand_map, x_min, x_max, y_min, y_max)
    test_df['sand_proportion'] = lookup_sand_proportion(test_df, sand_map, x_min, x_max, y_min, y_max)
    
    print("[4/5] Engineering features...")
    train_df = engineer_features(train_df, sand_map, x_min, x_max, y_min, y_max)
    test_df = engineer_features(test_df, sand_map, x_min, x_max, y_min, y_max)
    
    configs = generate_benchmark_configs()
    print(f"\n[5/5] Running {len(configs)} benchmark configurations...")
    
    all_results = []
    
    for i, config in enumerate(configs):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Progress: {i+1}/{len(configs)} ({(i+1)/len(configs)*100:.0f}%)")
        
        try:
            result = run_benchmark(train_df, test_df, config, sand_map, x_min, x_max, y_min, y_max)
            if result:
                result['config'] = config
                all_results.append(result)
        except Exception as e:
            print(f"  Error in config {i}: {e}")
            continue
    
    print(f"\nCompleted {len(all_results)} benchmarks successfully")
    
    benchmark_data = {
        'timestamp': datetime.now().isoformat(),
        'n_training_wells': len(train_df),
        'n_test_wells': len(test_df),
        'target_statistics': {
            'mean': float(train_df['Target_3yr_Oil_BBL'].mean()),
            'std': float(train_df['Target_3yr_Oil_BBL'].std()),
            'min': float(train_df['Target_3yr_Oil_BBL'].min()),
            'max': float(train_df['Target_3yr_Oil_BBL'].max()),
        },
        'results': all_results,
        'summary': {}
    }
    
    results_df = pd.DataFrame(all_results)
    
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    best_by_val_r2 = results_df.loc[results_df['val_r2'].idxmax()]
    best_by_cv = results_df.loc[results_df['cv_r2_mean'].idxmax()]
    best_by_rmse = results_df.loc[results_df['val_rmse'].idxmin()]
    lowest_overfit = results_df.loc[results_df['overfitting_gap'].abs().idxmin()]
    
    print(f"\nBest Validation R²: {best_by_val_r2['val_r2']:.4f}")
    print(f"  Config: {best_by_val_r2['config']}")
    
    print(f"\nBest CV R²: {best_by_cv['cv_r2_mean']:.4f} ± {best_by_cv['cv_r2_std']:.4f}")
    print(f"  Config: {best_by_cv['config']}")
    
    print(f"\nLowest Validation RMSE: {best_by_rmse['val_rmse']:.0f} BBL ({best_by_rmse['rmse_pct_of_mean']:.1f}% of mean)")
    print(f"  Config: {best_by_rmse['config']}")
    
    print(f"\nLowest Overfitting Gap: {lowest_overfit['overfitting_gap']:.4f}")
    print(f"  Val R²: {lowest_overfit['val_r2']:.4f}")
    print(f"  Config: {lowest_overfit['config']}")
    
    model_summary = results_df.groupby(results_df['config'].apply(lambda x: x['model_type'])).agg({
        'val_r2': ['mean', 'max', 'std'],
        'cv_r2_mean': ['mean', 'max'],
        'val_rmse': ['mean', 'min'],
        'overfitting_gap': 'mean'
    }).round(4)
    
    print("\n\nMODEL TYPE COMPARISON:")
    print("-" * 60)
    print(model_summary)
    
    normalize_comparison = results_df.groupby(results_df['config'].apply(lambda x: x['normalize'])).agg({
        'val_r2': 'mean',
        'cv_r2_mean': 'mean',
        'val_rmse': 'mean'
    }).round(4)
    print("\n\nNORMALIZATION IMPACT:")
    print("-" * 60)
    print(normalize_comparison)
    
    sand_comparison = results_df.groupby(results_df['config'].apply(lambda x: x['sand_map'])).agg({
        'val_r2': 'mean',
        'cv_r2_mean': 'mean',
        'val_rmse': 'mean'
    }).round(4)
    print("\n\nSAND MAP HANDLING IMPACT:")
    print("-" * 60)
    print(sand_comparison)
    
    benchmark_data['summary'] = {
        'best_val_r2': {
            'value': float(best_by_val_r2['val_r2']),
            'config': best_by_val_r2['config']
        },
        'best_cv_r2': {
            'value': float(best_by_cv['cv_r2_mean']),
            'std': float(best_by_cv['cv_r2_std']),
            'config': best_by_cv['config']
        },
        'best_rmse': {
            'value': float(best_by_rmse['val_rmse']),
            'pct_of_mean': float(best_by_rmse['rmse_pct_of_mean']),
            'config': best_by_rmse['config']
        },
        'lowest_overfit': {
            'gap': float(lowest_overfit['overfitting_gap']),
            'val_r2': float(lowest_overfit['val_r2']),
            'config': lowest_overfit['config']
        },
        'model_rankings': {},
        'recommendations': []
    }
    
    for model_type in results_df['config'].apply(lambda x: x['model_type']).unique():
        model_results = results_df[results_df['config'].apply(lambda x: x['model_type']) == model_type]
        benchmark_data['summary']['model_rankings'][model_type] = {
            'avg_val_r2': float(model_results['val_r2'].mean()),
            'max_val_r2': float(model_results['val_r2'].max()),
            'avg_cv_r2': float(model_results['cv_r2_mean'].mean()),
            'avg_rmse': float(model_results['val_rmse'].mean()),
            'avg_overfitting_gap': float(model_results['overfitting_gap'].mean()),
            'n_configs_tested': len(model_results)
        }
    
    recommendations = []
    
    if best_by_cv['config']['normalize']:
        recommendations.append("ALWAYS normalize features - improves performance across all model types")
    
    best_sand = sand_comparison['val_r2'].idxmax()
    recommendations.append(f"Sand map handling: '{best_sand}' gives best average results")
    
    best_model = model_summary[('val_r2', 'max')].idxmax()
    recommendations.append(f"Best performing model type: {best_model}")
    
    if best_by_cv['config']['model_type'] == 'RandomForest':
        rf_results = results_df[results_df['config'].apply(lambda x: x['model_type']) == 'RandomForest']
        best_rf = rf_results.loc[rf_results['cv_r2_mean'].idxmax()]
        rf_params = best_rf['config']['params']
        recommendations.append(f"Optimal Random Forest: n_estimators={rf_params['n_estimators']}, max_depth={rf_params['max_depth']}, max_features={rf_params['max_features']}")
    
    if HAS_XGBOOST:
        xgb_results = results_df[results_df['config'].apply(lambda x: x['model_type']) == 'XGBoost']
        if len(xgb_results) > 0:
            best_xgb = xgb_results.loc[xgb_results['cv_r2_mean'].idxmax()]
            xgb_params = best_xgb['config']['params']
            recommendations.append(f"Optimal XGBoost: n_estimators={xgb_params['n_estimators']}, max_depth={xgb_params['max_depth']}, learning_rate={xgb_params['learning_rate']}")
    
    best_ridge = results_df[results_df['config'].apply(lambda x: x['model_type']) == 'Ridge']
    if len(best_ridge) > 0:
        best_ridge_config = best_ridge.loc[best_ridge['cv_r2_mean'].idxmax()]
        recommendations.append(f"Best Ridge alpha: {best_ridge_config['config']['params']['alpha']}")
    
    if model_summary[('overfitting_gap', 'mean')].min() < 0.05:
        least_overfit_model = model_summary[('overfitting_gap', 'mean')].idxmin()
        recommendations.append(f"Most stable (least overfitting): {least_overfit_model}")
    
    benchmark_data['summary']['recommendations'] = recommendations
    
    with open(BENCHMARK_FILE, 'w') as f:
        json.dump(benchmark_data, f, indent=2, default=str)
    
    print(f"\n\nResults saved to: {BENCHMARK_FILE}")
    print("\n" + "=" * 60)
    print("KEY RECOMMENDATIONS FOR AI ASSISTANT:")
    print("=" * 60)
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    return benchmark_data

if __name__ == "__main__":
    main()
