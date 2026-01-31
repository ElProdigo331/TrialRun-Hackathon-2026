import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import warnings
warnings.filterwarnings('ignore')

from openai import OpenAI

def get_openai_client():
    return OpenAI(
        api_key=os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY"),
        base_url=os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL")
    )

st.set_page_config(
    page_title="Oil Production Prediction - Energy AI 2026",
    page_icon="🛢️",
    layout="wide"
)

st.title("🛢️ Energy AI Hackathon 2026")
st.markdown("**Predict 3-Year Cumulative Oil Production for 12 Preproduction Wells**")

DATA_DIR = "data"
OUTPUT_DIR = "outputs"

@st.cache_data
def load_2026_data():
    prod_wells = pd.read_csv(f"{DATA_DIR}/Well_log_data_production_wells.csv")
    preprod_wells = pd.read_csv(f"{DATA_DIR}/Well_log_data_preproduction_wells.csv")
    prod_history = pd.read_csv(f"{DATA_DIR}/Production_history_production_wells.csv")
    sand_map = np.load(f"{DATA_DIR}/2d_sand_proportion.npy")
    solution_template = pd.read_csv(f"{DATA_DIR}/solution.csv")
    return prod_wells, preprod_wells, prod_history, sand_map, solution_template

@st.cache_data
def aggregate_well_logs(well_logs_df):
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
    
    return aggregated

@st.cache_data
def calculate_3year_targets(prod_history):
    prod_history['Date'] = pd.to_datetime(prod_history['Date'])
    
    targets = []
    for well_id in prod_history['Well_ID'].unique():
        well_data = prod_history[prod_history['Well_ID'] == well_id].sort_values('Date')
        start_date = well_data['Date'].min()
        end_date = start_date + pd.DateOffset(years=3)
        
        within_3yr = well_data[well_data['Date'] <= end_date]
        if len(within_3yr) > 0:
            final_oil = within_3yr['Cumulative Oil Production, BBL'].iloc[-1]
            targets.append({'Well_ID': well_id, 'Target_3yr_Oil_BBL': final_oil})
    
    return pd.DataFrame(targets)

def lookup_sand_proportion(df, sand_map, x_min, x_max, y_min, y_max):
    x_coords = df['X'].values
    y_coords = df['Y'].values
    
    x_scaled = np.clip(((x_coords - x_min) / (x_max - x_min) * (sand_map.shape[1] - 1)).astype(int), 0, sand_map.shape[1] - 1)
    y_scaled = np.clip(((y_coords - y_min) / (y_max - y_min) * (sand_map.shape[0] - 1)).astype(int), 0, sand_map.shape[0] - 1)
    
    sand_values = sand_map[y_scaled, x_scaled]
    return sand_values

sidebar = st.sidebar
sidebar.header("Navigation")

pages = [
    "1. Data Loading & Aggregation",
    "2. Data Cleaning (MICE)", 
    "3. Exploratory Data Analysis",
    "4. Feature Engineering",
    "5. Model Training",
    "6. Generate Solution",
    "7. AI Assistant"
]

page = sidebar.radio("Select Step:", pages)

if 'train_df' not in st.session_state:
    st.session_state.train_df = None
if 'test_df' not in st.session_state:
    st.session_state.test_df = None
if 'targets' not in st.session_state:
    st.session_state.targets = None
if 'sand_map' not in st.session_state:
    st.session_state.sand_map = None
if 'solution_template' not in st.session_state:
    st.session_state.solution_template = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'residuals' not in st.session_state:
    st.session_state.residuals = None
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = None

if page == "1. Data Loading & Aggregation":
    st.header("Step 1: Data Loading & Well Log Aggregation")
    
    st.markdown("""
    **2026 Hackathon Problem:**
    - Predict **3-year cumulative oil production (BBL)** for **12 preproduction wells** (IDs 72-83)
    - Training data: 71 production wells with ~21 depth measurements each
    - Must aggregate depth measurements to one feature vector per well
    """)
    
    if st.button("Load 2026 Hackathon Data", type="primary"):
        with st.spinner("Loading data files..."):
            prod_wells, preprod_wells, prod_history, sand_map, solution_template = load_2026_data()
            
            st.session_state.sand_map = sand_map
            st.session_state.solution_template = solution_template
            
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Production Wells (Training)", prod_wells['Well_ID'].nunique())
            st.metric("Raw Rows", len(prod_wells))
        with col2:
            st.metric("Preproduction Wells (Test)", preprod_wells['Well_ID'].nunique())
            st.metric("Raw Rows", len(preprod_wells))
        with col3:
            st.metric("Production History Rows", len(prod_history))
            st.metric("Sand Map Shape", f"{sand_map.shape}")
        
        st.subheader("Aggregating Well Logs (Multi-Row → One Row per Well)")
        
        with st.spinner("Aggregating well logs..."):
            train_agg = aggregate_well_logs(prod_wells)
            test_agg = aggregate_well_logs(preprod_wells)
            
        with st.spinner("Calculating 3-year oil production targets..."):
            targets = calculate_3year_targets(prod_history)
            train_agg = train_agg.merge(targets, on='Well_ID', how='left')
        
        with st.spinner("Looking up sand proportion from map..."):
            all_x = pd.concat([train_agg['X'], test_agg['X']])
            all_y = pd.concat([train_agg['Y'], test_agg['Y']])
            x_min, x_max = all_x.min(), all_x.max()
            y_min, y_max = all_y.min(), all_y.max()
            
            train_agg['sand_proportion'] = lookup_sand_proportion(train_agg, sand_map, x_min, x_max, y_min, y_max)
            test_agg['sand_proportion'] = lookup_sand_proportion(test_agg, sand_map, x_min, x_max, y_min, y_max)
        
        st.session_state.train_df = train_agg
        st.session_state.test_df = test_agg
        st.session_state.targets = targets
        
        st.success(f"Data aggregated! Training: {len(train_agg)} wells, Test: {len(test_agg)} wells")
        
        st.subheader("Aggregated Training Data Preview")
        st.dataframe(train_agg.head(10), use_container_width=True)
        
        st.subheader("Target Distribution (3-Year Cumulative Oil BBL)")
        fig = px.histogram(train_agg, x='Target_3yr_Oil_BBL', nbins=20, 
                          title="Distribution of 3-Year Oil Production")
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Min Target", f"{train_agg['Target_3yr_Oil_BBL'].min():,.0f} BBL")
        col2.metric("Mean Target", f"{train_agg['Target_3yr_Oil_BBL'].mean():,.0f} BBL")
        col3.metric("Max Target", f"{train_agg['Target_3yr_Oil_BBL'].max():,.0f} BBL")
    
    if st.session_state.train_df is not None:
        st.subheader("Current Data Status")
        st.success(f"✅ Training data: {len(st.session_state.train_df)} wells")
        st.success(f"✅ Test data: {len(st.session_state.test_df)} wells")

elif page == "2. Data Cleaning (MICE)":
    st.header("Step 2: Data Cleaning with MICE Imputation")
    
    if st.session_state.train_df is None:
        st.warning("Please load data in Step 1 first.")
    else:
        train_df = st.session_state.train_df.copy()
        test_df = st.session_state.test_df.copy()
        
        st.subheader("Missing Values Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Training Data Missing Values:**")
            missing_train = train_df.isnull().sum()
            missing_train = missing_train[missing_train > 0]
            if len(missing_train) > 0:
                missing_df = pd.DataFrame({
                    'Column': missing_train.index, 
                    'Missing': missing_train.values,
                    'Pct': (missing_train.values / len(train_df) * 100).round(1)
                })
                st.dataframe(missing_df)
            else:
                st.success("No missing values!")
                
        with col2:
            st.markdown("**Test Data Missing Values:**")
            missing_test = test_df.isnull().sum()
            missing_test = missing_test[missing_test > 0]
            if len(missing_test) > 0:
                missing_df = pd.DataFrame({
                    'Column': missing_test.index, 
                    'Missing': missing_test.values,
                    'Pct': (missing_test.values / len(test_df) * 100).round(1)
                })
                st.dataframe(missing_df)
            else:
                st.success("No missing values!")
        
        st.subheader("Imputation Strategy")
        
        imputation_method = st.radio(
            "Choose Imputation Method:",
            ["Simple (Median)", "MICE (Multivariate Imputation by Chained Equations)"],
            index=1,
            help="MICE uses relationships between features for smarter imputation - recommended by hackathon host!"
        )
        
        if imputation_method == "MICE (Multivariate Imputation by Chained Equations)":
            st.info("**MICE:** Uses all features to predict missing values iteratively. Preserves correlations between features.")
        
        if st.button("Apply Imputation", type="primary"):
            numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c not in ['Well_ID', 'Target_3yr_Oil_BBL']]
            
            if imputation_method == "MICE (Multivariate Imputation by Chained Equations)":
                from sklearn.experimental import enable_iterative_imputer
                from sklearn.impute import IterativeImputer
                
                with st.spinner("Applying MICE imputation..."):
                    mice_imputer = IterativeImputer(random_state=42, max_iter=10)
                    
                    train_numeric = train_df[numeric_cols].copy()
                    test_numeric = test_df[[c for c in numeric_cols if c in test_df.columns]].copy()
                    
                    mice_imputer.fit(train_numeric)
                    train_df[numeric_cols] = mice_imputer.transform(train_numeric)
                    
                    test_num_cols = [c for c in numeric_cols if c in test_df.columns]
                    if test_num_cols:
                        test_df[test_num_cols] = mice_imputer.transform(test_df[test_num_cols])
                    
                    st.success("MICE imputation complete!")
            else:
                with st.spinner("Applying median imputation..."):
                    for col in numeric_cols:
                        if col in train_df.columns:
                            median_val = train_df[col].median()
                            train_df[col] = train_df[col].fillna(median_val)
                            if col in test_df.columns:
                                test_df[col] = test_df[col].fillna(median_val)
                    st.success("Median imputation complete!")
            
            train_df = train_df.fillna(0)
            test_df = test_df.fillna(0)
            
            st.session_state.train_df = train_df
            st.session_state.test_df = test_df
            
            remaining = train_df.isnull().sum().sum() + test_df.isnull().sum().sum()
            st.metric("Remaining Missing Values", remaining)

elif page == "3. Exploratory Data Analysis":
    st.header("Step 3: Exploratory Data Analysis")
    
    if st.session_state.train_df is None:
        st.warning("Please load data in Step 1 first.")
    else:
        df = st.session_state.train_df
        
        tab1, tab2, tab3 = st.tabs(["Target Analysis", "Feature Analysis", "Correlations"])
        
        with tab1:
            st.subheader("3-Year Oil Production Distribution")
            
            fig = px.histogram(df, x='Target_3yr_Oil_BBL', nbins=25, 
                              title="Distribution of 3-Year Cumulative Oil Production (BBL)")
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Min", f"{df['Target_3yr_Oil_BBL'].min():,.0f}")
            col2.metric("Mean", f"{df['Target_3yr_Oil_BBL'].mean():,.0f}")
            col3.metric("Median", f"{df['Target_3yr_Oil_BBL'].median():,.0f}")
            col4.metric("Max", f"{df['Target_3yr_Oil_BBL'].max():,.0f}")
            
            st.subheader("Spatial Distribution")
            fig = px.scatter(df, x='X', y='Y', color='Target_3yr_Oil_BBL',
                           size='Target_3yr_Oil_BBL', hover_data=['Well_ID'],
                           title="Well Locations Colored by Oil Production",
                           color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Key Petrophysical Features")
            
            key_features = ['phi_mean', 'perm_mean', 'GR_mean', 'AI_mean', 'sand_proportion']
            available = [f for f in key_features if f in df.columns]
            
            selected_feature = st.selectbox("Select Feature:", available)
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(df, x=selected_feature, nbins=25, 
                                  title=f"{selected_feature} Distribution")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.scatter(df, x=selected_feature, y='Target_3yr_Oil_BBL',
                               title=f"{selected_feature} vs Oil Production",
                               trendline="ols")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Feature Correlations with Target")
            
            numeric_df = df.select_dtypes(include=[np.number])
            if 'Target_3yr_Oil_BBL' in numeric_df.columns:
                correlations = numeric_df.corr()['Target_3yr_Oil_BBL'].drop('Target_3yr_Oil_BBL')
                correlations = correlations.abs().sort_values(ascending=False).head(20)
                
                fig = px.bar(x=correlations.values, y=correlations.index, orientation='h',
                           title="Top 20 Features Correlated with Oil Production",
                           labels={'x': 'Absolute Correlation', 'y': 'Feature'})
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Top 10 Correlations")
                for feat, corr in correlations.head(10).items():
                    st.markdown(f"- **{feat}**: {corr:.3f}")

elif page == "4. Feature Engineering":
    st.header("Step 4: Feature Engineering")
    
    if st.session_state.train_df is None:
        st.warning("Please load data in Step 1 first.")
    else:
        df = st.session_state.train_df.copy()
        test_df = st.session_state.test_df.copy()
        
        st.markdown("""
        **Current Features (from aggregation):**
        - Mean, std, min, max of petrophysical logs (phi, perm, AI, GR, etc.)
        - Spatial features (X, Y, depth range)
        - Facies percentages
        - Sand proportion from seismic map
        """)
        
        st.subheader("Additional Feature Engineering")
        
        st.markdown("""
        **Proposed Additional Features:**
        1. **phi_perm_product** = phi_mean × log(perm_mean) - productivity indicator
        2. **rock_quality** = phi_mean / GR_mean - reservoir quality index
        3. **impedance_ratio** = AI_mean / SI_mean - lithology indicator
        """)
        
        if st.button("Apply Feature Engineering", type="primary"):
            for dataset in [df, test_df]:
                if 'phi_mean' in dataset.columns and 'perm_mean' in dataset.columns:
                    dataset['phi_perm_product'] = dataset['phi_mean'] * np.log1p(dataset['perm_mean'])
                
                if 'phi_mean' in dataset.columns and 'GR_mean' in dataset.columns:
                    dataset['rock_quality'] = dataset['phi_mean'] / (dataset['GR_mean'] + 1)
                
                if 'AI_mean' in dataset.columns and 'SI_mean' in dataset.columns:
                    dataset['impedance_ratio'] = dataset['AI_mean'] / (dataset['SI_mean'] + 1)
            
            st.session_state.train_df = df
            st.session_state.test_df = test_df
            st.success("Feature engineering complete!")
            
            st.subheader("New Features Preview")
            new_cols = ['Well_ID', 'phi_perm_product', 'rock_quality', 'impedance_ratio']
            available_cols = [c for c in new_cols if c in df.columns]
            st.dataframe(df[available_cols].head(10), use_container_width=True)

elif page == "5. Model Training":
    st.header("Step 5: Model Training")
    
    if st.session_state.train_df is None:
        st.warning("Please load data in Step 1 first.")
    else:
        train_df = st.session_state.train_df.copy()
        
        st.subheader("Model Configuration")
        
        tuning_mode = st.radio(
            "Hyperparameter Tuning Mode:",
            ["Manual", "Optuna (Auto-Tune)"],
            help="Optuna automatically finds optimal hyperparameters using Bayesian optimization"
        )
        
        col1, col2 = st.columns(2)
        
        if tuning_mode == "Manual":
            with col1:
                n_estimators = st.slider("Number of Trees", 50, 300, 100, 50)
                max_depth = st.selectbox("Max Depth", [None, 5, 10, 15, 20], index=3)
            with col2:
                min_samples_split = st.slider("Min Samples Split", 2, 20, 5)
                cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
        else:
            with col1:
                n_trials = st.slider("Optuna Trials", 10, 100, 30, 10)
                cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
            with col2:
                st.info("Optuna will search:\n- n_estimators: 50-300\n- max_depth: 3-20\n- min_samples_split: 2-20")
        
        exclude_cols = ['Well_ID', 'Target_3yr_Oil_BBL']
        feature_cols = [c for c in train_df.columns if c not in exclude_cols and train_df[c].dtype in ['float64', 'int64']]
        
        st.markdown(f"**Features available:** {len(feature_cols)}")
        
        if st.button("Train Model", type="primary"):
            X = train_df[feature_cols].fillna(0)
            y = train_df['Target_3yr_Oil_BBL']
            
            st.session_state.feature_cols = feature_cols
            
            if tuning_mode == "Optuna (Auto-Tune)":
                import optuna
                optuna.logging.set_verbosity(optuna.logging.WARNING)
                
                with st.spinner(f"Optuna tuning model ({n_trials} trials)..."):
                    def objective(trial):
                        params = {
                            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                            'max_depth': trial.suggest_int('max_depth', 3, 20),
                            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                            'random_state': 42,
                            'n_jobs': -1
                        }
                        model = RandomForestRegressor(**params)
                        scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
                        return scores.mean()
                    
                    study = optuna.create_study(direction='maximize')
                    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
                    
                    best_params = study.best_params
                    best_params['random_state'] = 42
                    best_params['n_jobs'] = -1
                    
                    st.success(f"Best params: n_estimators={best_params['n_estimators']}, max_depth={best_params['max_depth']}, min_samples_split={best_params['min_samples_split']}")
                    
                    model = RandomForestRegressor(**best_params)
            else:
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42,
                    n_jobs=-1
                )
            
            with st.spinner("Training final model..."):
                from sklearn.model_selection import cross_val_predict
                
                cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
                
                y_pred_cv = cross_val_predict(model, X, y, cv=cv_folds)
                cv_residuals = y - y_pred_cv
                
                model.fit(X, y)
                y_pred = model.predict(X)
                residuals = cv_residuals
                
                st.session_state.model = model
                st.session_state.residuals = residuals.values
            
            st.success("Model trained successfully!")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("CV R² Mean", f"{cv_scores.mean():.4f}")
            col2.metric("CV R² Std", f"{cv_scores.std():.4f}")
            col3.metric("Train R²", f"{r2_score(y, y_pred):.4f}")
            
            st.subheader("Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(20)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                        title="Top 20 Feature Importances")
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Residual Analysis")
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(x=residuals, nbins=30, title="Residual Distribution")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.scatter(x=y_pred, y=residuals, title="Residuals vs Predicted",
                               labels={'x': 'Predicted', 'y': 'Residual'})
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)

elif page == "6. Generate Solution":
    st.header("Step 6: Generate Solution File")
    
    if st.session_state.model is None:
        st.warning("Please train a model in Step 5 first.")
    else:
        model = st.session_state.model
        test_df = st.session_state.test_df
        residuals = st.session_state.residuals
        feature_cols = st.session_state.feature_cols
        
        st.markdown("""
        **Solution Requirements:**
        - 12 rows (Wells 72-83)
        - Columns: Well_ID, Prediction_BBL, R1-R100
        - Replace all -9999 values with predictions
        """)
        
        n_realizations = st.slider("Number of Realizations", 10, 100, 100)
        
        if st.button("Generate Predictions", type="primary"):
            test_df_sorted = test_df.sort_values('Well_ID').reset_index(drop=True)
            X_test = test_df_sorted[feature_cols].fillna(0)
            
            point_predictions = model.predict(X_test)
            
            realizations = np.zeros((len(test_df_sorted), n_realizations))
            for i in range(n_realizations):
                sampled_residuals = np.random.choice(residuals, size=len(test_df_sorted), replace=True)
                realizations[:, i] = point_predictions + sampled_residuals
                realizations[:, i] = np.maximum(realizations[:, i], 0)
            
            solution = pd.DataFrame()
            solution['Well_ID'] = test_df_sorted['Well_ID'].astype(int).values
            solution['Prediction_BBL'] = point_predictions.round(0).astype(int)
            
            for i in range(n_realizations):
                solution[f'R{i+1}'] = realizations[:, i].round(0).astype(int)
            
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            solution.to_csv(f"{OUTPUT_DIR}/solution.csv", index=False)
            solution.to_csv(f"{DATA_DIR}/solution.csv", index=False)
            
            st.success("Solution generated and saved!")
            
            st.subheader("Solution Preview")
            st.dataframe(solution.head(12), use_container_width=True)
            
            st.subheader("Prediction Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Min Prediction", f"{solution['Prediction_BBL'].min():,.0f} BBL")
            col2.metric("Mean Prediction", f"{solution['Prediction_BBL'].mean():,.0f} BBL")
            col3.metric("Max Prediction", f"{solution['Prediction_BBL'].max():,.0f} BBL")
            
            st.subheader("Uncertainty Visualization")
            fig = go.Figure()
            for idx, row in solution.iterrows():
                well_id = row['Well_ID']
                reals = row[[f'R{i+1}' for i in range(min(100, n_realizations))]].values
                fig.add_trace(go.Box(y=reals, name=f"Well {int(well_id)}", 
                                    boxpoints=False, marker_color='steelblue'))
            fig.update_layout(title="Uncertainty Distribution by Well", 
                            xaxis_title="Well", yaxis_title="Oil Production (BBL)")
            st.plotly_chart(fig, use_container_width=True)
            
            csv_data = solution.to_csv(index=False)
            st.download_button(
                label="Download solution.csv",
                data=csv_data,
                file_name="solution.csv",
                mime="text/csv"
            )

elif page == "7. AI Assistant":
    st.header("🤖 AI ML Assistant")
    
    st.markdown("""
    Ask me anything about:
    - The 2026 hackathon problem (oil production prediction)
    - Feature engineering for petrophysical data
    - Model selection and tuning
    - Uncertainty quantification
    - Adapting to different datasets
    """)
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask about ML, the hackathon, or get help..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    client = get_openai_client()
                    
                    system_prompt = """You are an expert ML assistant for the Energy AI Hackathon 2026, built by Team Brain Oil.

The 2026 hackathon problem is:
- Predict 3-year cumulative oil production (BBL) for 12 preproduction wells
- Training data: 71 wells with petrophysical well logs (multiple depth measurements per well)
- Features: porosity (phi), permeability (perm), gamma ray (GR), acoustic impedance (AI), facies, etc.
- Must aggregate depth measurements per well (mean, std, min, max)
- Output: Point estimate + 100 realizations (R1-R100) for uncertainty

Key techniques being used:
- MICE imputation for missing values (recommended by hackathon host)
- Optuna for hyperparameter tuning
- Random Forest regression
- Residual bootstrapping for uncertainty quantification
- Shapley values for feature importance (from workshop)

Provide helpful, practical advice for winning the hackathon."""

                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=1000
                    )
                    
                    assistant_response = response.choices[0].message.content
                    st.markdown(assistant_response)
                    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
