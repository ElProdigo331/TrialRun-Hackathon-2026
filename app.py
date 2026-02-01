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
def apply_mice_to_raw_data(train_raw, test_raw):
    """
    Apply MICE + CART imputation at the depth level (before aggregation).
    Uses Decision Trees (CART) as the estimator per SPE 218890 (Abdulkhaleq et al. 2024):
    "MICE + CART outperformed other methods for both clastic and carbonate reservoirs."
    
    This preserves correlations between features and is the academically
    recommended approach per Van Buuren (2018) and Hallam et al. (2022).
    """
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.tree import DecisionTreeRegressor
    
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
    
    # BEST ZONE FEATURES: Preserve depth heterogeneity by extracting "pay zone" characteristics
    # Find the depth with best rock quality (high phi, high perm, low GR)
    if all(col in well_logs_df.columns for col in ['phi', 'perm', 'GR']):
        # Create rock quality score at each depth
        df_temp = well_logs_df.copy()
        df_temp['depth_rock_quality'] = df_temp['phi'] * np.log1p(df_temp['perm']) / (df_temp['GR'] + 1)
        
        # Get features from best depth (highest rock quality)
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
        
        # Also get worst zone to capture heterogeneity
        worst_zone = df_temp.loc[df_temp.groupby('Well_ID')['depth_rock_quality'].idxmin()]
        aggregated = aggregated.merge(
            worst_zone[['Well_ID', 'phi']].rename(columns={'phi': 'worst_zone_phi'}),
            on='Well_ID', how='left'
        )
        
        # Quality contrast: difference between best and worst zones
        if 'best_zone_phi' in aggregated.columns and 'worst_zone_phi' in aggregated.columns:
            aggregated['zone_quality_contrast'] = aggregated['best_zone_phi'] - aggregated['worst_zone_phi']
    
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
    "7. AI Assistant",
    "8. Scholarly Analysis"
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
    st.header("Step 1: Data Loading, MICE Imputation & Aggregation")
    
    st.markdown("""
    **2026 Hackathon Problem:**
    - Predict **3-year cumulative oil production (BBL)** for **12 preproduction wells** (IDs 72-83)
    - Training data: 71 production wells with ~21 depth measurements each
    - Must aggregate depth measurements to one feature vector per well
    
    **Workflow (Academically Correct Order):**
    1. Load raw well log data
    2. **Apply MICE imputation at depth level** (per Van Buuren 2018, Hallam et al. 2022)
    3. Aggregate imputed data to one row per well
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
        
        st.subheader("Step 1a: MICE Imputation (Before Aggregation)")
        
        numeric_cols = ['AI', 'SI', 'Vp', 'Vs', 'rho_b', 'rho_f', 'rho_m', 
                        'K0', 'Kdry', 'Kf', 'Ksat', 'G0', 'Gdry', 'Gsat', 
                        'phi', 'perm', 'GR']
        available_cols = [c for c in numeric_cols if c in prod_wells.columns]
        
        missing_before = prod_wells[available_cols].isnull().sum().sum()
        missing_pct = missing_before / (len(prod_wells) * len(available_cols)) * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Missing Values (Raw Data)", f"{missing_before:,}")
            st.metric("Missing Percentage", f"{missing_pct:.1f}%")
        with col2:
            st.info("""
            **Why MICE + CART?**
            - **CART** (Decision Trees) handles non-linear relationships
            - Outperforms other methods per SPE 218890 (Abdulkhaleq 2024)
            - Preserves phi-perm-GR correlations
            - All 21 depth measurements contribute
            """)
        
        with st.spinner("Applying MICE + CART imputation at depth level..."):
            prod_wells_imputed, preprod_wells_imputed = apply_mice_to_raw_data(prod_wells, preprod_wells)
        
        missing_after = prod_wells_imputed[available_cols].isnull().sum().sum()
        st.success(f"MICE + CART imputation complete! Missing values: {missing_before:,} → {missing_after}")
        
        st.subheader("Step 1b: Aggregating Well Logs (Multi-Row → One Row per Well)")
        
        with st.spinner("Aggregating imputed well logs..."):
            train_agg = aggregate_well_logs(prod_wells_imputed)
            test_agg = aggregate_well_logs(preprod_wells_imputed)
            
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
        st.session_state.mice_applied = True
        
        st.success(f"Data processed! Training: {len(train_agg)} wells, Test: {len(test_agg)} wells")
        
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
    st.header("Step 2: Data Quality Verification")
    
    if st.session_state.train_df is None:
        st.warning("Please load data in Step 1 first.")
    else:
        train_df = st.session_state.train_df.copy()
        test_df = st.session_state.test_df.copy()
        
        st.success("""
        **MICE + CART imputation was already applied in Step 1 (at the depth level, before aggregation).**
        
        This is the academically correct approach per:
        - Van Buuren (2018): *Flexible Imputation of Missing Data*
        - Hallam et al. (2022): *Multivariate imputation for elastic well log data*
        - **SPE 218890 (Abdulkhaleq et al. 2024):** *MICE + CART outperformed other methods*
        
        Using CART (Decision Trees) as the estimator handles non-linear relationships in petrophysical data.
        """)
        
        st.subheader("Post-Imputation Data Quality Check")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Training Data (Aggregated):**")
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
                st.success("No missing values in aggregated training data!")
                
        with col2:
            st.markdown("**Test Data (Aggregated):**")
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
                st.success("No missing values in aggregated test data!")
        
        st.subheader("Why MICE Before Aggregation?")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Old Approach (Aggregate First):**
            - Missing depths contribute **nothing** to statistics
            - Mean calculated from 19 values (if 2 missing)
            - Loses information
            """)
        with col2:
            st.markdown("""
            **Current Approach (MICE First):**
            - Missing depths get estimated values
            - Mean calculated from **all 21 values**
            - Preserves phi-perm-GR correlations
            """)
        
        st.info("""
        **Research Backing:**
        > "Impute at the finest granularity level (before aggregation) to preserve 
        > within-cluster correlations and full variance structure."
        > — Van Buuren (2018), Hallam et al. (2022)
        """)
        
        if st.button("Verify Data Quality & Proceed", type="primary"):
            train_df = train_df.fillna(0)
            test_df = test_df.fillna(0)
            
            st.session_state.train_df = train_df
            st.session_state.test_df = test_df
            
            remaining = train_df.isnull().sum().sum() + test_df.isnull().sum().sum()
            st.metric("Remaining Missing Values", remaining)
            st.success("Data quality verified! Proceed to Step 3 for EDA.")

elif page == "3. Exploratory Data Analysis":
    st.header("Step 3: Exploratory Data Analysis")
    
    if st.session_state.train_df is None:
        st.warning("Please load data in Step 1 first.")
    else:
        df = st.session_state.train_df
        
        tab1, tab2, tab3, tab4 = st.tabs(["Target Analysis", "Feature Analysis", "Correlations", "Rock Quality Analysis"])
        
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
        
        with tab4:
            st.subheader("Rock Quality vs Production (Industry Expert Insights)")
            
            st.info("""
            **Industry Expert Advice:** "Good rock will more than likely have oil"
            
            In clastic/sandstone reservoirs, **good rock** means:
            - **High porosity (phi)** → storage capacity
            - **High permeability (perm)** → flow ability
            - **Low Gamma Ray (GR)** → clean sand, less shale
            - **High sand proportion** → better reservoir at location
            """)
            
            rock_quality_features = {
                'phi_mean': 'Porosity (Higher = More Storage)',
                'perm_mean': 'Permeability (Higher = Better Flow)',
                'GR_mean': 'Gamma Ray (Lower = Cleaner Sand)',
                'sand_proportion': 'Sand Proportion (Higher = Better Rock)',
                'rock_quality': 'Rock Quality Index (phi/GR)',
                'phi_perm_product': 'Productivity Index (phi × log(perm))'
            }
            
            available_rq = {k: v for k, v in rock_quality_features.items() if k in df.columns}
            
            if available_rq:
                st.subheader("Rock Quality Indicators vs Oil Production")
                
                selected_rq = st.selectbox("Select Rock Quality Indicator:", 
                                          list(available_rq.keys()),
                                          format_func=lambda x: f"{x} - {available_rq[x]}")
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.scatter(df, x=selected_rq, y='Target_3yr_Oil_BBL',
                                   hover_data=['Well_ID'],
                                   title=f"{selected_rq} vs Oil Production",
                                   trendline="ols",
                                   color='Target_3yr_Oil_BBL',
                                   color_continuous_scale='RdYlGn')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    corr_val = df[selected_rq].corr(df['Target_3yr_Oil_BBL'])
                    st.metric("Correlation with Production", f"{corr_val:.3f}")
                    
                    if corr_val > 0.3:
                        st.success("Strong positive relationship - good rock indicator!")
                    elif corr_val > 0.1:
                        st.info("Moderate positive relationship")
                    elif corr_val < -0.3:
                        st.warning("Strong negative relationship (inverse indicator)")
                    else:
                        st.info("Weak relationship")
                
                st.divider()
                st.subheader("Good vs Bad Rock Classification")
                
                if 'phi_mean' in df.columns and 'GR_mean' in df.columns:
                    median_phi = df['phi_mean'].median()
                    median_gr = df['GR_mean'].median()
                    
                    df['rock_class'] = 'Average'
                    df.loc[(df['phi_mean'] > median_phi) & (df['GR_mean'] < median_gr), 'rock_class'] = 'Good Rock (High φ, Low GR)'
                    df.loc[(df['phi_mean'] < median_phi) & (df['GR_mean'] > median_gr), 'rock_class'] = 'Poor Rock (Low φ, High GR)'
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = px.scatter(df, x='phi_mean', y='GR_mean', 
                                       color='rock_class',
                                       size='Target_3yr_Oil_BBL',
                                       hover_data=['Well_ID', 'Target_3yr_Oil_BBL'],
                                       title="Rock Quality Classification",
                                       color_discrete_map={
                                           'Good Rock (High φ, Low GR)': 'green',
                                           'Poor Rock (Low φ, High GR)': 'red',
                                           'Average': 'gray'
                                       })
                        fig.add_hline(y=median_gr, line_dash="dash", line_color="gray")
                        fig.add_vline(x=median_phi, line_dash="dash", line_color="gray")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        rock_stats = df.groupby('rock_class')['Target_3yr_Oil_BBL'].agg(['mean', 'count']).round(0)
                        rock_stats.columns = ['Mean Production (BBL)', 'Well Count']
                        st.dataframe(rock_stats, use_container_width=True)
                        
                        good_rock_prod = df[df['rock_class'] == 'Good Rock (High φ, Low GR)']['Target_3yr_Oil_BBL'].mean()
                        poor_rock_prod = df[df['rock_class'] == 'Poor Rock (Low φ, High GR)']['Target_3yr_Oil_BBL'].mean()
                        
                        if good_rock_prod > poor_rock_prod:
                            pct_diff = ((good_rock_prod - poor_rock_prod) / poor_rock_prod * 100)
                            st.success(f"✅ Good rock produces **{pct_diff:.0f}% more** oil than poor rock!")
                        
                    df.drop('rock_class', axis=1, inplace=True)
                
                st.divider()
                st.subheader("Location Impact on Production")
                
                if 'X' in df.columns and 'Y' in df.columns:
                    fig = px.scatter(df, x='X', y='Y', 
                                   color='Target_3yr_Oil_BBL',
                                   size='Target_3yr_Oil_BBL',
                                   hover_data=['Well_ID', 'phi_mean', 'perm_mean', 'GR_mean'],
                                   title="Well Locations - Size & Color by Production",
                                   color_continuous_scale='RdYlGn')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    x_corr = df['X'].corr(df['Target_3yr_Oil_BBL'])
                    y_corr = df['Y'].corr(df['Target_3yr_Oil_BBL'])
                    col1, col2 = st.columns(2)
                    col1.metric("X-coordinate Correlation", f"{x_corr:.3f}")
                    col2.metric("Y-coordinate Correlation", f"{y_corr:.3f}")

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
        
        st.subheader("Rock Quality Feature Engineering (Industry Expert Advice)")
        
        st.info("""
        **Industry Expert Insight:** "Take note of reservoir quality - good rock will more than likely have oil"
        
        Creating features that capture **good vs bad rock**:
        """)
        
        st.markdown("""
        **Rock Quality Features:**
        | Feature | Formula | Interpretation |
        |---------|---------|----------------|
        | **phi_perm_product** | phi × log(perm) | Flow productivity (higher = better) |
        | **rock_quality** | phi / GR | Clean sand index (higher = cleaner) |
        | **impedance_ratio** | AI / SI | Lithology contrast |
        | **net_to_gross** | (1 - facies_5% - facies_6%) | Sand vs shale ratio |
        | **storage_capacity** | phi × (depth_range) | Total pore volume proxy |
        | **flow_quality** | log(perm) / GR | Flow per unit shaliness |
        """)
        
        if st.button("Apply Feature Engineering", type="primary"):
            for dataset in [df, test_df]:
                if 'phi_mean' in dataset.columns and 'perm_mean' in dataset.columns:
                    dataset['phi_perm_product'] = dataset['phi_mean'] * np.log1p(dataset['perm_mean'])
                
                if 'phi_mean' in dataset.columns and 'GR_mean' in dataset.columns:
                    dataset['rock_quality'] = dataset['phi_mean'] / (dataset['GR_mean'] + 1)
                
                if 'AI_mean' in dataset.columns and 'SI_mean' in dataset.columns:
                    dataset['impedance_ratio'] = dataset['AI_mean'] / (dataset['SI_mean'] + 1)
                
                if 'facies_5_pct' in dataset.columns and 'facies_6_pct' in dataset.columns:
                    dataset['net_to_gross'] = 1 - dataset['facies_5_pct'] - dataset['facies_6_pct']
                
                if 'phi_mean' in dataset.columns and 'depth_range' in dataset.columns:
                    dataset['storage_capacity'] = dataset['phi_mean'] * dataset['depth_range']
                
                if 'perm_mean' in dataset.columns and 'GR_mean' in dataset.columns:
                    dataset['flow_quality'] = np.log1p(dataset['perm_mean']) / (dataset['GR_mean'] + 1)
                
                # INDUSTRY-STANDARD FEATURES (SPE Literature)
                # RQI - Reservoir Quality Index (Amaefule et al. 1993)
                # RQI = 0.0314 × sqrt(k/φ) - relates perm/porosity to flow capacity
                if 'phi_mean' in dataset.columns and 'perm_mean' in dataset.columns:
                    phi_safe = dataset['phi_mean'].replace(0, 0.001)  # Avoid division by zero
                    dataset['RQI'] = 0.0314 * np.sqrt(dataset['perm_mean'] / phi_safe)
                    
                    # FZI - Flow Zone Indicator (hydraulic flow unit classification)
                    # FZI = RQI / (φ / (1-φ))
                    phi_z = phi_safe / (1 - phi_safe)
                    dataset['FZI'] = dataset['RQI'] / phi_z
                
                # Vp/Vs ratio - Lithology and fluid indicator (rock physics)
                if 'Vp_mean' in dataset.columns and 'Vs_mean' in dataset.columns:
                    vs_safe = dataset['Vs_mean'].replace(0, 1)  # Avoid division by zero
                    dataset['Vp_Vs_ratio'] = dataset['Vp_mean'] / vs_safe
            
            # NATALY'S INSIGHT: Analog Well Similarity Feature
            # Correlate wells to known good producers based on rock quality
            st.info("Creating Analog Well Similarity feature (Nataly's insight)...")
            
            # Define rock quality features for similarity calculation
            rq_cols = ['phi_mean', 'perm_mean', 'GR_mean', 'sand_proportion']
            available_rq_cols = [c for c in rq_cols if c in df.columns]
            
            if len(available_rq_cols) >= 2 and 'Target_3yr_Oil_BBL' in df.columns:
                from sklearn.preprocessing import StandardScaler
                from scipy.spatial.distance import cdist
                
                # Identify "good producers" - top 25% by production with good rock
                prod_threshold = df['Target_3yr_Oil_BBL'].quantile(0.75)
                good_producers = df[df['Target_3yr_Oil_BBL'] >= prod_threshold].copy()
                
                # Normalize rock quality features for distance calculation
                scaler = StandardScaler()
                train_rq_scaled = scaler.fit_transform(df[available_rq_cols].fillna(0))
                test_rq_scaled = scaler.transform(test_df[available_rq_cols].fillna(0))
                good_rq_scaled = scaler.transform(good_producers[available_rq_cols].fillna(0))
                
                # Calculate similarity to good producers (inverse of distance)
                # For training wells
                train_distances = cdist(train_rq_scaled, good_rq_scaled, metric='euclidean')
                df['min_dist_to_good_producer'] = train_distances.min(axis=1)
                df['avg_dist_to_good_producer'] = train_distances.mean(axis=1)
                df['analog_similarity'] = 1 / (1 + df['min_dist_to_good_producer'])
                
                # Weighted average production of similar wells
                weights = 1 / (train_distances + 0.1)  # Avoid division by zero
                weights = weights / weights.sum(axis=1, keepdims=True)
                df['analog_production_proxy'] = (weights * good_producers['Target_3yr_Oil_BBL'].values).sum(axis=1)
                
                # For test wells
                test_distances = cdist(test_rq_scaled, good_rq_scaled, metric='euclidean')
                test_df['min_dist_to_good_producer'] = test_distances.min(axis=1)
                test_df['avg_dist_to_good_producer'] = test_distances.mean(axis=1)
                test_df['analog_similarity'] = 1 / (1 + test_df['min_dist_to_good_producer'])
                test_df['analog_production_proxy'] = (
                    (1 / (test_distances + 0.1)) / (1 / (test_distances + 0.1)).sum(axis=1, keepdims=True) 
                    * good_producers['Target_3yr_Oil_BBL'].values
                ).sum(axis=1)
                
                st.success(f"Analog features created! Found {len(good_producers)} good producer wells (top 25%)")
            
            # SPATIAL PROXIMITY TO HIGH-PRODUCTION REGIONS
            # User observation: Left side (low X) has higher production, especially corners
            st.info("Creating spatial proximity features...")
            
            if 'X' in df.columns and 'Y' in df.columns and 'Target_3yr_Oil_BBL' in df.columns:
                # Identify high-production region centroids from training data
                high_prod_wells = df[df['Target_3yr_Oil_BBL'] >= df['Target_3yr_Oil_BBL'].quantile(0.75)]
                
                # Calculate centroid of high-production region
                high_prod_centroid_x = high_prod_wells['X'].mean()
                high_prod_centroid_y = high_prod_wells['Y'].mean()
                
                # Distance to high-production centroid (lower = closer to good area)
                for dataset in [df, test_df]:
                    dataset['dist_to_high_prod_region'] = np.sqrt(
                        (dataset['X'] - high_prod_centroid_x)**2 + 
                        (dataset['Y'] - high_prod_centroid_y)**2
                    )
                    # Inverse: proximity score (higher = closer to good area)
                    dataset['proximity_to_high_prod'] = 1 / (1 + dataset['dist_to_high_prod_region'])
                    
                    # Distance to left edge (low X = high production area)
                    dataset['dist_from_left'] = dataset['X'] - df['X'].min()
                    dataset['left_region_score'] = 1 / (1 + dataset['dist_from_left'])
                    
                    # Inverse distance weighted production proxy (spatial)
                    if 'Target_3yr_Oil_BBL' in df.columns:
                        from scipy.spatial.distance import cdist
                        train_coords = df[['X', 'Y']].values
                        test_coords = dataset[['X', 'Y']].values
                        spatial_dists = cdist(test_coords, train_coords, metric='euclidean')
                        spatial_weights = 1 / (spatial_dists + 1)
                        spatial_weights = spatial_weights / spatial_weights.sum(axis=1, keepdims=True)
                        dataset['spatial_production_proxy'] = (spatial_weights * df['Target_3yr_Oil_BBL'].values).sum(axis=1)
                
                st.success(f"Spatial proximity features created! High-prod centroid at X={high_prod_centroid_x:.1f}, Y={high_prod_centroid_y:.1f}")
            
            st.session_state.train_df = df
            st.session_state.test_df = test_df
            st.success("Rock quality features created!")
            
            st.subheader("New Features Preview")
            new_cols = ['Well_ID', 'phi_perm_product', 'rock_quality', 'RQI', 'FZI', 'Vp_Vs_ratio', 'impedance_ratio', 'net_to_gross', 'storage_capacity', 'flow_quality', 'analog_similarity', 'analog_production_proxy']
            available_cols = [c for c in new_cols if c in df.columns]
            st.dataframe(df[available_cols].head(10), use_container_width=True)
            
            st.subheader("Feature Correlations with Production")
            st.markdown("**Industry-Standard Features (SPE Literature):**")
            for col in ['RQI', 'FZI', 'Vp_Vs_ratio']:
                if col in df.columns:
                    corr = df[col].corr(df['Target_3yr_Oil_BBL'])
                    direction = "↑" if corr > 0 else "↓"
                    st.markdown(f"- **{col}**: {corr:.3f} {direction}")
            
            st.markdown("**Rock Quality Features:**")
            for col in ['phi_perm_product', 'rock_quality', 'net_to_gross', 'storage_capacity', 'flow_quality', 'analog_similarity', 'analog_production_proxy']:
                if col in df.columns:
                    corr = df[col].corr(df['Target_3yr_Oil_BBL'])
                    direction = "↑" if corr > 0 else "↓"
                    st.markdown(f"- **{col}**: {corr:.3f} {direction}")

elif page == "5. Model Training":
    st.header("Step 5: Model Training")
    
    if st.session_state.train_df is None:
        st.warning("Please load data in Step 1 first.")
    else:
        train_df = st.session_state.train_df.copy()
        test_df = st.session_state.test_df.copy()
        
        st.subheader("Experiment Configuration (Dr. Pyrcz's Advice)")
        
        st.info("**Dr. Pyrcz's Recommendations:** Normalize everything, try simplest things first (Linear), look at highs/lows.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            model_type = st.selectbox(
                "Model Type",
                ["Random Forest", "XGBoost", "Linear Regression", "Ridge Regression", "Elastic Net"],
                help="Try Linear first as baseline (Dr. Pyrcz's advice)"
            )
        
        with col2:
            normalize_features = st.checkbox(
                "Normalize Features (StandardScaler)",
                value=True,
                help="Recommended by Dr. Pyrcz - equalizes feature scales"
            )
        
        with col3:
            sand_map_option = st.selectbox(
                "Sand Map Handling",
                ["Include", "Exclude", "Smooth (3x3)", "Smooth (5x5)"],
                help="Dinghan Wang: sand map has deliberate noise"
            )
        
        st.divider()
        
        st.subheader("Model Configuration")
        
        if model_type == "Random Forest":
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
                    min_samples_split = st.slider("Min Samples Split", 2, 20, 5)
                    max_features_option = st.selectbox("Max Features", ["sqrt", "log2", "0.5", "0.75", "1.0"], index=0, help="Number of features to consider for best split")
                with col2:
                    min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 1)
                    ccp_alpha = st.slider("CCP Alpha (Pruning)", 0.0, 0.05, 0.0, 0.005)
                    min_impurity_decrease = st.slider("Min Impurity Decrease", 0.0, 0.1, 0.0, 0.01)
                cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
                use_oob = st.checkbox("Use Out-of-Bag (OOB) Score", value=True, help="Evaluate model on OOB samples for additional insights")
            else:
                with col1:
                    n_trials = st.slider("Optuna Trials", 10, 100, 30, 10)
                    cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
                    use_oob = st.checkbox("Use Out-of-Bag (OOB) Score", value=True)
                with col2:
                    st.info("Optuna will search:\n- n_estimators: 50-300\n- max_depth: 3-20\n- min_samples_split: 2-20\n- min_samples_leaf: 1-10\n- max_features: sqrt, log2, 0.5\n- ccp_alpha: 0.0-0.05")
        
        elif model_type == "XGBoost":
            tuning_mode = st.radio(
                "Hyperparameter Tuning Mode:",
                ["Manual", "Optuna (Auto-Tune)"],
                help="Optuna automatically finds optimal hyperparameters"
            )
            
            col1, col2 = st.columns(2)
            
            if tuning_mode == "Manual":
                with col1:
                    xgb_n_estimators = st.slider("Number of Trees", 50, 500, 100, 50)
                    xgb_max_depth = st.slider("Max Depth", 3, 15, 6)
                    xgb_learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01)
                with col2:
                    xgb_subsample = st.slider("Subsample", 0.5, 1.0, 0.8, 0.1)
                    xgb_colsample = st.slider("Column Sample", 0.5, 1.0, 0.8, 0.1)
                    cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
            else:
                with col1:
                    n_trials = st.slider("Optuna Trials", 10, 100, 30, 10)
                    cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
                with col2:
                    st.info("Optuna will search:\n- n_estimators: 50-500\n- max_depth: 3-15\n- learning_rate: 0.01-0.3\n- subsample: 0.5-1.0\n- colsample: 0.5-1.0")
        
        elif model_type == "Elastic Net":
            tuning_mode = "Manual"
            cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
            col1, col2 = st.columns(2)
            with col1:
                elastic_alpha = st.slider("Alpha (Regularization Strength)", 0.01, 100.0, 1.0, help="Higher = more regularization")
            with col2:
                l1_ratio = st.slider("L1 Ratio", 0.0, 1.0, 0.5, 0.1, help="0 = Ridge only, 1 = Lasso only, 0.5 = balanced")
            st.info(f"L1 Ratio: {l1_ratio:.1f} → {int(l1_ratio*100)}% Lasso (feature selection) + {int((1-l1_ratio)*100)}% Ridge (shrinkage)")
        
        else:
            tuning_mode = "Manual"
            cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
            if model_type == "Ridge Regression":
                alpha = st.slider("Ridge Alpha (Regularization)", 0.01, 100.0, 1.0)
        
        exclude_cols = ['Well_ID', 'Target_3yr_Oil_BBL']
        
        if sand_map_option == "Exclude":
            exclude_cols.append('sand_proportion')
        elif sand_map_option == "Smooth (3x3)" or sand_map_option == "Smooth (5x5)":
            kernel_size = 5 if sand_map_option == "Smooth (5x5)" else 3
            sand_map = st.session_state.sand_map
            from scipy.ndimage import uniform_filter
            smoothed_map = uniform_filter(sand_map, size=kernel_size)
            st.session_state.sand_map = smoothed_map
            all_x = pd.concat([train_df['X'], test_df['X']])
            all_y = pd.concat([train_df['Y'], test_df['Y']])
            x_min, x_max = all_x.min(), all_x.max()
            y_min, y_max = all_y.min(), all_y.max()
            train_df['sand_proportion'] = lookup_sand_proportion(train_df, smoothed_map, x_min, x_max, y_min, y_max)
            test_df['sand_proportion'] = lookup_sand_proportion(test_df, smoothed_map, x_min, x_max, y_min, y_max)
        
        feature_cols = [c for c in train_df.columns if c not in exclude_cols and train_df[c].dtype in ['float64', 'int64']]
        
        st.markdown(f"**Features available:** {len(feature_cols)}")
        
        st.divider()
        
        with st.expander("🔬 Stepwise Feature Selection (Optional)", expanded=False):
            st.markdown("""
            **Stepwise regression** automatically selects the optimal subset of features by iteratively adding/removing 
            variables based on cross-validation score. This can reduce overfitting and improve interpretability.
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                stepwise_direction = st.selectbox(
                    "Selection Direction",
                    ["Forward", "Backward"],
                    help="Forward: start empty, add best features. Backward: start full, remove worst features."
                )
            with col2:
                stepwise_k_features = st.slider(
                    "Target Features",
                    min_value=3, max_value=min(30, len(feature_cols)), value=min(10, len(feature_cols)),
                    help="Number of features to select"
                )
            
            stepwise_cv = st.slider("CV Folds for Feature Selection", 3, 10, 5)
            
            if st.button("Run Stepwise Feature Selection", type="secondary"):
                with st.spinner(f"Running {stepwise_direction} stepwise selection..."):
                    try:
                        from mlxtend.feature_selection import SequentialFeatureSelector
                        from sklearn.linear_model import Ridge
                        from sklearn.preprocessing import StandardScaler
                        
                        X_step = train_df[feature_cols].fillna(0)
                        y_step = train_df['Target_3yr_Oil_BBL']
                        
                        scaler = StandardScaler()
                        X_step_scaled = pd.DataFrame(scaler.fit_transform(X_step), columns=X_step.columns)
                        
                        base_model = Ridge(alpha=1.0, random_state=42)
                        
                        sfs = SequentialFeatureSelector(
                            base_model,
                            k_features=stepwise_k_features,
                            forward=(stepwise_direction == "Forward"),
                            floating=False,
                            scoring='r2',
                            cv=stepwise_cv,
                            n_jobs=-1,
                            verbose=0
                        )
                        
                        sfs.fit(X_step_scaled, y_step)
                        
                        selected_features = list(sfs.k_feature_names_)
                        st.session_state.selected_features = selected_features
                        
                        st.success(f"Selected {len(selected_features)} features with R² = {sfs.k_score_:.4f}")
                        
                        st.markdown("**Selected Features:**")
                        for i, feat in enumerate(selected_features, 1):
                            st.write(f"{i}. {feat}")
                        
                        st.info("These features will be used when you click 'Train Model' below.")
                        
                    except Exception as e:
                        st.error(f"Stepwise selection failed: {str(e)}")
            
            if 'selected_features' in st.session_state and st.session_state.selected_features:
                use_stepwise_features = st.checkbox(
                    f"Use stepwise-selected features ({len(st.session_state.selected_features)} features)",
                    value=True
                )
                if use_stepwise_features:
                    feature_cols = st.session_state.selected_features
                    st.markdown(f"**Using {len(feature_cols)} stepwise-selected features**")
        
        st.divider()
        
        experiment_name = st.text_input("Experiment Name (for output file)", value=f"{model_type.replace(' ', '_')}_norm{normalize_features}_sand{sand_map_option}")
        
        if st.button("Train Model", type="primary"):
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
            from sklearn.model_selection import cross_val_predict
            
            X = train_df[feature_cols].fillna(0)
            y = train_df['Target_3yr_Oil_BBL']
            
            st.session_state.feature_cols = feature_cols
            
            scaler = None
            if normalize_features:
                scaler = StandardScaler()
                X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
                X = X_scaled
                st.success("Features normalized with StandardScaler")
            
            st.session_state.scaler = scaler
            st.session_state.normalize_features = normalize_features
            st.session_state.experiment_name = experiment_name
            
            if model_type == "Random Forest":
                if tuning_mode == "Optuna (Auto-Tune)":
                    import optuna
                    optuna.logging.set_verbosity(optuna.logging.WARNING)
                    
                    with st.spinner(f"Optuna tuning model ({n_trials} trials)..."):
                        def objective(trial):
                            max_feat = trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5])
                            params = {
                                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                                'max_depth': trial.suggest_int('max_depth', 3, 20),
                                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                                'max_features': max_feat,
                                'ccp_alpha': trial.suggest_float('ccp_alpha', 0.0, 0.05),
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
                        best_params['oob_score'] = use_oob
                        
                        st.success(f"Best: trees={best_params['n_estimators']}, depth={best_params['max_depth']}, max_features={best_params['max_features']}, leaf={best_params['min_samples_leaf']}")
                        
                        model = RandomForestRegressor(**best_params)
                else:
                    max_feat = float(max_features_option) if max_features_option in ['0.5', '0.75', '1.0'] else max_features_option
                    model = RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_feat,
                        ccp_alpha=ccp_alpha,
                        min_impurity_decrease=min_impurity_decrease,
                        oob_score=use_oob,
                        random_state=42,
                        n_jobs=-1
                    )
            elif model_type == "XGBoost":
                import xgboost as xgb
                
                if tuning_mode == "Optuna (Auto-Tune)":
                    import optuna
                    optuna.logging.set_verbosity(optuna.logging.WARNING)
                    
                    with st.spinner(f"Optuna tuning XGBoost ({n_trials} trials)..."):
                        def objective(trial):
                            params = {
                                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                                'max_depth': trial.suggest_int('max_depth', 3, 15),
                                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                                'random_state': 42,
                                'n_jobs': -1
                            }
                            model = xgb.XGBRegressor(**params)
                            scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
                            return scores.mean()
                        
                        study = optuna.create_study(direction='maximize')
                        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
                        
                        best_params = study.best_params
                        best_params['random_state'] = 42
                        best_params['n_jobs'] = -1
                        
                        st.success(f"Best params: lr={best_params['learning_rate']:.3f}, depth={best_params['max_depth']}, trees={best_params['n_estimators']}")
                        
                        model = xgb.XGBRegressor(**best_params)
                else:
                    model = xgb.XGBRegressor(
                        n_estimators=xgb_n_estimators,
                        max_depth=xgb_max_depth,
                        learning_rate=xgb_learning_rate,
                        subsample=xgb_subsample,
                        colsample_bytree=xgb_colsample,
                        random_state=42,
                        n_jobs=-1
                    )
                    st.info(f"Using XGBoost: lr={xgb_learning_rate}, depth={xgb_max_depth}, trees={xgb_n_estimators}")
            
            elif model_type == "Linear Regression":
                model = LinearRegression()
                st.info("Using Linear Regression (Dr. Pyrcz's baseline recommendation)")
            elif model_type == "Elastic Net":
                model = ElasticNet(alpha=elastic_alpha, l1_ratio=l1_ratio, random_state=42, max_iter=10000)
                st.info(f"Using Elastic Net with alpha={elastic_alpha}, L1 ratio={l1_ratio}")
            else:
                model = Ridge(alpha=alpha, random_state=42)
                st.info(f"Using Ridge Regression with alpha={alpha}")
            
            st.session_state.model_type = model_type
            
            with st.spinner("Training model..."):
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import mean_absolute_error, mean_squared_error
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
                
                y_pred_cv = cross_val_predict(model, X, y, cv=cv_folds)
                cv_residuals = y - y_pred_cv
                
                if model_type == "Random Forest":
                    model_for_split = RandomForestRegressor(
                        n_estimators=model.n_estimators, max_depth=model.max_depth,
                        min_samples_split=model.min_samples_split, min_samples_leaf=model.min_samples_leaf,
                        max_features=model.max_features, random_state=42, n_jobs=-1
                    )
                elif model_type == "XGBoost":
                    model_for_split = xgb.XGBRegressor(
                        n_estimators=model.n_estimators, max_depth=model.max_depth,
                        learning_rate=model.learning_rate, random_state=42, n_jobs=-1
                    )
                else:
                    model_for_split = model.__class__(**model.get_params())
                
                model_for_split.fit(X_train, y_train)
                y_pred_train = model_for_split.predict(X_train)
                y_pred_test = model_for_split.predict(X_test)
                
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                train_mae = mean_absolute_error(y_train, y_pred_train)
                test_mae = mean_absolute_error(y_test, y_pred_test)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                
                model.fit(X, y)
                y_pred = model.predict(X)
                residuals = cv_residuals
                
                st.session_state.model = model
                st.session_state.residuals = residuals.values
            
            st.success(f"{model_type} trained successfully!")
            
            st.subheader("📊 Train/Test Split Evaluation (80/20 Split)")
            st.markdown("*Model trained on 80% of data, evaluated on held-out 20%*")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Training Set Metrics:**")
                st.metric("Train R²", f"{train_r2:.4f}")
                st.metric("Train MAE", f"{train_mae:,.0f} BBL")
                st.metric("Train RMSE", f"{train_rmse:,.0f} BBL")
            
            with col2:
                st.markdown("**Test Set Metrics (Held-Out):**")
                delta_val = f"{test_r2 - train_r2:.4f}" if test_r2 < train_r2 else None
                st.metric("Test R²", f"{test_r2:.4f}", delta=delta_val)
                st.metric("Test MAE", f"{test_mae:,.0f} BBL")
                st.metric("Test RMSE", f"{test_rmse:,.0f} BBL")
            
            if train_r2 - test_r2 > 0.15:
                st.warning("⚠️ Large gap between Train and Test R² may indicate overfitting. Consider simpler model or more regularization.")
            elif test_r2 > 0.3:
                st.success("✅ Model generalizes well to unseen data!")
            
            st.divider()
            st.subheader("📈 Cross-Validation & Full Model Metrics")
            
            mae = mean_absolute_error(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("CV R² Mean", f"{cv_scores.mean():.4f}")
            col2.metric("CV R² Std", f"±{cv_scores.std():.4f}")
            col3.metric("Full Data MAE", f"{mae:,.0f} BBL")
            col4.metric("Full Data RMSE", f"{rmse:,.0f} BBL")
            
            if hasattr(model, 'oob_score_') and model.oob_score:
                st.metric("OOB R² Score", f"{model.oob_score_:.4f}")
                st.info("OOB (Out-of-Bag) score provides an unbiased estimate of model performance using samples not used in each tree's training.")
            
            if hasattr(model, 'feature_importances_'):
                st.subheader("Feature Importance")
                importance_df = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False).head(20)
                
                fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                            title="Top 20 Feature Importances (Built-in)")
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("SHAP Feature Importance Analysis")
                st.markdown("*SHAP values quantify each feature's contribution to predictions (Lundberg & Lee 2017)*")
                
                with st.spinner("Computing SHAP values..."):
                    try:
                        import shap
                        import matplotlib.pyplot as plt
                        
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X)
                        
                        # SHAP Bar Plot - SHAP creates its own figure, so use gcf()
                        plt.figure(figsize=(10, 8))
                        shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=20)
                        st.pyplot(plt.gcf())
                        plt.close('all')
                        
                        st.markdown("**SHAP Summary Plot (Beeswarm):**")
                        plt.figure(figsize=(10, 8))
                        shap.summary_plot(shap_values, X, show=False, max_display=15)
                        st.pyplot(plt.gcf())
                        plt.close('all')
                        
                        st.success("SHAP analysis complete! This shows which geological/petrophysical factors drive production predictions.")
                    except Exception as e:
                        st.warning(f"SHAP analysis skipped: {str(e)}")
            
            elif hasattr(model, 'coef_'):
                st.subheader("Feature Coefficients (Linear Model)")
                coef_df = pd.DataFrame({
                    'Feature': feature_cols,
                    'Coefficient': model.coef_
                }).sort_values('Coefficient', key=abs, ascending=False).head(20)
                
                fig = px.bar(coef_df, x='Coefficient', y='Feature', orientation='h',
                            title="Top 20 Feature Coefficients (Absolute)")
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
            
            st.divider()
            st.subheader("🎯 Uncertainty Calibration Check")
            st.markdown("""
            *Tests if the model's uncertainty (R1-R100) is reliable by holding out some training wells,
            predicting them with uncertainty, and checking if actual values fall within the predicted range.*
            """)
            
            if st.button("Run Calibration Check", type="secondary"):
                with st.spinner("Running calibration check (hold-out validation)..."):
                    from sklearn.model_selection import KFold
                    
                    train_df = st.session_state.train_df
                    n_folds = 5
                    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
                    
                    coverage_90 = []
                    coverage_50 = []
                    interval_widths = []
                    all_results = []
                    
                    X_full = train_df[feature_cols].fillna(0)
                    y_full = train_df['Target_3yr_Oil_BBL']
                    
                    if normalize_features:
                        X_full_scaled = pd.DataFrame(scaler.transform(X_full), columns=X_full.columns, index=X_full.index)
                    else:
                        X_full_scaled = X_full
                    
                    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_full_scaled)):
                        X_fold_train = X_full_scaled.iloc[train_idx]
                        X_fold_test = X_full_scaled.iloc[test_idx]
                        y_fold_train = y_full.iloc[train_idx]
                        y_fold_test = y_full.iloc[test_idx]
                        
                        if model_type == "Random Forest":
                            fold_model = RandomForestRegressor(
                                n_estimators=model.n_estimators, max_depth=model.max_depth,
                                min_samples_split=model.min_samples_split, min_samples_leaf=model.min_samples_leaf,
                                max_features=model.max_features, random_state=42, n_jobs=-1
                            )
                        elif model_type == "XGBoost":
                            fold_model = xgb.XGBRegressor(
                                n_estimators=model.n_estimators, max_depth=model.max_depth,
                                learning_rate=model.learning_rate, random_state=42, n_jobs=-1
                            )
                        else:
                            fold_model = model.__class__(**model.get_params())
                        
                        fold_model.fit(X_fold_train, y_fold_train)
                        
                        fold_preds = fold_model.predict(X_fold_train)
                        fold_residuals = y_fold_train.values - fold_preds
                        
                        point_preds = fold_model.predict(X_fold_test)
                        
                        n_realizations = 100
                        for i, (idx, actual) in enumerate(zip(test_idx, y_fold_test)):
                            realizations = point_preds[i] + np.random.choice(fold_residuals, size=n_realizations, replace=True)
                            realizations = np.maximum(realizations, 0)
                            
                            p5 = np.percentile(realizations, 5)
                            p95 = np.percentile(realizations, 95)
                            p25 = np.percentile(realizations, 25)
                            p75 = np.percentile(realizations, 75)
                            
                            in_90 = 1 if p5 <= actual <= p95 else 0
                            in_50 = 1 if p25 <= actual <= p75 else 0
                            
                            coverage_90.append(in_90)
                            coverage_50.append(in_50)
                            interval_widths.append(p95 - p5)
                            
                            all_results.append({
                                'Well_ID': train_df.iloc[idx]['Well_ID'],
                                'Actual': actual,
                                'Predicted': point_preds[i],
                                'P5': p5,
                                'P95': p95,
                                'In_90_Range': 'Yes' if in_90 else 'No'
                            })
                    
                    actual_coverage_90 = np.mean(coverage_90) * 100
                    actual_coverage_50 = np.mean(coverage_50) * 100
                    avg_width = np.mean(interval_widths)
                    
                    st.markdown("### Calibration Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        delta_90 = actual_coverage_90 - 90
                        color = "normal" if abs(delta_90) < 10 else "off"
                        st.metric("90% Interval Coverage", f"{actual_coverage_90:.1f}%", 
                                  delta=f"{delta_90:+.1f}% vs target 90%",
                                  delta_color="normal" if abs(delta_90) < 10 else "inverse")
                    
                    with col2:
                        delta_50 = actual_coverage_50 - 50
                        st.metric("50% Interval Coverage", f"{actual_coverage_50:.1f}%",
                                  delta=f"{delta_50:+.1f}% vs target 50%",
                                  delta_color="normal" if abs(delta_50) < 15 else "inverse")
                    
                    with col3:
                        st.metric("Avg 90% Interval Width", f"{avg_width/1e6:.1f}M BBL")
                    
                    if actual_coverage_90 >= 85 and actual_coverage_90 <= 95:
                        st.success("✅ **Well-Calibrated!** The 90% prediction intervals contain ~90% of actual values. Uncertainty estimates are reliable.")
                    elif actual_coverage_90 < 85:
                        st.warning(f"⚠️ **Under-Coverage ({actual_coverage_90:.0f}%):** Model is over-confident. Actual values fall outside predicted ranges too often. Consider widening uncertainty or using a different model.")
                    else:
                        st.info(f"📊 **Over-Coverage ({actual_coverage_90:.0f}%):** Model is under-confident. Prediction intervals are wider than necessary, but this is conservative (safe).")
                    
                    results_df = pd.DataFrame(all_results)
                    
                    st.markdown("### Individual Well Calibration")
                    
                    fig = go.Figure()
                    
                    for i, row in results_df.iterrows():
                        color = 'green' if row['In_90_Range'] == 'Yes' else 'red'
                        fig.add_trace(go.Scatter(
                            x=[row['P5'], row['P95']], y=[i, i],
                            mode='lines', line=dict(color=color, width=8),
                            showlegend=False, hoverinfo='skip'
                        ))
                        fig.add_trace(go.Scatter(
                            x=[row['Actual']], y=[i],
                            mode='markers', marker=dict(color='black', size=10, symbol='diamond'),
                            showlegend=False,
                            hovertemplate=f"Well {row['Well_ID']}<br>Actual: {row['Actual']/1e6:.1f}M<br>Range: {row['P5']/1e6:.1f}M - {row['P95']/1e6:.1f}M"
                        ))
                    
                    fig.update_layout(
                        title="Calibration Plot: Actual (◆) vs 90% Prediction Interval",
                        xaxis_title="Oil Production (BBL)",
                        yaxis_title="Well Index",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.caption("**Green bars:** Actual value within 90% range (good). **Red bars:** Actual value outside range (model was wrong).")

elif page == "6. Generate Solution":
    st.header("Step 6: Generate Solution File")
    
    if st.session_state.model is None:
        st.warning("Please train a model in Step 5 first.")
    else:
        model = st.session_state.model
        test_df = st.session_state.test_df
        residuals = st.session_state.residuals
        feature_cols = st.session_state.feature_cols
        scaler = st.session_state.get('scaler', None)
        normalize_features = st.session_state.get('normalize_features', False)
        experiment_name = st.session_state.get('experiment_name', 'experiment')
        model_type = st.session_state.get('model_type', 'Random Forest')
        
        st.info(f"**Current Experiment:** {experiment_name} | **Model:** {model_type} | **Normalized:** {normalize_features}")
        
        st.markdown("""
        **Solution Requirements:**
        - 12 rows (Wells 72-83)
        - Columns: Well_ID, Prediction_BBL, R1-R100
        - Replace all -9999 values with predictions
        """)
        
        n_realizations = st.slider("Number of Realizations", 10, 100, 100)
        
        col1, col2 = st.columns(2)
        with col1:
            save_as_main = st.checkbox("Save as main solution.csv", value=True, help="Overwrite the main submission file")
        with col2:
            save_experiment = st.checkbox("Also save as experiment file", value=True, help="Save with experiment name for comparison")
        
        if st.button("Generate Predictions", type="primary"):
            test_df_sorted = test_df.sort_values('Well_ID').reset_index(drop=True)
            X_test = test_df_sorted[feature_cols].fillna(0)
            
            if normalize_features and scaler is not None:
                X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
                st.success("Test features normalized using training scaler")
            
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
            
            files_saved = []
            if save_as_main:
                solution.to_csv(f"{OUTPUT_DIR}/solution.csv", index=False)
                solution.to_csv(f"{DATA_DIR}/solution.csv", index=False)
                files_saved.append("solution.csv")
            
            if save_experiment:
                exp_filename = f"solution_{experiment_name}.csv"
                solution.to_csv(f"{OUTPUT_DIR}/{exp_filename}", index=False)
                files_saved.append(exp_filename)
            
            st.success(f"Solution generated! Saved: {', '.join(files_saved)}")
            
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
            
            col1, col2 = st.columns(2)
            with col1:
                csv_data = solution.to_csv(index=False)
                st.download_button(
                    label="Download solution.csv",
                    data=csv_data,
                    file_name="solution.csv",
                    mime="text/csv"
                )
            with col2:
                st.download_button(
                    label=f"Download {experiment_name}.csv",
                    data=csv_data,
                    file_name=f"solution_{experiment_name}.csv",
                    mime="text/csv"
                )
        
        st.divider()
        st.subheader("Compare Experiments")
        
        experiment_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('solution_') and f.endswith('.csv')]
        if experiment_files:
            st.markdown(f"**Available experiment files:** {len(experiment_files)}")
            for f in experiment_files:
                exp_df = pd.read_csv(f"{OUTPUT_DIR}/{f}")
                mean_pred = exp_df['Prediction_BBL'].mean()
                st.markdown(f"- **{f}**: Mean = {mean_pred:,.0f} BBL")
        else:
            st.info("No experiment files yet. Run multiple experiments to compare!")

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

elif page == "8. Scholarly Analysis":
    st.header("Scholarly Analysis: Research Backing for Our Workflow")
    
    st.markdown("""
    This document provides peer-reviewed justification for every methodological decision in our ML pipeline.
    """)
    
    scholarly_file = "outputs/WORKFLOW_SCHOLARLY_ANALYSIS.md"
    
    try:
        with open(scholarly_file, 'r') as f:
            scholarly_content = f.read()
        
        import markdown
        
        html_template = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Scholarly Analysis - Brain Oil Team</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 900px;
            margin: 40px auto;
            padding: 20px 40px;
            line-height: 1.6;
            color: #333;
        }}
        h1 {{ color: #1a5f7a; border-bottom: 3px solid #1a5f7a; padding-bottom: 10px; }}
        h2 {{ color: #2c3e50; margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
        h3 {{ color: #34495e; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #1a5f7a; color: white; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        blockquote {{ 
            border-left: 4px solid #1a5f7a; 
            margin: 20px 0; 
            padding: 10px 20px; 
            background: #f5f5f5;
            font-style: italic;
        }}
        code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
        hr {{ border: none; border-top: 2px solid #eee; margin: 30px 0; }}
        .header {{ text-align: center; margin-bottom: 40px; }}
        .footer {{ text-align: center; margin-top: 40px; color: #777; font-size: 0.9em; }}
        @media print {{
            body {{ margin: 0; padding: 20px; }}
            h1, h2 {{ page-break-after: avoid; }}
            table {{ page-break-inside: avoid; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Energy AI Hackathon 2026</h1>
        <p><strong>Team Brain Oil</strong> - Scholarly Analysis</p>
    </div>
    {markdown.markdown(scholarly_content, extensions=['tables', 'fenced_code'])}
    <div class="footer">
        <hr>
        <p>Generated by Brain Oil ML Pipeline | Energy AI Hackathon 2026</p>
    </div>
</body>
</html>"""
        
        st.info("**To save as PDF:** Download the HTML file below, open it in your browser, then press **Ctrl+P** (or Cmd+P on Mac) and select 'Save as PDF'.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download HTML (Open & Print as PDF)",
                data=html_template,
                file_name="WORKFLOW_SCHOLARLY_ANALYSIS.html",
                mime="text/html",
                type="primary"
            )
        with col2:
            st.download_button(
                label="Download Markdown",
                data=scholarly_content,
                file_name="WORKFLOW_SCHOLARLY_ANALYSIS.md",
                mime="text/markdown"
            )
        
        st.divider()
        
        with st.expander("View Full Scholarly Analysis", expanded=True):
            st.markdown(scholarly_content)
        
        st.divider()
        
        st.subheader("Quick Reference: Key Citations")
        
        citations = {
            "MICE Imputation": "Van Buuren (2018); Hallam et al. (2022); SPE 218890 (Abdulkhaleq 2024)",
            "CART Estimator": "SPE 218890: 'MICE + CART outperformed other methods'",
            "Random Forest": "Al shaba'an & Nemer (2024): 99% accuracy in oil production prediction",
            "Optuna (TPE)": "Akiba et al. (2019, KDD Best Paper)",
            "Residual Bootstrap": "Pan & Politis (2014); Palmer et al. (2022)",
            "Feature Engineering": "Amaefule et al. (1993); Cao et al. (2025)"
        }
        
        for technique, citation in citations.items():
            st.markdown(f"**{technique}:** {citation}")
            
    except FileNotFoundError:
        st.error(f"Scholarly analysis file not found at {scholarly_file}. Please run the workflow first.")
