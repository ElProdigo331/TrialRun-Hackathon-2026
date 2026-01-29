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
    page_title="Energy AI Hackathon Workflow",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Energy AI Hackathon 2026 Workflow")
st.markdown("**Complete ML Pipeline for Predicting Energy Usage in Hydraulic Fracturing Operations**")

DATA_DIR = "data"
OUTPUT_DIR = "outputs"

@st.cache_data
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    train_df.columns = train_df.columns.str.strip()
    test_df.columns = test_df.columns.str.strip()
    return train_df, test_df

def get_feature_columns(df):
    exclude_cols = ['Well Name', 'Grid', 'Diesel', 'CNG', 'Fuel Type']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    return feature_cols

sidebar = st.sidebar
sidebar.header("Navigation")

pages = [
    "1. Data Upload & Inspection",
    "2. Data Cleaning & Imputation", 
    "3. Exploratory Data Analysis",
    "4. Feature Engineering",
    "5. Model Training",
    "6. Uncertainty Quantification",
    "7. Generate Predictions",
    "8. Hackathon Quick Start",
    "9. AI ML Assistant"
]

page = sidebar.radio("Select Step:", pages)

if 'train_df' not in st.session_state:
    st.session_state.train_df = None
if 'test_df' not in st.session_state:
    st.session_state.test_df = None
if 'cleaned_train' not in st.session_state:
    st.session_state.cleaned_train = None
if 'cleaned_test' not in st.session_state:
    st.session_state.cleaned_test = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'residuals' not in st.session_state:
    st.session_state.residuals = {}
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

if page == "1. Data Upload & Inspection":
    st.header("Step 1: Data Upload & Initial Inspection")
    
    st.subheader("Upload Data Files")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Training Data**")
        train_file = st.file_uploader("Upload training CSV", type=['csv'], key='train')
        if os.path.exists(f"{DATA_DIR}/HackathonData2025.csv"):
            use_default_train = st.checkbox("Use 2025 training data", value=True)
        else:
            use_default_train = False
            
    with col2:
        st.markdown("**Test Data**")
        test_file = st.file_uploader("Upload test CSV", type=['csv'], key='test')
        if os.path.exists(f"{DATA_DIR}/testing.csv"):
            use_default_test = st.checkbox("Use 2025 test data", value=True)
        else:
            use_default_test = False
    
    if st.button("Load Data", type="primary"):
        if use_default_train and use_default_test:
            train_path = f"{DATA_DIR}/HackathonData2025.csv"
            test_path = f"{DATA_DIR}/testing.csv"
            st.session_state.train_df, st.session_state.test_df = load_data(train_path, test_path)
            st.success("Loaded 2025 hackathon data successfully!")
        elif train_file and test_file:
            st.session_state.train_df = pd.read_csv(train_file)
            st.session_state.test_df = pd.read_csv(test_file)
            st.session_state.train_df.columns = st.session_state.train_df.columns.str.strip()
            st.session_state.test_df.columns = st.session_state.test_df.columns.str.strip()
            st.success("Uploaded data loaded successfully!")
        else:
            st.error("Please select data sources")
    
    if st.session_state.train_df is not None:
        st.subheader("Training Data Overview")
        df = st.session_state.train_df
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Wells", len(df))
        col2.metric("Features", len(df.columns) - 4)
        col3.metric("Grid Wells", len(df[df['Fuel Type'] == 'Grid']))
        col4.metric("DGB Wells", len(df[df['Fuel Type'] == 'DGB']))
        
        st.markdown("**First 10 Rows:**")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("**Data Types & Missing Values:**")
        info_df = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.values,
            'Non-Null': df.count().values,
            'Null': df.isnull().sum().values,
            'Null %': (df.isnull().sum() / len(df) * 100).round(2).values
        })
        st.dataframe(info_df, use_container_width=True)
        
        st.markdown("**Numeric Statistics:**")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.subheader("Test Data Overview")
        test_df = st.session_state.test_df
        col1, col2 = st.columns(2)
        col1.metric("Test Wells", len(test_df))
        col2.metric("Fuel Types", test_df['Fuel Type'].nunique())
        
        st.markdown("**Test Data Sample:**")
        st.dataframe(test_df.head(10), use_container_width=True)
        
        st.markdown("**Fuel Type Distribution (Test):**")
        fuel_counts = test_df['Fuel Type'].value_counts()
        fig = px.pie(values=fuel_counts.values, names=fuel_counts.index, title="Test Wells by Fuel Type")
        st.plotly_chart(fig, use_container_width=True)

elif page == "2. Data Cleaning & Imputation":
    st.header("Step 2: Data Cleaning & Imputation")
    
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
                st.dataframe(pd.DataFrame({'Column': missing_train.index, 'Missing': missing_train.values}))
            else:
                st.success("No missing values in training data!")
                
        with col2:
            st.markdown("**Test Data Missing Values:**")
            missing_test = test_df.isnull().sum()
            missing_test = missing_test[missing_test > 0]
            if len(missing_test) > 0:
                st.dataframe(pd.DataFrame({'Column': missing_test.index, 'Missing': missing_test.values}))
            else:
                st.success("No missing values in test data!")
        
        st.subheader("Imputation Strategy")
        
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
        categorical_cols = [c for c in categorical_cols if c not in ['Well Name']]
        
        st.markdown("**Numeric Columns:** Fill with median")
        st.markdown("**Categorical Columns:** Fill with mode")
        
        if st.button("Apply Imputation", type="primary"):
            for col in numeric_cols:
                median_val = train_df[col].median()
                train_df[col] = train_df[col].fillna(median_val)
                test_df[col] = test_df[col].fillna(median_val)
                
            for col in categorical_cols:
                if col in train_df.columns:
                    mode_val = train_df[col].mode()[0] if len(train_df[col].mode()) > 0 else 'Unknown'
                    train_df[col] = train_df[col].fillna(mode_val)
                    if col in test_df.columns:
                        test_df[col] = test_df[col].fillna(mode_val)
            
            target_cols = ['Grid', 'Diesel', 'CNG']
            for col in target_cols:
                if col in train_df.columns:
                    train_df[col] = train_df[col].fillna(0)
                if col in test_df.columns:
                    test_df[col] = test_df[col].fillna(0)
            
            st.session_state.cleaned_train = train_df
            st.session_state.cleaned_test = test_df
            
            st.success("Imputation complete! Data cleaned and ready for analysis.")
            
            remaining_missing = train_df.isnull().sum().sum() + test_df.isnull().sum().sum()
            st.metric("Remaining Missing Values", remaining_missing)

elif page == "3. Exploratory Data Analysis":
    st.header("Step 3: Exploratory Data Analysis")
    
    df = st.session_state.cleaned_train if st.session_state.cleaned_train is not None else st.session_state.train_df
    
    if df is None:
        st.warning("Please load data in Step 1 first.")
    else:
        tab1, tab2, tab3 = st.tabs(["Target Distributions", "Feature Analysis", "Correlations"])
        
        with tab1:
            st.subheader("Energy Consumption Distributions")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fig = px.histogram(df[df['Grid'] > 0], x='Grid', nbins=30, title="Grid Energy (kWh)")
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                fig = px.histogram(df[df['Diesel'] > 0], x='Diesel', nbins=30, title="Diesel (gal)")
                st.plotly_chart(fig, use_container_width=True)
                
            with col3:
                fig = px.histogram(df[df['CNG'] > 0], x='CNG', nbins=30, title="CNG (MMBTU)")
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Energy by Fuel Type")
            fuel_stats = df.groupby('Fuel Type')[['Grid', 'Diesel', 'CNG']].mean().reset_index()
            fig = px.bar(fuel_stats.melt(id_vars='Fuel Type'), x='Fuel Type', y='value', 
                        color='variable', barmode='group', title="Average Energy by Fuel Type")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Feature Distributions")
            
            numeric_features = ['# Stages', '# Clusters', 'Estimated Average Stage Time', 
                              'Actual Average Stage Time', 'Ambient Temperature']
            available_features = [f for f in numeric_features if f in df.columns]
            
            selected_feature = st.selectbox("Select Feature:", available_features)
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(df, x=selected_feature, nbins=30, title=f"{selected_feature} Distribution")
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                fig = px.box(df, x='Fuel Type', y=selected_feature, title=f"{selected_feature} by Fuel Type")
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Feature vs Energy Scatter Plots")
            target = st.selectbox("Select Target:", ['Grid', 'Diesel', 'CNG'])
            
            filtered_df = df[df[target] > 0] if target in df.columns else df
            
            fig = px.scatter(filtered_df, x=selected_feature, y=target, color='Fuel Type',
                           title=f"{selected_feature} vs {target}", hover_data=['Well Name'])
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Feature Correlations")
            
            numeric_df = df.select_dtypes(include=[np.number])
            corr_matrix = numeric_df.corr()
            
            fig = px.imshow(corr_matrix, text_auto='.2f', aspect='auto',
                          title="Correlation Heatmap", color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Top Correlations with Targets")
            for target in ['Grid', 'Diesel', 'CNG']:
                if target in corr_matrix.columns:
                    correlations = corr_matrix[target].drop(['Grid', 'Diesel', 'CNG'], errors='ignore')
                    correlations = correlations.abs().sort_values(ascending=False).head(5)
                    st.markdown(f"**{target}:** {', '.join([f'{c} ({v:.2f})' for c, v in correlations.items()])}")

elif page == "4. Feature Engineering":
    st.header("Step 4: Feature Engineering")
    
    df = st.session_state.cleaned_train if st.session_state.cleaned_train is not None else st.session_state.train_df
    test_df = st.session_state.cleaned_test if st.session_state.cleaned_test is not None else st.session_state.test_df
    
    if df is None:
        st.warning("Please load and clean data first.")
    else:
        st.subheader("Create Engineered Features")
        
        st.markdown("""
        **Proposed Features:**
        1. **Time Overrun** = Actual Stage Time - Estimated Stage Time (indicates efficiency)
        2. **Total Pumping Time** = # Stages × Actual Stage Time
        3. **Clusters per Stage** = # Clusters / # Stages
        """)
        
        if st.button("Apply Feature Engineering", type="primary"):
            for dataset in [df, test_df]:
                if 'Actual Average Stage Time' in dataset.columns and 'Estimated Average Stage Time' in dataset.columns:
                    dataset['Time_Overrun'] = dataset['Actual Average Stage Time'] - dataset['Estimated Average Stage Time']
                    dataset['Time_Overrun'] = dataset['Time_Overrun'].fillna(0)
                
                if '# Stages' in dataset.columns and 'Actual Average Stage Time' in dataset.columns:
                    dataset['Total_Pumping_Time'] = dataset['# Stages'] * dataset['Actual Average Stage Time'].fillna(
                        dataset['Estimated Average Stage Time'])
                    dataset['Total_Pumping_Time'] = dataset['Total_Pumping_Time'].fillna(0)
                
                if '# Clusters' in dataset.columns and '# Stages' in dataset.columns:
                    dataset['Clusters_per_Stage'] = dataset['# Clusters'] / dataset['# Stages'].replace(0, 1)
            
            st.session_state.cleaned_train = df
            st.session_state.cleaned_test = test_df
            st.success("Features engineered successfully!")
        
        if 'Time_Overrun' in df.columns:
            st.subheader("Engineered Feature Preview")
            eng_features = ['Time_Overrun', 'Total_Pumping_Time', 'Clusters_per_Stage']
            available = [f for f in eng_features if f in df.columns]
            st.dataframe(df[['Well Name'] + available].head(10), use_container_width=True)
        
        st.subheader("Categorical Encoding Preview")
        cat_cols = ['Frac Fleet', 'Fleet Type', 'Target Formation', 'Field Area', 'Fuel Type', 'Sand Provider']
        available_cats = [c for c in cat_cols if c in df.columns]
        
        st.markdown("These categorical columns will be encoded during model training:")
        for col in available_cats:
            unique_vals = df[col].nunique()
            st.markdown(f"- **{col}**: {unique_vals} unique values")

elif page == "5. Model Training":
    st.header("Step 5: Model Training")
    
    train_df = st.session_state.cleaned_train if st.session_state.cleaned_train is not None else st.session_state.train_df
    
    if train_df is None:
        st.warning("Please load and process data first.")
    else:
        st.subheader("Model Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            n_estimators = st.slider("Number of Trees", 50, 300, 100, 50)
            max_depth = st.selectbox("Max Depth", [None, 5, 10, 15, 20], index=0)
        with col2:
            min_samples_split = st.slider("Min Samples Split", 2, 20, 2)
            cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
        
        targets = ['Grid', 'Diesel', 'CNG']
        selected_targets = st.multiselect("Select Targets to Train:", targets, default=targets)
        
        if st.button("Train Models", type="primary"):
            feature_cols = ['# Stages', '# Clusters', 'Estimated Average Stage Time', 
                          'Actual Average Stage Time', 'Ambient Temperature']
            
            eng_features = ['Time_Overrun', 'Total_Pumping_Time', 'Clusters_per_Stage']
            feature_cols += [f for f in eng_features if f in train_df.columns]
            
            cat_cols = ['Frac Fleet', 'Fleet Type', 'Target Formation', 'Field Area', 'Fuel Type', 'Sand Provider']
            
            X = train_df.copy()
            
            for col in feature_cols:
                if col in X.columns:
                    X[col] = X[col].fillna(X[col].median())
            
            label_encoders = {}
            for col in cat_cols:
                if col in X.columns:
                    le = LabelEncoder()
                    X[col + '_encoded'] = le.fit_transform(X[col].astype(str))
                    label_encoders[col] = le
                    feature_cols.append(col + '_encoded')
            
            available_features = [f for f in feature_cols if f in X.columns]
            X_features = X[available_features].fillna(0)
            
            st.session_state.label_encoders = label_encoders
            st.session_state.feature_cols = available_features
            
            progress_bar = st.progress(0)
            results = []
            
            for i, target in enumerate(selected_targets):
                with st.spinner(f"Training {target} model..."):
                    y = train_df[target].fillna(0)
                    
                    model = RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        random_state=42,
                        n_jobs=-1
                    )
                    
                    cv_scores = cross_val_score(model, X_features, y, cv=cv_folds, scoring='r2')
                    
                    model.fit(X_features, y)
                    
                    y_pred = model.predict(X_features)
                    residuals = y - y_pred
                    
                    st.session_state.models[target] = model
                    st.session_state.residuals[target] = residuals.values
                    
                    results.append({
                        'Target': target,
                        'CV R² Mean': cv_scores.mean(),
                        'CV R² Std': cv_scores.std(),
                        'Train R²': r2_score(y, y_pred),
                        'Train MAE': mean_absolute_error(y, y_pred),
                        'Train RMSE': np.sqrt(mean_squared_error(y, y_pred))
                    })
                    
                progress_bar.progress((i + 1) / len(selected_targets))
            
            st.success("All models trained successfully!")
            
            st.subheader("Model Performance")
            results_df = pd.DataFrame(results)
            st.dataframe(results_df.round(4), use_container_width=True)
            
            st.subheader("Feature Importance")
            for target in selected_targets:
                model = st.session_state.models[target]
                importance_df = pd.DataFrame({
                    'Feature': available_features,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False).head(10)
                
                fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                           title=f"{target} - Top 10 Feature Importances")
                st.plotly_chart(fig, use_container_width=True)

elif page == "6. Uncertainty Quantification":
    st.header("Step 6: Uncertainty Quantification")
    
    if not st.session_state.models:
        st.warning("Please train models in Step 5 first.")
    else:
        st.subheader("Bootstrap Uncertainty Estimation")
        
        st.markdown("""
        **Method: Residual Bootstrapping**
        
        For each prediction, we:
        1. Generate a point estimate using the trained model
        2. Sample 100 residuals from training with replacement (using `random_state=42` for reproducibility)
        3. Add sampled residuals to create 100 realizations (Real_1 through Real_100)
        
        **Why Residual Bootstrapping?**
        - Captures both model uncertainty and inherent data variability
        - Non-parametric approach - makes no assumptions about error distribution
        - Preserves the empirical error structure from cross-validation
        - Ensures predictions reflect realistic ranges based on training performance
        
        **Reproducibility:** All random operations use `random_state=42` to ensure 
        identical results when re-running the pipeline.
        """)
        
        st.subheader("Residual Distributions")
        
        cols = st.columns(len(st.session_state.residuals))
        for i, (target, residuals) in enumerate(st.session_state.residuals.items()):
            with cols[i]:
                fig = px.histogram(x=residuals, nbins=50, title=f"{target} Residuals")
                st.plotly_chart(fig, use_container_width=True)
                st.metric(f"{target} Residual Std", f"{np.std(residuals):.2f}")
        
        st.subheader("Uncertainty Coverage Analysis")
        
        for target, residuals in st.session_state.residuals.items():
            p5, p95 = np.percentile(residuals, [5, 95])
            coverage = np.mean((residuals >= p5) & (residuals <= p95)) * 100
            st.markdown(f"**{target}**: 90% prediction interval covers {coverage:.1f}% of residuals (P5={p5:.1f}, P95={p95:.1f})")

elif page == "7. Generate Predictions":
    st.header("Step 7: Generate Predictions")
    
    if not st.session_state.models:
        st.warning("Please train models first.")
    else:
        test_df = st.session_state.cleaned_test if st.session_state.cleaned_test is not None else st.session_state.test_df
        
        if test_df is None:
            st.warning("No test data loaded.")
        else:
            st.subheader("Test Data Summary")
            st.dataframe(test_df.head(), use_container_width=True)
            
            fuel_counts = test_df['Fuel Type'].value_counts()
            st.markdown(f"**Wells by Fuel Type:** {dict(fuel_counts)}")
            
            n_realizations = st.slider("Number of Realizations", 10, 100, 100)
            
            if st.button("Generate Predictions", type="primary"):
                np.random.seed(42)
                X_test = test_df.copy()
                
                feature_cols = st.session_state.feature_cols
                label_encoders = st.session_state.label_encoders
                
                for col, le in label_encoders.items():
                    if col in X_test.columns:
                        X_test[col + '_encoded'] = X_test[col].astype(str).map(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
                
                available = [f for f in feature_cols if f in X_test.columns]
                X_test_features = X_test[available].fillna(0)
                
                results = []
                
                progress_bar = st.progress(0)
                
                for idx, row in test_df.iterrows():
                    well_name = row['Well Name']
                    fuel_type = row['Fuel Type']
                    
                    x_single = X_test_features.iloc[[idx - test_df.index[0]]]
                    
                    if fuel_type == 'Grid':
                        targets = [('Grid', 'Grid')]
                    elif fuel_type == 'Diesel':
                        targets = [('Diesel', 'Diesel')]
                    elif fuel_type == 'Turbine':
                        targets = [('Turbine', 'CNG')]
                    elif fuel_type == 'DGB':
                        targets = [('DGB_Diesel', 'Diesel'), ('DGB_CNG', 'CNG')]
                    else:
                        targets = [('Grid', 'Grid')]
                    
                    for target_name, model_key in targets:
                        if model_key in st.session_state.models:
                            model = st.session_state.models[model_key]
                            residuals = st.session_state.residuals[model_key]
                            
                            point_estimate = model.predict(x_single)[0]
                            
                            sampled_residuals = np.random.choice(residuals, size=n_realizations, replace=True)
                            realizations = point_estimate + sampled_residuals
                            realizations = np.maximum(realizations, 0)
                            
                            result = {
                                'Masked Well Name': well_name,
                                'Fuel Type': target_name,
                                'Fuel Value': max(0, point_estimate)
                            }
                            for r in range(n_realizations):
                                result[f'Real_{r+1}'] = realizations[r]
                            
                            results.append(result)
                    
                    progress_bar.progress((idx - test_df.index[0] + 1) / len(test_df))
                
                predictions_df = pd.DataFrame(results)
                st.session_state.predictions = predictions_df
                
                st.success(f"Generated {len(predictions_df)} prediction rows!")
                
                st.subheader("Predictions Preview")
                display_cols = ['Masked Well Name', 'Fuel Type', 'Fuel Value', 'Real_1', 'Real_2', 'Real_3', 'Real_4', 'Real_5']
                st.dataframe(predictions_df[display_cols].head(20), use_container_width=True)
                
                csv = predictions_df.to_csv(index=False)
                st.download_button(
                    label="Download Solution CSV",
                    data=csv,
                    file_name="solution.csv",
                    mime="text/csv"
                )
                
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                predictions_df.to_csv(f"{OUTPUT_DIR}/solution.csv", index=False)
                st.info(f"Solution saved to {OUTPUT_DIR}/solution.csv")

elif page == "8. Hackathon Quick Start":
    st.header("Quick Start Guide for 2026 Hackathon")
    
    st.markdown("""
    ## When the 2026 Hackathon Starts:
    
    ### Step 1: Upload New Data
    1. Go to **Step 1: Data Upload**
    2. Upload the new training and test CSV files provided
    3. Review the data structure and any new columns
    
    ### Step 2: Quick Data Prep
    1. Go to **Step 2: Data Cleaning** - Apply imputation
    2. Go to **Step 4: Feature Engineering** - Apply feature engineering
    
    ### Step 3: Train Models
    1. Go to **Step 5: Model Training**
    2. Use default settings or tune as needed
    3. Review performance metrics
    
    ### Step 4: Generate Submission
    1. Go to **Step 7: Generate Predictions**
    2. Click "Generate Predictions"
    3. Download the solution CSV
    
    ---
    
    ## Key Adaptations for 2026:
    
    1. **Check new columns** - If new features are added, update the feature list
    2. **Check target format** - Verify the solution format requirements
    3. **Check fuel types** - New fuel types may require model adjustments
    4. **Review uncertainty requirements** - Confirm 100 realizations are still needed
    
    ---
    
    ## Tips for Success:
    
    - **Run EDA first** to understand data patterns
    - **Check for outliers** in new data
    - **Cross-validate** to ensure model generalization
    - **Save your work** - Download the solution immediately after generation
    """)
    
    st.subheader("Current Model Status")
    
    if st.session_state.models:
        for target, model in st.session_state.models.items():
            st.success(f"{target} model: Trained")
    else:
        st.warning("No models trained yet. Complete Steps 1-5 first.")
    
    if st.session_state.predictions is not None:
        st.success(f"Predictions generated: {len(st.session_state.predictions)} rows")
    else:
        st.info("No predictions generated yet.")

elif page == "9. AI ML Assistant":
    st.header("AI-Powered ML Assistant")
    
    st.markdown("""
    **Ask me anything about machine learning, data science, or how to adapt this pipeline!**
    
    I can help you with:
    - Suggesting features for your specific dataset
    - Recommending ML models for different problem types
    - Explaining uncertainty quantification methods
    - Adapting this pipeline to different industries
    - Troubleshooting model performance issues
    - Understanding your data patterns
    """)
    
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    
    def get_data_context():
        context_parts = []
        
        if st.session_state.train_df is not None:
            df = st.session_state.train_df
            context_parts.append(f"LOADED TRAINING DATA: {len(df)} rows, {len(df.columns)} columns")
            context_parts.append(f"Columns: {', '.join(df.columns.tolist())}")
            context_parts.append(f"Numeric columns: {', '.join(df.select_dtypes(include=[np.number]).columns.tolist())}")
            context_parts.append(f"Categorical columns: {', '.join(df.select_dtypes(include=['object']).columns.tolist())}")
            
            if 'Fuel Type' in df.columns:
                context_parts.append(f"Fuel Types: {df['Fuel Type'].value_counts().to_dict()}")
            
            missing = df.isnull().sum()
            missing = missing[missing > 0]
            if len(missing) > 0:
                context_parts.append(f"Missing values: {missing.to_dict()}")
        
        if st.session_state.test_df is not None:
            context_parts.append(f"TEST DATA: {len(st.session_state.test_df)} wells to predict")
        
        if st.session_state.models:
            context_parts.append(f"TRAINED MODELS: {list(st.session_state.models.keys())}")
            for target, residuals in st.session_state.residuals.items():
                context_parts.append(f"{target} residual std: {np.std(residuals):.2f}")
        
        if st.session_state.predictions is not None:
            context_parts.append(f"PREDICTIONS: {len(st.session_state.predictions)} rows generated")
        
        return "\n".join(context_parts) if context_parts else "No data loaded yet."
    
    def get_system_prompt():
        data_context = get_data_context()
        
        return f"""You are an expert ML assistant integrated into the Energy AI Hackathon 2026 workflow application, built by Team Energy Gladiators.

YOUR TWO ROLES:
1. TEAM ONBOARDING: Help teammates understand this application and walk them through each step
2. ML EXPERT: Help adapt the pipeline to any dataset, industry, or problem type

CURRENT APPLICATION STATE:
{data_context}

=== THIS APPLICATION'S WORKFLOW (9 STEPS) ===

STEP 1 - DATA UPLOAD & INSPECTION:
- Upload training data (historical wells with energy usage) and test data (wells to predict)
- View basic statistics, data types, missing values
- The training data has ~1,082 wells, test data has 50 wells

STEP 2 - DATA CLEANING & IMPUTATION:
- Handle missing values: median for numbers, mode for categories
- No rows are dropped - all data is preserved

STEP 3 - EXPLORATORY DATA ANALYSIS (EDA):
- Visualize energy distributions (histograms)
- Correlation heatmaps to see relationships
- Box plots by Fleet Type and Formation

STEP 4 - FEATURE ENGINEERING:
- Create derived features:
  * Time_Overrun = Actual Stage Time - Estimated Stage Time
  * Total_Pumping_Time = Number of Stages × Stage Time
  * Clusters_per_Stage = Number of Clusters / Number of Stages
- These capture operational patterns that affect energy usage

STEP 5 - MODEL TRAINING:
- Train separate Random Forest models for Grid (kWh), Diesel (gal), CNG (MMBTU)
- Uses 5-fold cross-validation
- Key insight: Fleet Type determines which fuel a well uses:
  * Grid fleet → only Grid electricity
  * Diesel fleet → only Diesel
  * Turbine fleet → only CNG
  * DGB fleet → BOTH Diesel AND CNG

STEP 6 - UNCERTAINTY QUANTIFICATION:
- Residual bootstrapping: sample 100 residuals and add to predictions
- This creates 100 "realizations" showing the range of possible outcomes
- Gives operators a planning buffer, not just a single number

STEP 7 - GENERATE PREDICTIONS:
- Creates solution.csv with columns: Masked Well Name, Fuel Type, Fuel Value, Real_1 through Real_100
- DGB wells generate TWO rows (one for Diesel, one for CNG)
- Total: 63 rows for 50 wells (13 DGB wells × 2 = 26, plus 37 single-fuel wells)

STEP 8 - QUICK START GUIDE:
- Instructions for hackathon execution

STEP 9 - AI ML ASSISTANT (this chat):
- That's me! Here to help.

=== WHY THIS IS INNOVATIVE ===
Commercial tools like Spotfire cost $3,000-5,000/year and only show WHAT HAPPENED (descriptive).
Our solution predicts WHAT WILL HAPPEN with uncertainty ranges (predictive + probabilistic).
Plus, I (the AI assistant) can help anyone adapt it to new problems without coding.

=== YOUR ML EXPERTISE ===
1. MODEL SELECTION: Random Forest, XGBoost, LightGBM, Neural Networks, Linear Models, SVR
2. FEATURE ENGINEERING: Domain-specific features, interactions, polynomial features
3. UNCERTAINTY METHODS: Bootstrapping, Monte Carlo dropout, Bayesian approaches, quantile regression
4. DATA PREPROCESSING: Scaling, encoding, outlier handling, missing value strategies
5. PROBLEM TYPES: Regression, classification, time series, anomaly detection
6. INDUSTRIES: Energy, oil & gas, finance, healthcare, manufacturing, retail

=== LOCAL INSTALLATION INSTRUCTIONS ===
If someone asks how to run this on their own laptop/machine:

1. REQUIREMENTS:
   - Python 3.8 or higher
   - pip (Python package manager)

2. DOWNLOAD THE PROJECT:
   - Download the recipe zip file from Replit (or clone the GitHub repo)
   - Extract to a folder on your computer

3. INSTALL DEPENDENCIES:
   Open terminal/command prompt in the project folder and run:
   ```
   pip install streamlit pandas numpy scikit-learn matplotlib seaborn plotly openai
   ```

4. SET UP OPENAI API KEY (for AI Assistant only):
   - Get an API key from https://platform.openai.com/api-keys
   - Set environment variable:
     * Windows: set OPENAI_API_KEY=your-key-here
     * Mac/Linux: export OPENAI_API_KEY=your-key-here
   - Note: The ML workflow (Steps 1-7) works WITHOUT an API key. Only Step 9 (AI Assistant) needs it.

5. RUN THE APP:
   ```
   streamlit run app.py
   ```
   This opens the app in your browser at http://localhost:8501

6. UPLOAD DATA:
   - Use the same training/test CSV files
   - Follow Steps 1-7 to generate predictions

TROUBLESHOOTING:
- "Module not found" → Run pip install for the missing package
- "streamlit not recognized" → Add Python Scripts folder to PATH, or use: python -m streamlit run app.py
- App won't start → Check you're in the right folder containing app.py

=== HOW TO RESPOND ===
- If someone asks "what does Step X do?" - explain it clearly with the hackathon context
- If someone asks "how do I use this?" - walk them through step by step
- If someone asks about ML concepts - explain thoroughly
- If someone wants to adapt the pipeline - suggest specific changes
- If someone asks about local installation - give the step-by-step instructions above
- Use bullet points for clarity
- Reference the current data state when relevant

Be friendly, concise, and helpful. You're here to make sure everyone on the team understands the workflow and can execute it confidently."""

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask me about ML, features, models, or how to adapt this pipeline..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    client = get_openai_client()
                    
                    messages = [{"role": "system", "content": get_system_prompt()}]
                    for msg in st.session_state.chat_messages[-10:]:
                        messages.append({"role": msg["role"], "content": msg["content"]})
                    
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        max_tokens=2000,
                        temperature=0.7
                    )
                    
                    assistant_response = response.choices[0].message.content
                    st.markdown(assistant_response)
                    st.session_state.chat_messages.append({"role": "assistant", "content": assistant_response})
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
    
    with st.expander("Quick Prompts - Click to Use"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Data & Features:**")
            prompts1 = [
                "What features should I engineer from my current data?",
                "How can I handle missing values better?",
                "What patterns do you see in my data?",
                "How should I encode my categorical variables?"
            ]
            for p in prompts1:
                if st.button(p, key=f"p1_{p[:20]}"):
                    st.session_state.pending_prompt = p
                    st.rerun()
        
        with col2:
            st.markdown("**Models & Methods:**")
            prompts2 = [
                "What ML models would work better for this problem?",
                "How can I improve my uncertainty quantification?",
                "Should I use ensemble methods?",
                "How do I adapt this for a classification problem?"
            ]
            for p in prompts2:
                if st.button(p, key=f"p2_{p[:20]}"):
                    st.session_state.pending_prompt = p
                    st.rerun()
    
    if 'pending_prompt' in st.session_state:
        prompt = st.session_state.pending_prompt
        del st.session_state.pending_prompt
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        st.rerun()
    
    if st.button("Clear Chat History"):
        st.session_state.chat_messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("**Current Data Context:**")
    st.code(get_data_context(), language="text")

st.sidebar.markdown("---")
st.sidebar.markdown("**Energy AI Hackathon 2026**")
st.sidebar.markdown("Built for rapid ML workflow execution")
