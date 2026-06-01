import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import sys
from datetime import datetime

# Machine Learning & Stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    silhouette_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.api import ExponentialSmoothing

# Page Configuration
st.set_page_config(
    page_title="Time Series Anomaly Detection Platform",
    page_icon="⏰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    /* Styling cards */
    .metric-card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
    }
    .metric-title {
        color: #94a3b8;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #38bdf8;
        font-size: 24px;
        font-weight: 700;
    }
    .metric-sub {
        color: #64748b;
        font-size: 11px;
        margin-top: 4px;
    }
    
    /* Title and Header styling */
    .main-title {
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Code/Formula styling */
    .formula-box {
        background-color: #0f172a;
        padding: 15px;
        border-left: 4px solid #818cf8;
        border-radius: 4px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Main Title & Subtitle
st.markdown('<div class="main-title">⏰ Multi-Variate Time Series Anomaly Detection Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload arbitrary time series data, automatically analyze statistical properties, and deploy advanced anomaly detection models.</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# 1. MATHEMATICALLY RIGOROUS HELPER FUNCTIONS
# -------------------------------------------------------------

@st.cache_data
def generate_synthetic_data(n_samples=500):
    """Generates synthetic multi-variate time-series sensor data with injected anomalies."""
    np.random.seed(42)
    timestamps = pd.date_range(start="2026-01-01", periods=n_samples, freq="h")
    
    t = np.arange(n_samples)
    
    # Sensor 1: Temperature (daily sine pattern + upward trend)
    temp_trend = 0.01 * t
    temp_season = 5 * np.sin(2 * np.pi * t / 24)
    temp_noise = np.random.normal(0, 1.2, n_samples)
    temperature = 20 + temp_trend + temp_season + temp_noise
    
    # Sensor 2: Pressure (correlated with temperature + daily cosine + noise)
    press_season = 3 * np.cos(2 * np.pi * t / 24)
    press_noise = np.random.normal(0, 0.8, n_samples)
    pressure = 101.3 + (temperature - 20) * 0.1 + press_season + press_noise
    
    # Sensor 3: Vibration (random walk process + high frequency noise)
    vib_noise = np.random.normal(0, 0.5, n_samples)
    vibration = np.cumsum(np.random.normal(0, 0.05, n_samples)) + vib_noise + 2.0
    
    df = pd.DataFrame({
        "Timestamp": timestamps,
        "Temperature": temperature,
        "Pressure": pressure,
        "Vibration": vibration
    })
    
    # Inject ground truth anomalies (labeled as 1)
    labels = np.zeros(n_samples, dtype=int)
    
    # Type 1: Point anomaly (Spikes in Temperature)
    temp_spike_idx = [80, 220, 360]
    for idx in temp_spike_idx:
        df.loc[idx, "Temperature"] += 15.0
        labels[idx] = 1
        
    # Type 2: Contextual anomaly (Sudden pressure drops for 5 steps)
    press_drop_idx = range(140, 145)
    for idx in press_drop_idx:
        df.loc[idx, "Pressure"] -= 8.0
        labels[idx] = 1
        
    # Type 3: Collective anomaly (Sudden shift/vibration increase for 8 steps)
    vib_shift_idx = range(290, 298)
    for idx in vib_shift_idx:
        df.loc[idx, "Vibration"] += 3.5
        labels[idx] = 1
        
    df["Is_Anomaly"] = labels
    return df

def hampel_filter_vectorized(series, window_size=15, n_sigmas=3):
    """Computes Hampel Filter for anomaly detection using rolling window median and MAD."""
    roll = pd.Series(series)
    rolling_median = roll.rolling(window=window_size, center=True, min_periods=1).median()
    rolling_mad = (roll - rolling_median).abs().rolling(window=window_size, center=True, min_periods=1).median()
    scale = 1.4826 * rolling_mad
    scale = scale.replace(0, 1e-6) # Avoid division by zero
    
    scores = (roll - rolling_median).abs() / scale
    flags = scores > n_sigmas
    return flags.values, scores.values

def mahalanobis_distance_with_contrib(X):
    """Calculates Mahalanobis distance and local feature contributions for multivariate data."""
    X_arr = np.array(X, dtype=float)
    mean = np.mean(X_arr, axis=0)
    cov = np.cov(X_arr, rowvar=False)
    
    if cov.ndim == 0 or cov.shape == ():
        var = cov + 1e-8
        diff = X_arr - mean
        md = np.sqrt(diff * (1.0 / var) * diff)
        contrib = np.abs(diff).reshape(-1, 1)
        return md, contrib
    
    # Check singularity; use pseudo-inverse if singular
    if np.linalg.cond(cov) > 1 / sys.float_info.epsilon:
        cov_inv = np.linalg.pinv(cov)
    else:
        cov_inv = np.linalg.inv(cov)
        
    diff = X_arr - mean
    # Mahalanobis Distance squared is diff * cov_inv * diff^T
    # The contribution of feature j is diff_j * [cov_inv * diff^T]_j
    proj = np.dot(diff, cov_inv)
    contrib = diff * proj
    abs_contrib = np.abs(contrib)
    md = np.sqrt(np.sum(contrib, axis=1))
    return md, abs_contrib

def pca_reconstruction_with_contrib(X, n_components=2):
    """Calculates PCA reconstruction error and the individual feature contribution to that error."""
    from sklearn.decomposition import PCA
    n_feats = X.shape[1]
    pca = PCA(n_components=min(n_components, n_feats))
    X_proj = pca.fit_transform(X)
    X_recon = pca.inverse_transform(X_proj)
    
    # Reconstruction error per feature: (x_j - x_recon_j)^2
    recon_error_features = (X - X_recon) ** 2
    total_recon_error = np.sum(recon_error_features, axis=1)
    return total_recon_error, recon_error_features

def fit_predict_ridge_lags(X_scaled, split_idx, lags=[1, 2, 3, 24]):
    """Fits separate Ridge regression models with lag features to predict multivariate steps."""
    n_samples, n_features = X_scaled.shape
    preds = np.zeros_like(X_scaled)
    
    for f_idx in range(n_features):
        y = X_scaled[:, f_idx]
        
        # Build Lag variables matrix
        X_lags = []
        for lag in lags:
            shifted = np.roll(y, lag)
            shifted[:lag] = y[0] # Backfill values
            X_lags.append(shifted)
            
        X_design = np.column_stack(X_lags)
        
        # Fit Ridge regression model on the training partition
        model = Ridge(alpha=1.0)
        model.fit(X_design[:split_idx], y[:split_idx])
        preds[:, f_idx] = model.predict(X_design)
        
    return preds

# -------------------------------------------------------------
# 2. SIDEBAR CONFIGURATION
# -------------------------------------------------------------
st.sidebar.markdown("### 📂 Data Source & Variables")

data_source = st.sidebar.selectbox(
    "Select Data Source",
    options=["Industrial Sensor (Synthetic Sample)", "Upload Custom CSV"]
)

if data_source == "Upload Custom CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            st.sidebar.success("File uploaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")
            raw_df = generate_synthetic_data()
    else:
        st.sidebar.info("Using sample synthetic data until custom CSV is uploaded.")
        raw_df = generate_synthetic_data()
else:
    raw_df = generate_synthetic_data()

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Column Selection")
columns = list(raw_df.columns)

# Auto-detect timestamp column
date_cols = [col for col in columns if "time" in col.lower() or "date" in col.lower()]
default_time_idx = columns.index(date_cols[0]) if date_cols else 0
time_col = st.sidebar.selectbox("Timestamp Column", options=columns, index=default_time_idx)

# Auto-detect numeric columns
num_cols = list(raw_df.select_dtypes(include=[np.number]).columns)
if time_col in num_cols:
    num_cols.remove(time_col)

# Auto-detect ground truth anomaly column
gt_cols = [col for col in columns if "anomaly" in col.lower() or "label" in col.lower() or "gt" in col.lower() or "target" in col.lower()]
default_gt = gt_cols[0] if gt_cols else None
if default_gt in num_cols:
    num_cols.remove(default_gt)

target_cols = st.sidebar.multiselect(
    "Target Features (for Anomaly Detection)",
    options=num_cols,
    default=num_cols[:min(3, len(num_cols))]
)

has_gt = st.sidebar.checkbox("Has Ground Truth Labels?", value=(default_gt is not None))
if has_gt:
    gt_options = [col for col in columns if col != time_col]
    if not gt_options:
        st.sidebar.warning("No columns available for Ground Truth.")
        gt_col = None
    else:
        default_idx = gt_options.index(default_gt) if (default_gt in gt_options) else 0
        gt_col = st.sidebar.selectbox(
            "Ground Truth Column",
            options=gt_options,
            index=default_idx
        )
else:
    gt_col = None

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧹 Data Preprocessing")
missing_impute = st.sidebar.selectbox(
    "Missing Value Imputation",
    options=["Linear Interpolation", "Forward Fill", "Backward Fill", "No Imputation (Drop NA)"]
)
scaling_method = st.sidebar.selectbox(
    "Feature Scaling",
    options=["StandardScaler (z-score)", "MinMaxScaler (0-1)", "None"]
)
denoising_method = st.sidebar.selectbox(
    "Denoising/Smoothing Method",
    options=["None", "Simple Moving Average", "Differencing (t - (t-1))"]
)
if denoising_method == "Simple Moving Average":
    ma_window = st.sidebar.slider("Moving Average Window Size", min_value=3, max_value=31, value=5, step=2)
else:
    ma_window = 1

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Model Selection")
algo_type = st.sidebar.selectbox(
    "Anomaly Model Type",
    options=["Unsupervised Feature-Space", "Time-Series Forecasting Residuals"]
)

# Initialize parameter configs
model_params = {}

if algo_type == "Unsupervised Feature-Space":
    model_name = st.sidebar.selectbox(
        "Select Model",
        options=["Isolation Forest", "Mahalanobis Distance", "PCA Reconstruction Error", "Local Outlier Factor (LOF)"]
    )
    contamination = st.sidebar.slider("Contamination Rate (%)", min_value=0.1, max_value=20.0, value=2.0, step=0.1) / 100.0
    model_params["contamination"] = contamination
    
    if model_name == "Isolation Forest":
        n_estimators = st.sidebar.slider("Number of Trees", min_value=50, max_value=300, value=100, step=50)
        model_params["n_estimators"] = n_estimators
    elif model_name == "PCA Reconstruction Error":
        pca_components = st.sidebar.slider("Number of Principal Components", min_value=1, max_value=max(1, len(target_cols)), value=min(2, len(target_cols)))
        model_params["pca_components"] = pca_components
    elif model_name == "Local Outlier Factor (LOF)":
        n_neighbors = st.sidebar.slider("Number of Neighbors", min_value=5, max_value=50, value=20)
        model_params["n_neighbors"] = n_neighbors
else:
    model_name = st.sidebar.selectbox(
        "Select Model",
        options=["Rolling Hampel Filter", "STL Decomposition Residuals", "Forecasting Model Residuals"]
    )
    
    if model_name == "Rolling Hampel Filter":
        hampel_window = st.sidebar.slider("Hampel Window Size", min_value=5, max_value=51, value=15, step=2)
        hampel_sigmas = st.sidebar.slider("Hampel Sigma Threshold", min_value=1.5, max_value=5.0, value=3.0, step=0.1)
        model_params["hampel_window"] = hampel_window
        model_params["hampel_sigmas"] = hampel_sigmas
    elif model_name == "STL Decomposition Residuals":
        stl_period = st.sidebar.slider("Seasonal Period", min_value=4, max_value=168, value=24, step=1)
        stl_sigma = st.sidebar.slider("Residual Sigma Threshold", min_value=1.5, max_value=5.0, value=3.0, step=0.1)
        model_params["stl_period"] = stl_period
        model_params["stl_sigma"] = stl_sigma
    elif model_name == "Forecasting Model Residuals":
        forecaster_choice = st.sidebar.selectbox("Base Forecaster", ["Linear Ridge (Lags)", "Exponential Smoothing", "Rolling Mean Predictor"])
        fc_sigma = st.sidebar.slider("Residual Sigma Threshold", min_value=1.5, max_value=5.0, value=3.0, step=0.1)
        train_ratio = st.sidebar.slider("Train Split Ratio", min_value=0.5, max_value=0.9, value=0.7, step=0.05)
        model_params["forecaster_choice"] = forecaster_choice
        model_params["fc_sigma"] = fc_sigma
        model_params["train_ratio"] = train_ratio

if not target_cols:
    st.error("Please select at least one Target Feature in the sidebar to perform anomaly detection.")
    st.stop()

# -------------------------------------------------------------
# 3. PIPELINE - DATA PREPROCESSING
# -------------------------------------------------------------
df = raw_df.copy()
df[time_col] = pd.to_datetime(df[time_col])
df = df.sort_values(by=time_col).reset_index(drop=True)

# Missing values treatment
if missing_impute == "Linear Interpolation":
    df[target_cols] = df[target_cols].interpolate(method="linear", limit_direction="both")
elif missing_impute == "Forward Fill":
    df[target_cols] = df[target_cols].ffill().bfill()
elif missing_impute == "Backward Fill":
    df[target_cols] = df[target_cols].bfill().ffill()
elif missing_impute == "No Imputation (Drop NA)":
    df = df.dropna(subset=target_cols).reset_index(drop=True)

# Deep copy of parsed target data for plotting
df_original = df.copy()

# Denoising operations
if denoising_method == "Simple Moving Average":
    df[target_cols] = df[target_cols].rolling(window=ma_window, min_periods=1, center=True).mean()
elif denoising_method == "Differencing (t - (t-1))":
    df[target_cols] = df[target_cols].diff().fillna(0)

# Multi-variate scaling transformation
X_raw = df[target_cols].values.astype(float)
if scaling_method == "StandardScaler (z-score)":
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
elif scaling_method == "MinMaxScaler (0-1)":
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)
else:
    X_scaled = X_raw

df_scaled = df.copy()
df_scaled[target_cols] = X_scaled

# -------------------------------------------------------------
# 4. TAB CONTROLS
# -------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📂 Data Overview & Preprocessing",
    "📈 Exploratory Time-Series Analysis (EDA)",
    "🔍 Anomaly Detection Results",
    "📊 Evaluation Dashboard"
])

# -------------------------------------------------------------
# TAB 1: DATA OVERVIEW & PREPROCESSING
# -------------------------------------------------------------
with tab1:
    st.markdown("### 📊 Dataset Properties & Preprocessing Inspection")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Records</div>
            <div class="metric-value">{len(df)}</div>
            <div class="metric-sub">Time spans from {df[time_col].min().strftime('%Y-%m-%d')} to {df[time_col].max().strftime('%Y-%m-%d')}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Selected Target Features</div>
            <div class="metric-value">{len(target_cols)}</div>
            <div class="metric-sub">{', '.join(target_cols)}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        na_count = raw_df[target_cols].isna().sum().sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Missing Values (Raw Targets)</div>
            <div class="metric-value">{na_count}</div>
            <div class="metric-sub">Handling: {missing_impute}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Sample Dataset View (First 5 Rows)")
    st.dataframe(df.head(), use_container_width=True)
    
    st.markdown("#### Raw vs. Preprocessed Time Series Comparison")
    st.caption("Inspect the effect of selected imputation, smoothing, and scaling configurations.")
    
    feature_to_plot = st.selectbox("Select Feature to Visualize", options=target_cols)
    
    fig_prep = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=["Raw Series", "Preprocessed Series"])
    fig_prep.add_trace(
        go.Scatter(x=df_original[time_col], y=df_original[feature_to_plot], name="Raw Data", line=dict(color="#94a3b8")),
        row=1, col=1
    )
    fig_prep.add_trace(
        go.Scatter(x=df[time_col], y=df_scaled[feature_to_plot], name="Preprocessed", line=dict(color="#38bdf8")),
        row=2, col=1
    )
    fig_prep.update_layout(
        height=500,
        showlegend=True,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_prep, use_container_width=True)

# -------------------------------------------------------------
# TAB 2: EXPLORATORY TIME-SERIES ANALYSIS
# -------------------------------------------------------------
with tab2:
    st.markdown("### 📈 Time-Series Statistical & Structural Analysis")
    
    st.markdown("#### 1. Stationarity Assessment: Augmented Dickey-Fuller (ADF) Test")
    st.markdown("""
    The ADF test checks for the presence of a unit root (non-stationarity). 
    *   **Null Hypothesis ($H_0$)**: The series has a unit root (is non-stationary).
    *   **Alternative Hypothesis ($H_1$)**: The series is stationary ($p\\text{-value} < 0.05$).
    """)
    
    adf_results = []
    for col in target_cols:
        series_clean = df[col].dropna()
        if series_clean.std() == 0:
            adf_results.append({
                "Feature": col,
                "ADF Statistic": np.nan,
                "p-value": np.nan,
                "Lags Used": 0,
                "Stationary?": "Constant Value"
            })
            continue
            
        try:
            res = adfuller(series_clean, regression="c", autolag="AIC")
            adf_stat = res[0]
            p_val = res[1]
            lags = res[2]
            stationary = "✅ Yes (p < 0.05)" if p_val < 0.05 else "❌ No (Non-Stationary)"
            adf_results.append({
                "Feature": col,
                "ADF Statistic": f"{adf_stat:.4f}",
                "p-value": f"{p_val:.4f}" if p_val >= 0.0001 else "< 0.0001",
                "Lags Used": lags,
                "Stationary?": stationary
            })
        except Exception as e:
            adf_results.append({
                "Feature": col,
                "ADF Statistic": "Fail",
                "p-value": "Fail",
                "Lags Used": 0,
                "Stationary?": f"Error: {e}"
            })
            
    st.table(pd.DataFrame(adf_results))
    
    st.markdown("#### 2. Autocorrelation Analysis (ACF / PACF)")
    st.caption("Identify temporal dependencies, lags, and seasonalities in the target series.")
    
    acf_feature = st.selectbox("Select Feature for ACF/PACF", options=target_cols, key="acf_feat")
    n_lags = st.slider("Number of Lags", min_value=10, max_value=100, value=40, step=5)
    
    series_acf = df[acf_feature].values
    lag_acf = acf(series_acf, nlags=n_lags, fft=True)
    lag_pacf = pacf(series_acf, nlags=n_lags, method="yw")
    
    fig_corr = make_subplots(rows=1, cols=2, subplot_titles=["Autocorrelation (ACF)", "Partial Autocorrelation (PACF)"])
    fig_corr.add_trace(
        go.Bar(x=list(range(n_lags + 1)), y=lag_acf, name="ACF", marker_color="#818cf8"),
        row=1, col=1
    )
    fig_corr.add_trace(
        go.Bar(x=list(range(n_lags + 1)), y=lag_pacf, name="PACF", marker_color="#38bdf8"),
        row=1, col=2
    )
    
    # Standard confidence interval boundaries (+/- 1.96 / sqrt(N))
    ci = 1.96 / np.sqrt(len(series_acf))
    fig_corr.add_hline(y=ci, line_dash="dash", line_color="#ef4444", line_width=1, row=1, col=1)
    fig_corr.add_hline(y=-ci, line_dash="dash", line_color="#ef4444", line_width=1, row=1, col=1)
    fig_corr.add_hline(y=ci, line_dash="dash", line_color="#ef4444", line_width=1, row=1, col=2)
    fig_corr.add_hline(y=-ci, line_dash="dash", line_color="#ef4444", line_width=1, row=1, col=2)
    
    fig_corr.update_layout(
        height=350,
        showlegend=False,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("#### 3. Multivariate Feature Correlation Matrix")
    if len(target_cols) > 1:
        corr_matrix = df[target_cols].corr()
        fig_corr_mat = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title="Pearson Correlation Heatmap"
        )
        fig_corr_mat.update_layout(
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_corr_mat, use_container_width=True)
    else:
        st.info("Correlation matrix requires at least two selected features.")

# -------------------------------------------------------------
# 5. CORE PIPELINE - ANOMALY DETECTION ENGINE (CACHED)
# -------------------------------------------------------------

@st.cache_data(show_spinner=True)
def run_model_pipeline(X_scaled, target_cols, algo_type, model_name, params):
    """Executes the selected anomaly detection algorithms and returns the score, flag, and metric structures."""
    n_samples = len(X_scaled)
    anomalies_flag = np.zeros(n_samples, dtype=bool)
    anomaly_scores = np.zeros(n_samples)
    local_contrib = np.zeros_like(X_scaled)
    info_dict = {}

    if algo_type == "Unsupervised Feature-Space":
        contamination = params["contamination"]
        
        if model_name == "Isolation Forest":
            model = IsolationForest(
                n_estimators=params["n_estimators"],
                contamination=contamination,
                random_state=42
            )
            model.fit(X_scaled)
            preds = model.predict(X_scaled)
            anomalies_flag = (preds == -1)
            # Invert Isolation Forest's score so higher values mean more anomalous
            anomaly_scores = -model.score_samples(X_scaled)
            # Explainable AI: Standardized absolute deviations from median as fallback contribution
            medians = np.median(X_scaled, axis=0)
            stds = np.std(X_scaled, axis=0) + 1e-8
            local_contrib = np.abs(X_scaled - medians) / stds

        elif model_name == "Mahalanobis Distance":
            md, contrib = mahalanobis_distance_with_contrib(X_scaled)
            anomaly_scores = md
            thresh = np.quantile(md, 1.0 - contamination)
            anomalies_flag = (md > thresh)
            local_contrib = contrib
            info_dict["Threshold"] = thresh

        elif model_name == "PCA Reconstruction Error":
            recon_err, contrib = pca_reconstruction_with_contrib(X_scaled, n_components=params["pca_components"])
            anomaly_scores = recon_err
            thresh = np.quantile(recon_err, 1.0 - contamination)
            anomalies_flag = (recon_err > thresh)
            local_contrib = contrib
            info_dict["Threshold"] = thresh

        elif model_name == "Local Outlier Factor (LOF)":
            model = LocalOutlierFactor(
                n_neighbors=params["n_neighbors"],
                contamination=contamination,
                novelty=True
            )
            model.fit(X_scaled)
            preds = model.predict(X_scaled)
            anomalies_flag = (preds == -1)
            anomaly_scores = -model.negative_outlier_factor_
            # Explainable AI fallback
            medians = np.median(X_scaled, axis=0)
            stds = np.std(X_scaled, axis=0) + 1e-8
            local_contrib = np.abs(X_scaled - medians) / stds

    else: # Time-Series Forecasting Residuals
        if model_name == "Rolling Hampel Filter":
            anomalies_matrix = []
            scores_matrix = []
            for i in range(len(target_cols)):
                flags, scores = hampel_filter_vectorized(
                    X_scaled[:, i],
                    window_size=params["hampel_window"],
                    n_sigmas=params["hampel_sigmas"]
                )
                anomalies_matrix.append(flags)
                scores_matrix.append(scores)
            
            anomalies_flag = np.any(anomalies_matrix, axis=0)
            anomaly_scores = np.max(scores_matrix, axis=0)
            local_contrib = np.column_stack(scores_matrix)

        elif model_name == "STL Decomposition Residuals":
            anomalies_matrix = []
            scores_matrix = []
            for i in range(len(target_cols)):
                series_vals = X_scaled[:, i]
                try:
                    res = STL(series_vals, period=params["stl_period"], robust=True).fit()
                    resid = res.resid
                except Exception as e:
                    # Fallback to rolling mean residual
                    rolling_mean = pd.Series(series_vals).rolling(window=params["stl_period"], min_periods=1, center=True).mean()
                    resid = series_vals - rolling_mean
                
                std_resid = np.std(resid) + 1e-8
                z_scores = np.abs(resid - np.mean(resid)) / std_resid
                flags = z_scores > params["stl_sigma"]
                anomalies_matrix.append(flags)
                scores_matrix.append(z_scores)
            
            anomalies_flag = np.any(anomalies_matrix, axis=0)
            anomaly_scores = np.max(scores_matrix, axis=0)
            local_contrib = np.column_stack(scores_matrix)

        elif model_name == "Forecasting Model Residuals":
            split_idx = int(n_samples * params["train_ratio"])
            fc_anomalies = np.zeros(n_samples, dtype=bool)
            fc_scores = np.zeros(n_samples)
            
            fc_metrics_dict = {}
            ts_series_dict = {}
            local_contrib_matrix = np.zeros_like(X_scaled)
            
            # Predict using selected base forecaster
            if params["forecaster_choice"] == "Linear Ridge (Lags)":
                preds = fit_predict_ridge_lags(X_scaled, split_idx, lags=[1, 2, 3, 24])
            elif params["forecaster_choice"] == "Exponential Smoothing":
                preds = np.zeros_like(X_scaled)
                for i in range(X_scaled.shape[1]):
                    y_train = X_scaled[:split_idx, i]
                    try:
                        fit = ExponentialSmoothing(y_train, seasonal_periods=24, trend="add", seasonal="add").fit()
                    except:
                        try:
                            fit = ExponentialSmoothing(y_train, trend="add").fit()
                        except:
                            fit = ExponentialSmoothing(y_train).fit()
                    
                    preds_train = fit.fittedvalues
                    preds_test = fit.predict(start=split_idx, end=n_samples - 1)
                    preds[:, i] = np.concatenate([preds_train, preds_test])
            else: # Rolling Mean Predictor
                preds = np.zeros_like(X_scaled)
                for i in range(X_scaled.shape[1]):
                    series_roll = pd.Series(X_scaled[:, i])
                    preds[:, i] = series_roll.shift(1).rolling(window=12, min_periods=1).mean().fillna(method="bfill").values
            
            # Compute Errors & Detect anomalies
            errors = X_scaled - preds
            abs_errors = np.abs(errors)
            
            for i, col_name in enumerate(target_cols):
                y_test = X_scaled[split_idx:, i]
                preds_test = preds[split_idx:, i]
                test_errors = errors[split_idx:, i]
                test_abs_errors = abs_errors[split_idx:, i]
                
                # Baseline training deviation
                train_std = np.std(errors[:split_idx, i]) + 1e-8
                col_scores = abs_errors[:, i] / train_std
                
                col_flags = np.zeros(n_samples, dtype=bool)
                col_flags[split_idx:] = col_scores[split_idx:] > params["fc_sigma"]
                
                fc_anomalies = fc_anomalies | col_flags
                fc_scores = np.maximum(fc_scores, col_scores)
                local_contrib_matrix[:, i] = col_scores
                
                # Compute forecasting metrics
                mae_val = np.mean(test_abs_errors)
                rmse_val = np.sqrt(np.mean(test_errors ** 2))
                denom_y = np.where(np.abs(y_test) < 1e-5, 1e-5, y_test)
                mape_val = np.mean(np.abs(test_errors / denom_y)) * 100
                smape_val = np.mean(2.0 * test_abs_errors / (np.abs(y_test) + np.abs(preds_test) + 1e-8)) * 100
                
                # MASE metric
                naive_mae_train = np.mean(np.abs(np.diff(X_scaled[:split_idx, i]))) if split_idx > 1 else 1.0
                mase_val = mae_val / (naive_mae_train + 1e-8)
                
                # Tracking Signal (TS) = Cumulative sum of errors / MAD over test set
                cum_errors = np.cumsum(test_errors)
                mad_t = np.array([np.mean(test_abs_errors[:k+1]) for k in range(len(test_abs_errors))])
                mad_t = np.where(mad_t < 1e-5, 1e-5, mad_t)
                ts_t = cum_errors / mad_t
                
                fc_metrics_dict[col_name] = {
                    "MAE": mae_val,
                    "RMSE": rmse_val,
                    "MAPE (%)": mape_val,
                    "SMAPE (%)": smape_val,
                    "MASE": mase_val
                }
                ts_series_dict[col_name] = ts_t
                
            anomalies_flag = fc_anomalies
            anomaly_scores = fc_scores
            local_contrib = local_contrib_matrix
            
            info_dict["split_idx"] = split_idx
            info_dict["fc_metrics"] = fc_metrics_dict
            info_dict["ts_series"] = ts_series_dict

    return anomalies_flag, anomaly_scores, local_contrib, info_dict

# Run model calculations
anomalies_flag, anomaly_scores, local_contrib, info_dict = run_model_pipeline(
    X_scaled, target_cols, algo_type, model_name, model_params
)

# Append findings to the main DataFrame
df["Anomaly_Score"] = anomaly_scores
df["Is_Anomaly_Detected"] = anomalies_flag

# -------------------------------------------------------------
# TAB 3: ANOMALY DETECTION RESULTS
# -------------------------------------------------------------
with tab3:
    st.markdown("### 🔍 Model Execution & Detected Anomalies Timeline")
    
    num_anomalies = df["Is_Anomaly_Detected"].sum()
    pct_anomalies = (num_anomalies / len(df)) * 100
    
    col_res1, col_res2, col_res3 = st.columns(3)
    with col_res1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Anomalies Detected</div>
            <div class="metric-value" style="color: #ef4444;">{num_anomalies}</div>
            <div class="metric-sub">Out of {len(df)} total observations</div>
        </div>
        """, unsafe_allow_html=True)
    with col_res2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Anomaly Rate</div>
            <div class="metric-value">{pct_anomalies:.2f}%</div>
            <div class="metric-sub">Contamination/Threshold configuration</div>
        </div>
        """, unsafe_allow_html=True)
    with col_res3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Active Model</div>
            <div class="metric-value" style="color: #a855f7;">{model_name}</div>
            <div class="metric-sub">Type: {algo_type}</div>
        </div>
        """, unsafe_allow_html=True)

    # Anomaly Visualization Timeline
    st.markdown("#### Time Series Anomaly Timeline Overlay")
    selected_vis_col = st.selectbox("Feature to View in Timeline", options=target_cols, key="vis_timeline")
    
    fig_timeline = go.Figure()
    fig_timeline.add_trace(go.Scatter(
        x=df[time_col],
        y=df[selected_vis_col],
        mode="lines",
        name="Series Line",
        line=dict(color="#6366f1", width=1.5)
    ))
    
    df_anom = df[df["Is_Anomaly_Detected"] == True]
    fig_timeline.add_trace(go.Scatter(
        x=df_anom[time_col],
        y=df_anom[selected_vis_col],
        mode="markers",
        name="Detected Anomaly",
        marker=dict(color="#ef4444", size=8, symbol="circle", line=dict(color="#ffffff", width=1))
    ))
    
    if algo_type == "Time-Series Forecasting Residuals" and model_name == "Forecasting Model Residuals":
        split_date = df.iloc[info_dict["split_idx"]][time_col]
        fig_timeline.add_vline(x=split_date, line_dash="dash", line_color="#e2e8f0", annotation_text="Train/Test Split", annotation_position="top left")
        
    fig_timeline.update_layout(
        height=450,
        showlegend=True,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b")
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

    # Local Explanations - Feature Contributions
    st.markdown("#### 🔍 Feature Contribution (Local Interpretability)")
    st.caption("Understand which variables contributed the most to the anomaly status for any given point in time.")
    
    if num_anomalies > 0:
        anom_dates = df_anom[time_col].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
        selected_anom_date_str = st.selectbox("Select an Anomaly Timestamp to Inspect", options=anom_dates)
        selected_idx = df[df[time_col].dt.strftime('%Y-%m-%d %H:%M:%S') == selected_anom_date_str].index[0]
        
        # Pull cached local contribution vector for index
        deviations = local_contrib[selected_idx]
        total_dev = np.sum(deviations) if np.sum(deviations) > 0 else 1.0
        contributions = (deviations / total_dev) * 100
        
        contrib_df = pd.DataFrame({
            "Feature": target_cols,
            "Raw Value": df.loc[selected_idx, target_cols].values,
            "Historical Median": df_original[target_cols].median().values,
            "Standardized Deviation (Score)": deviations,
            "Contribution Percentage (%)": contributions
        }).sort_values(by="Contribution Percentage (%)", ascending=False)
        
        col_contrib1, col_contrib2 = st.columns([1, 1])
        with col_contrib1:
            st.markdown(f"**Anomaly Profile at {selected_anom_date_str}**")
            st.dataframe(contrib_df.style.format({
                "Raw Value": "{:.4f}",
                "Historical Median": "{:.4f}",
                "Standardized Deviation (Score)": "{:.4f}",
                "Contribution Percentage (%)": "{:.1f}%"
            }), use_container_width=True)
            
        with col_contrib2:
            fig_contrib = px.bar(
                contrib_df,
                x="Contribution Percentage (%)",
                y="Feature",
                orientation="h",
                color="Contribution Percentage (%)",
                color_continuous_scale="Reds",
                title="Variable Contribution to Anomaly Score"
            )
            fig_contrib.update_layout(
                template="plotly_dark",
                margin=dict(l=20, r=20, t=40, b=20),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_contrib, use_container_width=True)
    else:
        st.info("No anomalies detected. Adjust your threshold or contamination settings in the sidebar.")
        
    st.markdown("#### Download Results")
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode('utf-8')
    st.download_button(
        label="📥 Download Detection Results (CSV)",
        data=csv_bytes,
        file_name=f"anomaly_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# -------------------------------------------------------------
# TAB 4: EVALUATION DASHBOARD
# -------------------------------------------------------------
with tab4:
    st.markdown("### 📊 Comprehensive Evaluation & Decision Support Dashboard")
    
    # Mathematical explanations of chosen model
    st.markdown("#### 📐 Mathematical Foundations")
    
    if model_name == "Mahalanobis Distance":
        st.markdown('<div class="formula-box">', unsafe_allow_html=True)
        st.write(r"**Mahalanobis Distance ($D_M$):** Accounts for correlations between features.")
        st.latex(r"D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}")
        st.write(r"Where $\mu$ is the multivariate mean vector and $\Sigma$ is the covariance matrix. Point $x$ is flagged as an anomaly if its distance exceeds the contamination percentile threshold.")
        st.markdown('</div>', unsafe_allow_html=True)
    elif model_name == "PCA Reconstruction Error":
        st.markdown('<div class="formula-box">', unsafe_allow_html=True)
        st.write(r"**PCA Reconstruction Error:** Measures distance of the sample from the principal subspace.")
        st.latex(r"e(x) = \|x - P_k P_k^T x\|^2")
        st.write(r"Where $P_k$ is the projection matrix containing the first $k$ principal component eigenvectors. High reconstruction error indicates that the observation does not match the historical correlation structure.")
        st.markdown('</div>', unsafe_allow_html=True)
    elif model_name == "Isolation Forest":
        st.markdown('<div class="formula-box">', unsafe_allow_html=True)
        st.write(r"**Isolation Forest Path Length Score:** Measures average tree depth to isolate a point.")
        st.latex(r"s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}")
        st.write(r"Where $E(h(x))$ is the average path length of point $x$ over a forest of isolation trees, and $c(n)$ is the average path length of an unsuccessful search in a Binary Search Tree of $n$ nodes. $s \to 1$ indicates anomalies.")
        st.markdown('</div>', unsafe_allow_html=True)
    elif model_name == "Rolling Hampel Filter":
        st.markdown('<div class="formula-box">', unsafe_allow_html=True)
        st.write(r"**Rolling Hampel Filter (Median Absolute Deviation):** Robust statistical outlier detection.")
        st.latex(r"|x_t - m_t| > 3 \times 1.4826 \times \text{MAD}_t")
        st.write(r"Where $m_t$ is the rolling window median and $\text{MAD}_t = \text{median}(|x_{t-k..t+k} - m_t|)$ is the rolling median absolute deviation. The constant $1.4826$ scales the MAD to be a consistent estimator of the standard deviation.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 1. Unsupervised evaluation profiles
    st.markdown("---")
    st.markdown("#### 1. Unsupervised Statistical Evaluation")
    
    col_eval1, col_eval2 = st.columns(2)
    with col_eval1:
        # Score distribution plot
        fig_dist = px.histogram(
            df,
            x="Anomaly_Score",
            color="Is_Anomaly_Detected",
            color_discrete_map={True: "#ef4444", False: "#6366f1"},
            nbins=50,
            title="Distribution of Anomaly Scores"
        )
        if "Threshold" in info_dict:
            fig_dist.add_vline(x=info_dict["Threshold"], line_dash="dash", line_color="#ef4444", annotation_text="Cutoff Threshold", annotation_position="top right")
            
        fig_dist.update_layout(
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend_title="Anomaly?"
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
    with col_eval2:
        # Silhouette Score / Clustering Separation
        if num_anomalies > 1 and num_anomalies < len(df) - 1:
            try:
                sil_score = silhouette_score(X_scaled, df["Is_Anomaly_Detected"])
                st.markdown(f"""
                <div class="metric-card" style="margin-top: 20px;">
                    <div class="metric-title">Silhouette Separation Score</div>
                    <div class="metric-value" style="color: #10b981;">{sil_score:.4f}</div>
                    <div class="metric-sub">Measures clustering separation. Range: [-1, 1]. Positive values indicate normal and anomaly groups are well-separated in feature space.</div>
                </div>
                """, unsafe_allow_html=True)
            except:
                st.warning("Could not calculate Silhouette score due to numerical dimensions.")
        else:
            st.info("Silhouette score requires at least 2 anomaly points and 2 normal points.")

        # Boxplots comparing distributions
        st.markdown("**Feature Distributions: Normal vs. Anomaly Periods**")
        box_feature = st.selectbox("Select Feature for Distribution Comparison", options=target_cols, key="box_feat")
        fig_box = px.box(
            df,
            x="Is_Anomaly_Detected",
            y=box_feature,
            color="Is_Anomaly_Detected",
            color_discrete_map={True: "#ef4444", False: "#6366f1"},
            labels={"Is_Anomaly_Detected": "Is Anomaly?"}
        )
        fig_box.update_layout(
            template="plotly_dark",
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # 2. Supervised Evaluation (Validation Metrics)
    if has_gt and gt_col in df.columns:
        st.markdown("---")
        st.markdown("#### 2. Supervised Validation Metrics (Against Ground Truth)")
        
        y_true = df[gt_col].astype(int).values
        y_pred = df["Is_Anomaly_Detected"].astype(int).values
        
        try:
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            norm_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min() + 1e-8)
            auc = roc_auc_score(y_true, norm_scores)
        except Exception as e:
            precision = recall = f1 = auc = 0.0
            st.warning(f"Failed to calculate validation metrics: {e}")
            
        col_sup1, col_sup2, col_sup3, col_sup4 = st.columns(4)
        with col_sup1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Precision</div>
                <div class="metric-value">{precision:.4f}</div>
                <div class="metric-sub">Proportion of true positive detections among all flagged points.</div>
            </div>
            """, unsafe_allow_html=True)
        with col_sup2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Recall (Sensitivity)</div>
                <div class="metric-value">{recall:.4f}</div>
                <div class="metric-sub">Proportion of actual anomalies successfully identified.</div>
            </div>
            """, unsafe_allow_html=True)
        with col_sup3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">F1-Score</div>
                <div class="metric-value" style="color: #f59e0b;">{f1:.4f}</div>
                <div class="metric-sub">Harmonic mean of precision and recall. Best balance.</div>
            </div>
            """, unsafe_allow_html=True)
        with col_sup4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">ROC-AUC Score</div>
                <div class="metric-value" style="color: #10b981;">{auc:.4f}</div>
                <div class="metric-sub">Area Under ROC Curve. Reflects the ranking performance of anomaly scores.</div>
            </div>
            """, unsafe_allow_html=True)
            
        col_curve1, col_curve2 = st.columns(2)
        with col_curve1:
            # Confusion Matrix Plot
            cm = confusion_matrix(y_true, y_pred)
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                x=["Predicted Normal", "Predicted Anomaly"],
                y=["Actual Normal", "Actual Anomaly"],
                color_continuous_scale="Blues",
                title="Confusion Matrix"
            )
            fig_cm.update_layout(
                template="plotly_dark",
                margin=dict(l=20, r=20, t=40, b=20),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with col_curve2:
            # ROC & PR curves
            curve_type = st.radio("Select Curve to View", ["ROC Curve", "Precision-Recall Curve"], horizontal=True)
            if curve_type == "ROC Curve":
                fpr, tpr, _ = roc_curve(y_true, norm_scores)
                fig_curve = px.line(x=fpr, y=tpr, labels={"x": "False Positive Rate", "y": "True Positive Rate"}, title="Receiver Operating Characteristic (ROC) Curve")
                fig_curve.add_shape(type="line", line=dict(dash="dash", color="#64748b"), x0=0, x1=1, y0=0, y1=1)
            else:
                prec, rec, _ = precision_recall_curve(y_true, norm_scores)
                fig_curve = px.line(x=rec, y=prec, labels={"x": "Recall", "y": "Precision"}, title="Precision-Recall Curve")
                
            fig_curve.update_layout(
                template="plotly_dark",
                margin=dict(l=20, r=20, t=40, b=20),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_curve, use_container_width=True)

    # 3. Forecasting-based Metrics (from 수업내용.md)
    if algo_type == "Time-Series Forecasting Residuals" and model_name == "Forecasting Model Residuals":
        st.markdown("---")
        st.markdown("#### 3. Forecasting Evaluation Metrics & Drift Diagnostics")
        st.markdown("These metrics reflect the forecasting accuracy of the base model over the Test split, matching the statistics from `수업내용.md`.")
        
        # Display Metrics Table
        fc_df = pd.DataFrame(info_dict["fc_metrics"]).T
        st.table(fc_df.style.format("{:.4f}"))
        
        # Tracking Signal (TS) Plot
        st.markdown("##### 📈 Tracking Signal (TS) Over Time")
        st.markdown("""
        The Tracking Signal indicates whether the forecasting model is systematically biased (under-predicting or over-predicting).
        $$TS_t = \\frac{\\sum_{i=1}^t e_i}{MAD_t}$$
        If the Tracking Signal leaves the range of $[-4, 4]$, the model is considered biased or indicates structural drift in the time series.
        """)
        
        selected_ts_col = st.selectbox("Select Feature to View Tracking Signal", options=target_cols, key="ts_feat")
        ts_values = info_dict["ts_series"][selected_ts_col]
        test_dates = df.iloc[info_dict["split_idx"]:][time_col]
        
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=test_dates,
            y=ts_values,
            name="Tracking Signal",
            line=dict(color="#38bdf8", width=2)
        ))
        
        # Add bounds +/- 4
        fig_ts.add_hline(y=4, line_dash="dash", line_color="#ef4444", line_width=1.5, annotation_text="Upper Bound (+4)", annotation_position="top right")
        fig_ts.add_hline(y=-4, line_dash="dash", line_color="#ef4444", line_width=1.5, annotation_text="Lower Bound (-4)", annotation_position="bottom right")
        fig_ts.add_hline(y=0, line_dash="solid", line_color="#475569", line_width=1)
        
        fig_ts.update_layout(
            template="plotly_dark",
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Tracking Signal (TS)",
            xaxis=dict(gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b")
        )
        st.plotly_chart(fig_ts, use_container_width=True)
