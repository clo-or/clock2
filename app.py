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
    page_title="시계열 이상탐지 플랫폼",
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
st.markdown('<div class="main-title">⏰ 다변량 시계열 이상탐지 플랫폼</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">임의의 시계열 데이터를 업로드하여 통계적 특성을 자동으로 분석하고, 고급 이상탐지 모델을 적용할 수 있습니다.</div>', unsafe_allow_html=True)

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
st.sidebar.markdown("### 📂 데이터 소스 및 변수 설정")

data_source_map = {
    "산업용 센서 데이터 (가상 샘플)": "Industrial Sensor (Synthetic Sample)",
    "사용자 정의 CSV 업로드": "Upload Custom CSV"
}
data_source_ui = st.sidebar.selectbox(
    "데이터 소스 선택",
    options=list(data_source_map.keys())
)
data_source = data_source_map[data_source_ui]

if data_source == "Upload Custom CSV":
    uploaded_file = st.sidebar.file_uploader("CSV 파일 업로드", type=["csv"])
    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            st.sidebar.success("파일이 성공적으로 업로드되었습니다!")
        except Exception as e:
            st.sidebar.error(f"파일 읽기 오류: {e}")
            raw_df = generate_synthetic_data()
    else:
        st.sidebar.info("사용자 정의 CSV 파일이 업로드되기 전까지 가상 샘플 데이터를 사용합니다.")
        raw_df = generate_synthetic_data()
else:
    raw_df = generate_synthetic_data()

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ 컬럼 선택")
columns = list(raw_df.columns)

# Auto-detect timestamp column
date_cols = [col for col in columns if "time" in col.lower() or "date" in col.lower()]
default_time_idx = columns.index(date_cols[0]) if date_cols else 0
time_col = st.sidebar.selectbox("타임스탬프 컬럼", options=columns, index=default_time_idx)

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
    "대상 피처 (이상탐지용)",
    options=num_cols,
    default=num_cols[:min(3, len(num_cols))]
)

has_gt = st.sidebar.checkbox("실제 이상(Ground Truth) 레이블이 있습니까?", value=(default_gt is not None))
if has_gt:
    gt_options = [col for col in columns if col != time_col]
    if not gt_options:
        st.sidebar.warning("실제 이상(Ground Truth)으로 설정할 컬럼이 없습니다.")
        gt_col = None
    else:
        default_idx = gt_options.index(default_gt) if (default_gt in gt_options) else 0
        gt_col = st.sidebar.selectbox(
            "실제 이상(Ground Truth) 컬럼",
            options=gt_options,
            index=default_idx
        )
else:
    gt_col = None

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧹 데이터 전처리")

missing_impute_map = {
    "선형 보간 (Linear Interpolation)": "Linear Interpolation",
    "직전 값으로 대체 (Forward Fill)": "Forward Fill",
    "직후 값으로 대체 (Backward Fill)": "Backward Fill",
    "대체 없음 (결측치 제거)": "No Imputation (Drop NA)"
}
missing_impute_ui = st.sidebar.selectbox(
    "결측치 대체 방법",
    options=list(missing_impute_map.keys())
)
missing_impute = missing_impute_map[missing_impute_ui]

scaling_method_map = {
    "StandardScaler (Z-점수)": "StandardScaler (z-score)",
    "MinMaxScaler (0~1)": "MinMaxScaler (0-1)",
    "적용 안 함": "None"
}
scaling_method_ui = st.sidebar.selectbox(
    "피처 스케일링",
    options=list(scaling_method_map.keys())
)
scaling_method = scaling_method_map[scaling_method_ui]

denoising_method_map = {
    "적용 안 함": "None",
    "단순 이동 평균 (Simple Moving Average)": "Simple Moving Average",
    "차분 (Differencing, t - (t-1))": "Differencing (t - (t-1))"
}
denoising_method_ui = st.sidebar.selectbox(
    "노이즈 제거 및 평활화 방법",
    options=list(denoising_method_map.keys())
)
denoising_method = denoising_method_map[denoising_method_ui]

if denoising_method == "Simple Moving Average":
    ma_window = st.sidebar.slider("이동 평균 윈도우 크기", min_value=3, max_value=31, value=5, step=2)
else:
    ma_window = 1

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 이상탐지 모델 선택")

algo_type_map = {
    "비지도 학습 기반 (피처 공간)": "Unsupervised Feature-Space",
    "시계열 예측 잔차 기반": "Time-Series Forecasting Residuals"
}
algo_type_ui = st.sidebar.selectbox(
    "이상탐지 모델 유형",
    options=list(algo_type_map.keys())
)
algo_type = algo_type_map[algo_type_ui]

# Initialize parameter configs
model_params = {}

if algo_type == "Unsupervised Feature-Space":
    model_name_map = {
        "아이솔레이션 포레스트 (Isolation Forest)": "Isolation Forest",
        "마할라노비스 거리 (Mahalanobis Distance)": "Mahalanobis Distance",
        "PCA 재구성 오차 (PCA Reconstruction Error)": "PCA Reconstruction Error",
        "LOF (Local Outlier Factor)": "Local Outlier Factor (LOF)"
    }
    model_name_ui = st.sidebar.selectbox(
        "모델 선택",
        options=list(model_name_map.keys())
    )
    model_name = model_name_map[model_name_ui]
    
    contamination = st.sidebar.slider("이상치 비율 설정 (%)", min_value=0.1, max_value=20.0, value=2.0, step=0.1) / 100.0
    model_params["contamination"] = contamination
    
    if model_name == "Isolation Forest":
        n_estimators = st.sidebar.slider("트리 개수", min_value=50, max_value=300, value=100, step=50)
        model_params["n_estimators"] = n_estimators
    elif model_name == "PCA Reconstruction Error":
        pca_components = st.sidebar.slider("주성분 개수", min_value=1, max_value=max(1, len(target_cols)), value=min(2, len(target_cols)))
        model_params["pca_components"] = pca_components
    elif model_name == "Local Outlier Factor (LOF)":
        n_neighbors = st.sidebar.slider("이웃 개수", min_value=5, max_value=50, value=20)
        model_params["n_neighbors"] = n_neighbors
else:
    model_name_map = {
        "롤링 햄펠 필터 (Rolling Hampel Filter)": "Rolling Hampel Filter",
        "STL 분해 잔차 (STL Decomposition)": "STL Decomposition Residuals",
        "예측 모델 잔차 (Forecasting Model Residuals)": "Forecasting Model Residuals"
    }
    model_name_ui = st.sidebar.selectbox(
        "모델 선택",
        options=list(model_name_map.keys())
    )
    model_name = model_name_map[model_name_ui]
    
    if model_name == "Rolling Hampel Filter":
        hampel_window = st.sidebar.slider("햄펠 윈도우 크기", min_value=5, max_value=51, value=15, step=2)
        hampel_sigmas = st.sidebar.slider("햄펠 임계값 (Sigma)", min_value=1.5, max_value=5.0, value=3.0, step=0.1)
        model_params["hampel_window"] = hampel_window
        model_params["hampel_sigmas"] = hampel_sigmas
    elif model_name == "STL Decomposition Residuals":
        stl_period = st.sidebar.slider("계절성 주기 (Seasonal Period)", min_value=4, max_value=168, value=24, step=1)
        stl_sigma = st.sidebar.slider("잔차 임계값 (Sigma)", min_value=1.5, max_value=5.0, value=3.0, step=0.1)
        model_params["stl_period"] = stl_period
        model_params["stl_sigma"] = stl_sigma
    elif model_name == "Forecasting Model Residuals":
        forecaster_choice_map = {
            "선형 릿지 회귀 (Ridge with Lags)": "Linear Ridge (Lags)",
            "지수 평활법 (Exponential Smoothing)": "Exponential Smoothing",
            "롤링 평균 예측기 (Rolling Mean Predictor)": "Rolling Mean Predictor"
        }
        forecaster_choice_ui = st.sidebar.selectbox(
            "기본 예측 모델",
            options=list(forecaster_choice_map.keys())
        )
        forecaster_choice = forecaster_choice_map[forecaster_choice_ui]
        
        fc_sigma = st.sidebar.slider("잔차 임계값 (Sigma)", min_value=1.5, max_value=5.0, value=3.0, step=0.1)
        train_ratio = st.sidebar.slider("학습 데이터 분할 비율", min_value=0.5, max_value=0.9, value=0.7, step=0.05)
        model_params["forecaster_choice"] = forecaster_choice
        model_params["fc_sigma"] = fc_sigma
        model_params["train_ratio"] = train_ratio

if not target_cols:
    st.error("이상탐지를 수행하려면 사이드바에서 최소 하나 이상의 대상 피처를 선택해 주세요.")
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
    "📂 데이터 개요 및 전처리",
    "📈 탐색적 시계열 분석 (EDA)",
    "🔍 이상탐지 결과 분석",
    "📊 모델 평가 및 진단 대시보드"
])

# -------------------------------------------------------------
# TAB 1: DATA OVERVIEW & PREPROCESSING
# -------------------------------------------------------------
with tab1:
    st.markdown("### 📊 데이터셋 속성 및 전처리 확인")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">총 레코드 수</div>
            <div class="metric-value">{len(df)}</div>
            <div class="metric-sub">{df[time_col].min().strftime('%Y-%m-%d')}부터 {df[time_col].max().strftime('%Y-%m-%d')}까지의 기간 데이터</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">선택된 대상 피처</div>
            <div class="metric-value">{len(target_cols)}</div>
            <div class="metric-sub">{', '.join(target_cols)}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        na_count = raw_df[target_cols].isna().sum().sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">결측치 수 (원본 데이터)</div>
            <div class="metric-value">{na_count}</div>
            <div class="metric-sub">처리 방법: {missing_impute_ui}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### 데이터셋 샘플 (상위 5행)")
    st.dataframe(df.head(), use_container_width=True)
    
    st.markdown("#### 원본 데이터 vs. 전처리 데이터 시각화 비교")
    st.caption("사이드바에서 설정한 결측치 대체, 평활화, 스케일링 설정에 따른 데이터 변환 효과를 비교해 보세요.")
    
    feature_to_plot = st.selectbox("시각화할 피처 선택", options=target_cols)
    
    fig_prep = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=["원본 시계열", "전처리된 시계열"])
    fig_prep.add_trace(
        go.Scatter(x=df_original[time_col], y=df_original[feature_to_plot], name="원본 데이터", line=dict(color="#94a3b8")),
        row=1, col=1
    )
    fig_prep.add_trace(
        go.Scatter(x=df[time_col], y=df_scaled[feature_to_plot], name="전처리 데이터", line=dict(color="#38bdf8")),
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
    st.markdown("### 📈 시계열 통계 및 구조적 분석")
    
    st.markdown("#### 1. 정상성(Stationarity) 검정: ADF (Augmented Dickey-Fuller) 테스트")
    st.markdown("""
    ADF 검정은 시계열 데이터에 단위근이 존재하는지(즉, 비정상성 시계열인지) 확인하는 통계적 검정입니다.
    *   **귀무가설 ($H_0$)**: 시계열이 단위근을 가집니다 (비정상성 시계열).
    *   **대립가설 ($H_1$)**: 시계열이 정상성을 띱니다 ($p\\text{-value} < 0.05$).
    """)
    
    adf_results = []
    for col in target_cols:
        series_clean = df[col].dropna()
        if series_clean.std() == 0:
            adf_results.append({
                "피처": col,
                "ADF 통계량": np.nan,
                "p-value": np.nan,
                "사용된 시차 (Lags)": 0,
                "정상성 만족 여부": "상수값 (변동 없음)"
            })
            continue
            
        try:
            res = adfuller(series_clean, regression="c", autolag="AIC")
            adf_stat = res[0]
            p_val = res[1]
            lags = res[2]
            stationary = "✅ 정상 (p < 0.05)" if p_val < 0.05 else "❌ 비정상 (정상성 미만족)"
            adf_results.append({
                "피처": col,
                "ADF 통계량": f"{adf_stat:.4f}",
                "p-value": f"{p_val:.4f}" if p_val >= 0.0001 else "< 0.0001",
                "사용된 시차 (Lags)": lags,
                "정상성 만족 여부": stationary
            })
        except Exception as e:
            adf_results.append({
                "피처": col,
                "ADF 통계량": "실패",
                "p-value": "실패",
                "사용된 시차 (Lags)": 0,
                "정상성 만족 여부": f"오류: {e}"
            })
            
    st.table(pd.DataFrame(adf_results))
    
    st.markdown("#### 2. 자기상관 및 부분자기상관 분석 (ACF / PACF)")
    st.caption("대상 시계열의 시간적 의존성, 시차(Lags) 및 계절성 패턴을 식별합니다.")
    
    acf_feature = st.selectbox("ACF/PACF 분석 대상 피처 선택", options=target_cols, key="acf_feat")
    n_lags = st.slider("시차(Lags) 개수", min_value=10, max_value=100, value=40, step=5)
    
    series_acf = df[acf_feature].values
    lag_acf = acf(series_acf, nlags=n_lags, fft=True)
    lag_pacf = pacf(series_acf, nlags=n_lags, method="yw")
    
    fig_corr = make_subplots(rows=1, cols=2, subplot_titles=["자기상관함수 (ACF)", "부분자기상관함수 (PACF)"])
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

    st.markdown("#### 3. 다변량 피처 간 상관관계 분석")
    if len(target_cols) > 1:
        corr_matrix = df[target_cols].corr()
        fig_corr_mat = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title="피어슨 상관계수 히트맵 (Pearson Correlation)"
        )
        fig_corr_mat.update_layout(
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_corr_mat, use_container_width=True)
    else:
        st.info("상관관계 행렬을 시각화하려면 최소 2개 이상의 대상 피처를 선택해야 합니다.")

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
# Mapped names for display
model_name_kr = model_name
for k, v in model_name_map.items():
    if v == model_name:
        model_name_kr = k
        break

algo_type_kr = algo_type
for k, v in algo_type_map.items():
    if v == algo_type:
        algo_type_kr = k
        break

with tab3:
    st.markdown("### 🔍 모델 실행 및 탐지된 이상치 타임라인")
    
    num_anomalies = df["Is_Anomaly_Detected"].sum()
    pct_anomalies = (num_anomalies / len(df)) * 100
    
    col_res1, col_res2, col_res3 = st.columns(3)
    with col_res1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">탐지된 이상치</div>
            <div class="metric-value" style="color: #ef4444;">{num_anomalies} 건</div>
            <div class="metric-sub">총 {len(df)}개 관측치 중</div>
        </div>
        """, unsafe_allow_html=True)
    with col_res2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">이상 비율</div>
            <div class="metric-value">{pct_anomalies:.2f}%</div>
            <div class="metric-sub">설정된 이상치 비율/임계값 기준</div>
        </div>
        """, unsafe_allow_html=True)
    with col_res3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">활성화된 모델</div>
            <div class="metric-value" style="color: #a855f7;">{model_name_kr}</div>
            <div class="metric-sub">유형: {algo_type_kr}</div>
        </div>
        """, unsafe_allow_html=True)

    # Anomaly Visualization Timeline
    st.markdown("#### 시계열 이상탐지 타임라인 오버레이")
    selected_vis_col = st.selectbox("타임라인에서 볼 피처 선택", options=target_cols, key="vis_timeline")
    
    fig_timeline = go.Figure()
    fig_timeline.add_trace(go.Scatter(
        x=df[time_col],
        y=df[selected_vis_col],
        mode="lines",
        name="시계열 데이터",
        line=dict(color="#6366f1", width=1.5)
    ))
    
    df_anom = df[df["Is_Anomaly_Detected"] == True]
    fig_timeline.add_trace(go.Scatter(
        x=df_anom[time_col],
        y=df_anom[selected_vis_col],
        mode="markers",
        name="탐지된 이상치",
        marker=dict(color="#ef4444", size=8, symbol="circle", line=dict(color="#ffffff", width=1))
    ))
    
    if algo_type == "Time-Series Forecasting Residuals" and model_name == "Forecasting Model Residuals":
        split_date = df.iloc[info_dict["split_idx"]][time_col]
        fig_timeline.add_vline(x=split_date, line_dash="dash", line_color="#e2e8f0", annotation_text="학습/테스트 데이터 분할선", annotation_position="top left")
        
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
    st.markdown("#### 🔍 피처 기여도 (설명 가능한 AI / 로컬 분석)")
    st.caption("특정 시점에 탐지된 이상치에 대해 어떤 변수(피처)가 가장 크게 기여했는지 개별 기여도를 분석합니다.")
    
    if num_anomalies > 0:
        anom_dates = df_anom[time_col].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
        selected_anom_date_str = st.selectbox("상세 분석할 이상치 발생 시각 선택", options=anom_dates)
        selected_idx = df[df[time_col].dt.strftime('%Y-%m-%d %H:%M:%S') == selected_anom_date_str].index[0]
        
        # Pull cached local contribution vector for index
        deviations = local_contrib[selected_idx]
        total_dev = np.sum(deviations) if np.sum(deviations) > 0 else 1.0
        contributions = (deviations / total_dev) * 100
        
        contrib_df = pd.DataFrame({
            "피처 (Feature)": target_cols,
            "실제값 (Raw Value)": df.loc[selected_idx, target_cols].values,
            "역사적 중앙값 (Median)": df_original[target_cols].median().values,
            "표준화된 편차 (Score)": deviations,
            "기여 비율 (%)": contributions
        }).sort_values(by="기여 비율 (%)", ascending=False)
        
        col_contrib1, col_contrib2 = st.columns([1, 1])
        with col_contrib1:
            st.markdown(f"**{selected_anom_date_str} 시점의 이상치 분석 프로필**")
            st.dataframe(contrib_df.style.format({
                "실제값 (Raw Value)": "{:.4f}",
                "역사적 중앙값 (Median)": "{:.4f}",
                "표준화된 편차 (Score)": "{:.4f}",
                "기여 비율 (%)": "{:.1f}%"
            }), use_container_width=True)
            
        with col_contrib2:
            fig_contrib = px.bar(
                contrib_df,
                x="기여 비율 (%)",
                y="피처 (Feature)",
                orientation="h",
                color="기여 비율 (%)",
                color_continuous_scale="Reds",
                title="이상치 점수에 대한 피처별 기여도"
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
        st.info("탐지된 이상치가 없습니다. 사이드바에서 임계값이나 이상치 비율 설정을 조정해 주세요.")
        
    st.markdown("#### 탐지 결과 다운로드")
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode('utf-8')
    st.download_button(
        label="📥 이상탐지 결과 다운로드 (CSV)",
        data=csv_bytes,
        file_name=f"anomaly_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# -------------------------------------------------------------
# TAB 4: EVALUATION DASHBOARD
# -------------------------------------------------------------
with tab4:
    st.markdown("### 📊 종합 모델 평가 및 진단 대시보드")
    
    # Mathematical explanations of chosen model
    st.markdown("#### 📐 적용된 모델의 수학적 원리")
    
    if model_name == "Mahalanobis Distance":
        st.markdown(r"""
        <div class="formula-box">
        
        **마할라노비스 거리 ($D_M$):** 피처 간의 상관관계를 반영하여 이상을 탐지합니다.
        
        $$D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$$
        
        여기서 $\mu$는 다변량 평균 벡터이고, $\Sigma$는 공분산 행렬입니다. 마할라노비스 거리가 설정된 이상치 비율의 분위수 임계값을 초과하는 시점이 이상치로 판단됩니다.
        
        </div>
        """, unsafe_allow_html=True)
    elif model_name == "PCA Reconstruction Error":
        st.markdown(r"""
        <div class="formula-box">
        
        **PCA 재구성 오차 (Reconstruction Error):** 주성분 공간(Subspace)으로부터 샘플이 떨어진 거리를 측정합니다.
        
        $$e(x) = \|x - P_k P_k^T x\|^2$$
        
        여기서 $P_k$는 상위 $k$개의 주성분 고유벡터를 포함하는 투영 행렬입니다. 재구성 오차가 높다는 것은 해당 시점의 관측치가 과거의 상관관계 구조를 따르지 않음을 나타냅니다.
        
        </div>
        """, unsafe_allow_html=True)
    elif model_name == "Isolation Forest":
        st.markdown(r"""
        <div class="formula-box">
        
        **아이솔레이션 포레스트 경로 길이 점수:** 특정 데이터를 고립시키기 위해 필요한 평균 트리 깊이를 기준으로 이상 점수를 계산합니다.
        
        $$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$
        
        여기서 $E(h(x))$는 생성된 의사결정 나무(Tree)들에서 데이터 $x$를 고립시키기 위한 평균 경로 길이이며, $c(n)$은 $n$개 노드로 구성된 이진 탐색 트리에서 탐색 실패 시의 평균 경로 길이입니다. 점수 $s$가 1에 가까울수록 이상치일 확률이 높습니다.
        
        </div>
        """, unsafe_allow_html=True)
    elif model_name == "Rolling Hampel Filter":
        st.markdown(r"""
        <div class="formula-box">
        
        **롤링 햄펠 필터 (Median Absolute Deviation):** 중앙값(Median)과 MAD를 이용한 로버스트(강건한) 통계적 아웃라이어 탐지 기법입니다.
        
        $$|x_t - m_t| > 3 \times 1.4826 \times \text{MAD}_t$$
        
        여기서 $m_t$는 롤링 윈도우 중앙값이며, $\text{MAD}_t = \text{median}(|x_{t-k..t+k} - m_t|)$는 롤링 중앙값 절대 편차입니다. 상수 1.4826은 정규분포에서 MAD가 표준편차의 일치추정량이 되도록 스케일을 조정해 주는 계수입니다.
        
        </div>
        """, unsafe_allow_html=True)

    # 1. Unsupervised evaluation profiles
    st.markdown("---")
    st.markdown("#### 1. 비지도 학습 통계 평가")
    
    col_eval1, col_eval2 = st.columns(2)
    with col_eval1:
        # Score distribution plot
        fig_dist = px.histogram(
            df,
            x="Anomaly_Score",
            color="Is_Anomaly_Detected",
            color_discrete_map={True: "#ef4444", False: "#6366f1"},
            nbins=50,
            title="이상치 점수 분포"
        )
        if "Threshold" in info_dict:
            fig_dist.add_vline(x=info_dict["Threshold"], line_dash="dash", line_color="#ef4444", annotation_text="판단 임계값", annotation_position="top right")
            
        fig_dist.update_layout(
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend_title="이상 여부"
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
    with col_eval2:
        # Silhouette Score / Clustering Separation
        if num_anomalies > 1 and num_anomalies < len(df) - 1:
            try:
                sil_score = silhouette_score(X_scaled, df["Is_Anomaly_Detected"])
                st.markdown(f"""
                <div class="metric-card" style="margin-top: 20px;">
                    <div class="metric-title">실루엣 분리도 점수 (Silhouette Score)</div>
                    <div class="metric-value" style="color: #10b981;">{sil_score:.4f}</div>
                    <div class="metric-sub">정상 그룹과 이상 그룹이 피처 공간에서 얼마나 잘 분리되어 있는지 측정합니다. 범위는 [-1, 1]이며, 양수 값이 크고 1에 가까울수록 두 그룹이 명확히 구분된다는 것을 의미합니다.</div>
                </div>
                """, unsafe_allow_html=True)
            except:
                st.warning("데이터 차원 문제로 인해 실루엣 점수를 계산할 수 없습니다.")
        else:
            st.info("실루엣 점수를 계산하려면 최소 2개 이상의 이상치와 2개 이상의 정상 데이터가 존재해야 합니다.")

        # Boxplots comparing distributions
        st.markdown("**정상 vs 이상 시점의 피처별 분포 비교**")
        box_feature = st.selectbox("분포를 비교할 피처 선택", options=target_cols, key="box_feat")
        
        df_box = df.copy()
        df_box["이상 여부"] = df_box["Is_Anomaly_Detected"].map({True: "이상치 (Anomaly)", False: "정상 (Normal)"})
        
        fig_box = px.box(
            df_box,
            x="이상 여부",
            y=box_feature,
            color="이상 여부",
            color_discrete_map={"이상치 (Anomaly)": "#ef4444", "정상 (Normal)": "#6366f1"},
            labels={"이상 여부": "상태 구분"}
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
        st.markdown("#### 2. 지도학습 기반 검증 지표 (실제 레이블 대비 평가)")
        
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
            st.warning(f"검증 지표 계산 실패: {e}")
            
        col_sup1, col_sup2, col_sup3, col_sup4 = st.columns(4)
        with col_sup1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">정밀도 (Precision)</div>
                <div class="metric-value">{precision:.4f}</div>
                <div class="metric-sub">모델이 이상이라고 분류한 것 중 실제로 이상인 비율입니다.</div>
            </div>
            """, unsafe_allow_html=True)
        with col_sup2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">재현율 (Recall / 민감도)</div>
                <div class="metric-value">{recall:.4f}</div>
                <div class="metric-sub">실제 이상치 중 모델이 성공적으로 찾아낸 비율입니다.</div>
            </div>
            """, unsafe_allow_html=True)
        with col_sup3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">F1-점수 (F1-Score)</div>
                <div class="metric-value" style="color: #f59e0b;">{f1:.4f}</div>
                <div class="metric-sub">정밀도와 재현율의 조화평균으로, 불균형 데이터셋에서 가장 균형 잡힌 평가지표입니다.</div>
            </div>
            """, unsafe_allow_html=True)
        with col_sup4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">ROC-AUC 점수</div>
                <div class="metric-value" style="color: #10b981;">{auc:.4f}</div>
                <div class="metric-sub">ROC 곡선 아래의 면적입니다. 이상치 점수의 랭킹 및 분류 판별 성능을 종합적으로 나타냅니다.</div>
            </div>
            """, unsafe_allow_html=True)
            
        col_curve1, col_curve2 = st.columns(2)
        with col_curve1:
            # Confusion Matrix Plot
            cm = confusion_matrix(y_true, y_pred)
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                x=["정상 예측 (Normal)", "이상 예측 (Anomaly)"],
                y=["실제 정상 (Normal)", "실제 이상 (Anomaly)"],
                color_continuous_scale="Blues",
                title="혼동 행렬 (Confusion Matrix)"
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
            curve_type = st.radio("시각화할 곡선 선택", ["ROC 곡선 (ROC Curve)", "정밀도-재현율 곡선 (Precision-Recall Curve)"], horizontal=True)
            if curve_type == "ROC 곡선 (ROC Curve)":
                fpr, tpr, _ = roc_curve(y_true, norm_scores)
                fig_curve = px.line(x=fpr, y=tpr, labels={"x": "위양성률 (False Positive Rate)", "y": "진양성률 (True Positive Rate)"}, title="ROC 곡선 (Receiver Operating Characteristic)")
                fig_curve.add_shape(type="line", line=dict(dash="dash", color="#64748b"), x0=0, x1=1, y0=0, y1=1)
            else:
                prec, rec, _ = precision_recall_curve(y_true, norm_scores)
                fig_curve = px.line(x=rec, y=prec, labels={"x": "재현율 (Recall)", "y": "정밀도 (Precision)"}, title="정밀도-재현율 곡선 (Precision-Recall Curve)")
                
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
        st.markdown("#### 3. 예측 기반 모델 평가지표 및 개념 드리프트 진단")
        st.markdown("이 지표들은 학습/테스트 분할 중 테스트 데이터 세트에 대한 기본 시계열 예측 모델의 정확도를 나타내며, 수업내용.md에 제시된 통계와 일치합니다.")
        
        # Display Metrics Table
        fc_df = pd.DataFrame(info_dict["fc_metrics"]).T
        st.table(fc_df.style.format("{:.4f}"))
        
        # Tracking Signal (TS) Plot
        st.markdown("##### 📈 시간 흐름에 따른 트래킹 시그널 (Tracking Signal)")
        st.markdown("""
        트래킹 시그널(Tracking Signal)은 예측 모델이 지속적으로 한쪽 방향(과소 예측 또는 과대 예측)으로 편향되는지를 진단하는 지표입니다.
        $$TS_t = \\frac{\\sum_{i=1}^t e_i}{MAD_t}$$
        트래킹 시그널이 $[-4, 4]$ 범위를 벗어나면 모델의 예측 편향이 심각하거나 시계열 데이터의 구조적 드리프트(Drift)가 발생했음을 의미합니다.
        """)
        
        selected_ts_col = st.selectbox("트래킹 시그널을 확인할 피처 선택", options=target_cols, key="ts_feat")
        ts_values = info_dict["ts_series"][selected_ts_col]
        test_dates = df.iloc[info_dict["split_idx"]:][time_col]
        
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=test_dates,
            y=ts_values,
            name="트래킹 시그널 (TS)",
            line=dict(color="#38bdf8", width=2)
        ))
        
        # Add bounds +/- 4
        fig_ts.add_hline(y=4, line_dash="dash", line_color="#ef4444", line_width=1.5, annotation_text="상한선 (+4)", annotation_position="top right")
        fig_ts.add_hline(y=-4, line_dash="dash", line_color="#ef4444", line_width=1.5, annotation_text="하한선 (-4)", annotation_position="bottom right")
        fig_ts.add_hline(y=0, line_dash="solid", line_color="#475569", line_width=1)
        
        fig_ts.update_layout(
            template="plotly_dark",
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis_title="트래킹 시그널 (TS)",
            xaxis=dict(gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b")
        )
        st.plotly_chart(fig_ts, use_container_width=True)
