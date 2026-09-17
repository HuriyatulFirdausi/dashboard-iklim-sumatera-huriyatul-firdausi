import streamlit as st


# ============================================================
# PENYIMPANAN HASIL MODEL SECARA PERMANEN
# ============================================================
# Catatan:
# File hasil disimpan di folder hasil_model/.
# Ini membuat hasil tetap tersedia ketika browser di-refresh
# selama instance aplikasi masih menggunakan filesystem yang sama.
# Untuk bertahan setelah redeploy/restart Streamlit Cloud,
# folder hasil_model perlu ikut di-commit ke repository atau
# menggunakan penyimpanan eksternal.
# ============================================================

RESULT_DIR = Path("hasil_model")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

def _safe_result_name(text):
    return (
        str(text)
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

def _result_folder(station, parameter):
    folder = (
        RESULT_DIR
        / _safe_result_name(station)
        / _safe_result_name(parameter)
    )
    folder.mkdir(parents=True, exist_ok=True)
    return folder

def save_persistent_result(
    station,
    parameter,
    metrics,
    evaluation,
    forecast,
    annual,
    features
):
    folder = _result_folder(station, parameter)

    pd.DataFrame([{
        "RMSE": metrics["RMSE"],
        "MAE": metrics["MAE"],
        "R2": metrics["R2"]
    }]).to_csv(folder / "metrics.csv", index=False)

    evaluation.to_csv(folder / "evaluasi.csv", index=False)
    forecast.to_csv(folder / "forecast_bulanan.csv", index=False)
    annual.to_csv(folder / "forecast_tahunan.csv", index=False)

    pd.DataFrame({"Feature": features}).to_csv(
        folder / "features.csv",
        index=False
    )

def load_persistent_result(station, parameter):
    folder = _result_folder(station, parameter)

    required = [
        "metrics.csv",
        "evaluasi.csv",
        "forecast_bulanan.csv",
        "forecast_tahunan.csv",
        "features.csv"
    ]

    if not all((folder / f).exists() for f in required):
        return None

    metrics_df = pd.read_csv(folder / "metrics.csv")
    evaluation = pd.read_csv(folder / "evaluasi.csv")
    forecast = pd.read_csv(folder / "forecast_bulanan.csv")
    annual = pd.read_csv(folder / "forecast_tahunan.csv")
    features_df = pd.read_csv(folder / "features.csv")

    if "MONTH" in evaluation.columns:
        evaluation["MONTH"] = pd.to_datetime(
            evaluation["MONTH"], errors="coerce"
        )

    if "MONTH" in forecast.columns:
        forecast["MONTH"] = pd.to_datetime(
            forecast["MONTH"], errors="coerce"
        )

    if "YEAR" in annual.columns:
        annual["YEAR"] = pd.to_numeric(
            annual["YEAR"], errors="coerce"
        )

    metrics = {
        "RMSE": float(metrics_df.iloc[0]["RMSE"]),
        "MAE": float(metrics_df.iloc[0]["MAE"]),
        "R2": float(metrics_df.iloc[0]["R2"])
    }

    return {
        "station": station,
        "parameter": parameter,
        "metrics": metrics,
        "evaluation": evaluation,
        "forecast": forecast,
        "annual": annual,
        "features": features_df["Feature"].tolist()
    }

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================================
# KONFIGURASI
# ============================================================

st.set_page_config(
    page_title="Dashboard Iklim Sumatera",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>
.stApp { background:#ffffff; }

section[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#f4f8fc 0%,#edf4fa 100%);
    border-right:1px solid #dce7f2;
}

.main-title {
    text-align:center;
    color:#12467a;
    font-size:30px;
    font-weight:800;
    line-height:1.25;
    margin-bottom:4px;
}

.subtitle {
    text-align:center;
    color:#58718b;
    font-size:15px;
    margin-bottom:18px;
}

.blue-line {
    height:2px;
    background:#d5e7f8;
    margin:14px 0 22px;
}

.section-title {
    color:#123f72;
    font-size:21px;
    font-weight:750;
    margin:15px 0 10px;
}

.profile-box {
    background:linear-gradient(135deg,#f2f8ff,#eaf4fd);
    border:1px solid #cfe3f6;
    border-radius:10px;
    padding:17px 21px;
    margin-bottom:18px;
}

.profile-item {
    font-size:14px;
    color:#173b60;
    line-height:1.9;
}

.metric-card {
    border-radius:10px;
    padding:15px 19px;
    min-height:105px;
    border:1px solid #dce8f4;
    margin-bottom:10px;
}

.metric-blue { background:#edf6ff; }
.metric-green { background:#eefaf3; }
.metric-yellow { background:#fff9e9; }
.metric-red { background:#fff0f1; }

.metric-label {
    color:#3f5872;
    font-size:13px;
}

.metric-value {
    color:#123d72;
    font-size:25px;
    font-weight:800;
}

.metric-unit {
    color:#71849a;
    font-size:12px;
}

.footer-box {
    background:#edf6ff;
    border:1px solid #d0e5fa;
    border-radius:9px;
    padding:12px 17px;
    color:#285581;
    font-size:13px;
    margin-top:20px;
}

.stButton > button {
    width:100%;
    border-radius:8px;
    border:none;
    background:#147de5;
    color:white;
    font-weight:700;
}

.stButton > button:hover {
    background:#0d69c7;
    color:white;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATASET
# ============================================================

DATASETS = {
    "Stasiun Minangkabau": "minang kabau data FIX.xlsx",
    "Stasiun Pesawaran": "pesawaran data FIX.xlsx",
    "Stasiun Maritim Panjang": "Maritim panjang data FIX.xlsx"
}


# ============================================================
# PARAMETER
# ============================================================

PARAMETER_INFO = {
    "TN": "Temperatur Minimum",
    "TX": "Temperatur Maksimum",
    "TAVG": "Temperatur Rata-rata",
    "RH_AVG": "Kelembapan Relatif Rata-rata",
    "RR": "Curah Hujan",
    "SS": "Lama Penyinaran Matahari",
    "FF_X": "Kecepatan Angin Maksimum",
    "FF_AVG": "Kecepatan Angin Rata-rata"
}


# ============================================================
# MODEL
# ============================================================

N_ESTIMATORS = 100
MAX_DEPTH = 12
MIN_SAMPLES_LEAF = 2
TRAIN_RATIO = 0.80
FORECAST_MONTHS = 360
RANDOM_STATE = 42


# ============================================================
# LOAD EXCEL
# ============================================================

@st.cache_data
def load_excel(file_path):
    raw = pd.read_excel(file_path, header=None)

    required = [
        "YEAR", "DOY", "TN", "TX", "TAVG",
        "RH_AVG", "RR", "SS", "FF_X", "FF_AVG"
    ]

    header_row = None

    for i in range(min(40, len(raw))):
        row = (
            raw.iloc[i]
            .astype(str)
            .str.strip()
            .str.upper()
            .tolist()
        )

        if sum(col in row for col in required) >= 5:
            header_row = i
            break

    df = pd.read_excel(
        file_path,
        header=header_row if header_row is not None else 0
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.upper()
    )

    return df


# ============================================================
# TANGGAL
# ============================================================

def create_date(df):
    df = df.copy()

    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(
            df["DATE"],
            errors="coerce"
        )
        return df

    if "TANGGAL" in df.columns:
        df["DATE"] = pd.to_datetime(
            df["TANGGAL"],
            errors="coerce"
        )
        return df

    if "YEAR" in df.columns and "DOY" in df.columns:

        year = pd.to_numeric(
            df["YEAR"],
            errors="coerce"
        )

        doy = pd.to_numeric(
            df["DOY"],
            errors="coerce"
        )

        df["DATE"] = pd.NaT

        valid = (
            year.notna()
            & doy.notna()
        )

        df.loc[valid, "DATE"] = (
            pd.to_datetime(
                year.loc[valid]
                .astype(int)
                .astype(str),
                format="%Y",
                errors="coerce"
            )
            + pd.to_timedelta(
                doy.loc[valid] - 1,
                unit="D"
            )
        )

    else:
        df["DATE"] = pd.NaT

    return df


# ============================================================
# CLEANING
# ============================================================

def clean_data(df):
    df = create_date(df)

    for col in PARAMETER_INFO:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    return (
        df.dropna(subset=["DATE"])
        .sort_values("DATE")
        .reset_index(drop=True)
    )


# ============================================================
# DATA BULANAN
# ============================================================

def monthly_data(df):
    df = df.copy()

    df["MONTH"] = (
        df["DATE"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    aggregation = {}

    for col in [
        "TN", "TX", "TAVG",
        "RH_AVG", "FF_X", "FF_AVG"
    ]:
        if col in df.columns:
            aggregation[col] = "mean"

    if "RR" in df.columns:
        aggregation["RR"] = "sum"

    if "SS" in df.columns:
        aggregation["SS"] = "sum"

    return (
        df.groupby("MONTH")
        .agg(aggregation)
        .reset_index()
        .sort_values("MONTH")
        .reset_index(drop=True)
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(df, target):
    data = df[
        ["MONTH", target]
    ].copy()

    data["lag1"] = data[target].shift(1)
    data["lag2"] = data[target].shift(2)
    data["lag3"] = data[target].shift(3)

    data["rolling_mean_3"] = (
        data[target]
        .shift(1)
        .rolling(3)
        .mean()
    )

    month = data["MONTH"].dt.month

    data["month_sin"] = np.sin(
        2 * np.pi * month / 12
    )

    data["month_cos"] = np.cos(
        2 * np.pi * month / 12
    )

    return data.dropna().reset_index(drop=True)


# ============================================================
# RANDOM FOREST
# ============================================================

@st.cache_data(show_spinner=False)
def train_model(monthly, target):

    data = create_features(
        monthly,
        target
    )

    features = [
        "lag1",
        "lag2",
        "lag3",
        "rolling_mean_3",
        "month_sin",
        "month_cos"
    ]

    X = data[features].values
    y = data[[target]].values

    split = int(
        len(data) * TRAIN_RATIO
    )

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_x.fit_transform(
        X_train
    )

    X_test_scaled = scaler_x.transform(
        X_test
    )

    y_train_scaled = scaler_y.fit_transform(
        y_train
    ).ravel()

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(
        X_train_scaled,
        y_train_scaled
    )

    pred_scaled = model.predict(
        X_test_scaled
    )

    pred = scaler_y.inverse_transform(
        pred_scaled.reshape(-1, 1)
    ).ravel()

    actual = y_test.ravel()

    metrics = {
        "RMSE": float(
            np.sqrt(
                mean_squared_error(
                    actual,
                    pred
                )
            )
        ),
        "MAE": float(
            mean_absolute_error(
                actual,
                pred
            )
        ),
        "R2": float(
            r2_score(
                actual,
                pred
            )
        )
    }

    evaluation = pd.DataFrame({
        "MONTH": data.iloc[split:]["MONTH"].values,
        "Aktual": actual,
        "Prediksi": pred
    })

    return (
        model,
        scaler_x,
        scaler_y,
        features,
        metrics,
        evaluation
    )


# ============================================================
# FORECAST
# ============================================================

def make_forecast(
    monthly,
    target,
    model,
    scaler_x,
    scaler_y
):

    history = (
        monthly[
            ["MONTH", target]
        ]
        .dropna()
        .sort_values("MONTH")
        .copy()
    )

    values = (
        history[target]
        .astype(float)
        .tolist()
    )

    last_month = history["MONTH"].max()

    dates = pd.date_range(
        start=last_month + pd.offsets.MonthBegin(1),
        periods=FORECAST_MONTHS,
        freq="MS"
    )

    predictions = []

    for date in dates:

        lag1 = values[-1]
        lag2 = values[-2]
        lag3 = values[-3]

        rolling_mean = np.mean(
            values[-3:]
        )

        month = date.month

        month_sin = np.sin(
            2 * np.pi * month / 12
        )

        month_cos = np.cos(
            2 * np.pi * month / 12
        )

        X_future = np.array([[
            lag1,
            lag2,
            lag3,
            rolling_mean,
            month_sin,
            month_cos
        ]])

        X_scaled = scaler_x.transform(
            X_future
        )

        prediction_scaled = model.predict(
            X_scaled
        )

        prediction = float(
            scaler_y.inverse_transform(
                prediction_scaled.reshape(-1, 1)
            )[0, 0]
        )

        predictions.append(
            prediction
        )

        values.append(
            prediction
        )

    return pd.DataFrame({
        "MONTH": dates,
        target: predictions
    })


# ============================================================
# FORECAST TAHUNAN
# ============================================================

def annual_forecast(forecast, target):

    data = forecast.copy()

    data["YEAR"] = (
        data["MONTH"].dt.year
    )

    if target in ["RR", "SS"]:
        annual = (
            data.groupby("YEAR")[target]
            .sum()
            .reset_index()
        )
    else:
        annual = (
            data.groupby("YEAR")[target]
            .mean()
            .reset_index()
        )

    return annual


# ============================================================
# KEY PENYIMPANAN HASIL
# ============================================================

def result_key(station, parameter):
    return f"{station}__{parameter}"


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div style="
        font-size:20px;
        font-weight:800;
        color:#123f72;">
        📌 Menu Navigasi
    </div>
    """,
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Pilih Tampilan:",
    [
        "🏠 Dashboard",
        "📊 Validasi & Evaluasi",
        "👤 Profil Peneliti"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    <div style="
        font-size:18px;
        font-weight:750;
        color:#123f72;">
        🌍 Wilayah Pesisir
    </div>
    """,
    unsafe_allow_html=True
)

station = st.sidebar.selectbox(
    "Pilih stasiun:",
    list(DATASETS.keys())
)

st.sidebar.markdown(
    """
    <div style="
        font-size:18px;
        font-weight:750;
        color:#123f72;
        margin-top:14px;">
        📅 Rentang Waktu
    </div>
    """,
    unsafe_allow_html=True
)

years = list(range(1985, 2026))

start_year = st.sidebar.selectbox(
    "Mulai",
    years,
    index=0
)

end_year = st.sidebar.selectbox(
    "Selesai",
    years,
    index=len(years) - 1
)

st.sidebar.markdown(
    """
    <div style="
        font-size:18px;
        font-weight:750;
        color:#123f72;
        margin-top:14px;">
        ⚙️ Parameter Iklim
    </div>
    """,
    unsafe_allow_html=True
)

parameter_selected = st.sidebar.selectbox(
    "Pilih parameter:",
    [
        f"{key} — {value}"
        for key, value in PARAMETER_INFO.items()
    ]
)

parameter = parameter_selected.split(" — ")[0]


# ============================================================
# LOAD DATA
# ============================================================

file_path = Path(
    DATASETS[station]
)

if not file_path.exists():
    st.error(
        f"File {file_path.name} tidak ditemukan."
    )
    st.stop()

with st.spinner("Memuat dataset..."):

    raw = load_excel(
        str(file_path)
    )

    daily_all = clean_data(
        raw
    )

# Data sesuai rentang sidebar
daily = daily_all[
    (daily_all["DATE"] >= pd.Timestamp(
        f"{start_year}-01-01"
    ))
    &
    (daily_all["DATE"] <= pd.Timestamp(
        f"{end_year}-12-31"
    ))
].copy()

monthly = monthly_data(
    daily
)

# Data historis penuh untuk model penelitian
daily_full = daily_all[
    (daily_all["DATE"] >= pd.Timestamp("1985-01-01"))
    &
    (daily_all["DATE"] <= pd.Timestamp("2025-12-31"))
].copy()

monthly_full = monthly_data(
    daily_full
)

if parameter not in monthly.columns:
    st.error(
        f"Parameter {parameter} tidak tersedia pada dataset."
    )
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
    🌦️ DASHBOARD MACHINE LEARNING UNTUK MEMPREDIKSI<br>
    PERUBAHAN IKLIM WILAYAH PESISIR PANTAI PULAU SUMATERA
    </div>

    <div class="subtitle">
    Analisis Temporal Jangka Panjang Berbasis Random Forest — 1985–2025
    </div>

    <div class="blue-line"></div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        """
        <div class="profile-box">

        <div class="section-title">
        👤 Profil Peneliti & Akademik
        </div>

        <div class="profile-item">
        <b>Nama Peneliti:</b> Huriyatul Firdausi
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
        <b>Dosen Pembimbing:</b> Dr. Melly Ariska, S.Pd., M.Sc.
        </div>

        <div class="profile-item">
        <b>NIM:</b> 06111382328074
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
        <b>Program Studi:</b> Pendidikan Fisika
        </div>

        <div class="profile-item">
        <b>Fakultas:</b> Keguruan dan Ilmu Pendidikan
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
        <b>Universitas:</b> Universitas Sriwijaya
        </div>

        <div class="profile-item">
        <b>Tahun:</b> 2026
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    cards = [
        (
            "🗄️ Data Harian",
            f"{len(daily):,}",
            "baris",
            "metric-blue"
        ),
        (
            "📅 Data Bulanan",
            f"{len(monthly):,}",
            "bulan",
            "metric-green"
        ),
        (
            "🗓️ Periode Awal",
            monthly["MONTH"].min().strftime("%Y-%m"),
            "tahun-bulan",
            "metric-yellow"
        ),
        (
            "🗓️ Periode Akhir",
            monthly["MONTH"].max().strftime("%Y-%m"),
            "tahun-bulan",
            "metric-red"
        )
    ]

    for col, card in zip(
        [c1, c2, c3, c4],
        cards
    ):
        with col:
            st.markdown(
                f"""
                <div class="metric-card {card[3]}">
                <div class="metric-label">{card[0]}</div>
                <div class="metric-value">{card[1]}</div>
                <div class="metric-unit">{card[2]}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    left, right = st.columns([2.5, 1])

    with left:

        st.markdown(
            """
            <div class="section-title">
            📊 Visualisasi Data Iklim Bulanan
            </div>
            """,
            unsafe_allow_html=True
        )

        chart = monthly[
            ["MONTH", parameter]
        ].dropna()

        fig = px.line(
            chart,
            x="MONTH",
            y=parameter,
            title=(
                f"{PARAMETER_INFO[parameter]} — "
                f"{station}"
            )
        )

        fig.update_layout(
            height=430,
            xaxis_title="Waktu",
            yaxis_title=parameter,
            hovermode="x unified"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        st.markdown(
            """
            <div class="section-title">
            📈 Statistik Deskriptif
            </div>
            """,
            unsafe_allow_html=True
        )

        values = monthly[
            parameter
        ].dropna()

        stats = pd.DataFrame({
            "Statistik": [
                "Rata-rata",
                "Minimum",
                "Maksimum",
                "Std. Deviasi",
                "Jumlah Data"
            ],
            "Nilai": [
                round(values.mean(), 2),
                round(values.min(), 2),
                round(values.max(), 2),
                round(values.std(), 2),
                len(values)
            ]
        })

        st.dataframe(
            stats,
            use_container_width=True,
            hide_index=True
        )

    st.markdown(
        """
        <div class="section-title">
        🤖 Random Forest
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        f"""
        **{parameter} — {PARAMETER_INFO[parameter]}**

        Data model menggunakan periode historis **1985–2025**,
        feature lag 1–3 bulan, rolling mean 3 bulan, serta month
        sin/cos. Pembagian data dilakukan secara temporal:
        **80% training dan 20% testing**.
        """
    )

    key = result_key(
        station,
        parameter
    )

    if key in st.session_state.get("results", {}):

        st.success(
            "Hasil analisis untuk stasiun dan parameter ini sudah tersimpan."
        )

        saved = st.session_state["results"][key]

        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric(
                "RMSE",
                f"{saved['metrics']['RMSE']:.4f}"
            )

        with m2:
            st.metric(
                "MAE",
                f"{saved['metrics']['MAE']:.4f}"
            )

        with m3:
            st.metric(
                "R²",
                f"{saved['metrics']['R2']:.4f}"
            )

    if st.button(
        "▶ Jalankan / Perbarui Analisis Random Forest",
        type="primary"
    ):

        with st.spinner(
            "Melatih Random Forest dan membuat forecast 2026–2055..."
        ):

            (
                model,
                scaler_x,
                scaler_y,
                features,
                metrics,
                evaluation
            ) = train_model(
                monthly_full,
                parameter
            )

            forecast = make_forecast(
                monthly_full,
                parameter,
                model,
                scaler_x,
                scaler_y
            )

            annual = annual_forecast(
                forecast,
                parameter
            )

        if "results" not in st.session_state:
            st.session_state["results"] = {}

        save_persistent_result(
            station=station,
            parameter=parameter,
            metrics=metrics,
            evaluation=evaluation,
            forecast=forecast,
            annual=annual,
            features=features
        )

        st.session_state["results"][key] = {
            "station": station,
            "parameter": parameter,
            "metrics": metrics,
            "evaluation": evaluation,
            "forecast": forecast,
            "annual": annual,
            "features": features
        }

        st.success(
            f"Hasil {station} — {parameter} berhasil disimpan."
        )

        st.rerun()


# ============================================================
# VALIDASI & EVALUASI
# ============================================================

elif page == "📊 Validasi & Evaluasi":

    st.markdown(
        """
        <div class="section-title">
        📊 Validasi & Evaluasi Model
        </div>
        """,
        unsafe_allow_html=True
    )

    key = result_key(
        station,
        parameter
    )

    results = st.session_state.get(
        "results",
        {}
    )

    if key not in results:

        st.info(
            f"""
            Hasil untuk **{station} — {parameter}** belum tersedia.

            Silakan buka halaman **Dashboard**, kemudian klik
            **Jalankan / Perbarui Analisis Random Forest**.
            Setelah tersimpan, hasilnya akan tetap tersedia ketika
            kamu berpindah stasiun atau halaman.
            """
        )

    else:

        result = results[key]

        st.success(
            f"Hasil tersimpan: **{station} — "
            f"{parameter} ({PARAMETER_INFO[parameter]})**"
        )

        # ----------------------------------------------------
        # METRIK
        # ----------------------------------------------------

        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric(
                "RMSE",
                f"{result['metrics']['RMSE']:.4f}"
            )

        with m2:
            st.metric(
                "MAE",
                f"{result['metrics']['MAE']:.4f}"
            )

        with m3:
            st.metric(
                "R²",
                f"{result['metrics']['R2']:.4f}"
            )

        st.caption(
            "Evaluasi dilakukan pada 20% data testing terakhir "
            "dari data historis 1985–2025."
        )

        # ----------------------------------------------------
        # VALIDASI
        # ----------------------------------------------------

        st.markdown(
            """
            <div class="section-title">
            📈 Validasi Data Uji
            </div>
            """,
            unsafe_allow_html=True
        )

        evaluation = result["evaluation"]

        fig_eval = go.Figure()

        fig_eval.add_trace(
            go.Scatter(
                x=evaluation["MONTH"],
                y=evaluation["Aktual"],
                mode="lines",
                name="Aktual"
            )
        )

        fig_eval.add_trace(
            go.Scatter(
                x=evaluation["MONTH"],
                y=evaluation["Prediksi"],
                mode="lines",
                name="Prediksi"
            )
        )

        fig_eval.update_layout(
            height=430,
            xaxis_title="Waktu",
            yaxis_title=parameter,
            hovermode="x unified"
        )

        st.plotly_chart(
            fig_eval,
            use_container_width=True
        )

        st.caption(
            "Grafik validasi hanya menggunakan periode historis "
            "untuk membandingkan nilai aktual dengan hasil prediksi "
            "data testing."
        )

        # ----------------------------------------------------
        # FORECAST DI HALAMAN VALIDASI
        # ----------------------------------------------------

        st.markdown(
            """
            <div class="section-title">
            🔮 Forecast 2026–2055
            </div>
            """,
            unsafe_allow_html=True
        )

        historical = (
            monthly_full[
                ["MONTH", parameter]
            ]
            .dropna()
        )

        forecast = result["forecast"]

        fig_forecast = go.Figure()

        fig_forecast.add_trace(
            go.Scatter(
                x=historical["MONTH"],
                y=historical[parameter],
                mode="lines",
                name="Data Historis 1985–2025"
            )
        )

        fig_forecast.add_trace(
            go.Scatter(
                x=forecast["MONTH"],
                y=forecast[parameter],
                mode="lines",
                name="Forecast 2026–2055"
            )
        )

        fig_forecast.add_vline(
            x=pd.Timestamp("2026-01-01"),
            line_dash="dash",
            annotation_text="Mulai Forecast 2026",
            annotation_position="top"
        )

        fig_forecast.update_layout(
            height=480,
            xaxis_title="Waktu",
            yaxis_title=parameter,
            hovermode="x unified"
        )

        st.plotly_chart(
            fig_forecast,
            use_container_width=True
        )

        # ----------------------------------------------------
        # TABEL FORECAST BULANAN
        # ----------------------------------------------------

        st.markdown(
            """
            <div class="section-title">
            📋 Forecast Bulanan 2026–2055
            </div>
            """,
            unsafe_allow_html=True
        )

        monthly_forecast_display = forecast.copy()

        monthly_forecast_display["MONTH"] = (
            monthly_forecast_display["MONTH"]
            .dt.strftime("%Y-%m")
        )

        st.dataframe(
            monthly_forecast_display,
            use_container_width=True,
            hide_index=True,
            height=350
        )

        # ----------------------------------------------------
        # TABEL FORECAST TAHUNAN
        # ----------------------------------------------------

        st.markdown(
            """
            <div class="section-title">
            📅 Forecast Tahunan 2026–2055
            </div>
            """,
            unsafe_allow_html=True
        )

        annual = result["annual"]

        st.dataframe(
            annual,
            use_container_width=True,
            hide_index=True,
            height=420
        )

        csv = annual.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "📥 Download Forecast Tahunan CSV",
            data=csv,
            file_name=(
                f"forecast_{station.replace(' ', '_')}_"
                f"{parameter}_2026_2055.csv"
            ),
            mime="text/csv"
        )

        # ----------------------------------------------------
        # INFO HASIL
        # ----------------------------------------------------

        st.markdown(
            """
            <div class="section-title">
            ℹ️ Keterangan
            </div>
            """,
            unsafe_allow_html=True
        )

        st.info(
            """
            **1985–2025** merupakan periode data historis yang
            digunakan dalam pembentukan model. **2026–2055**
            merupakan periode forecast menggunakan Random Forest.
            Nilai forecast bukan data observasi aktual.
            """
        )


# ============================================================
# PROFIL PENELITI
# ============================================================

else:

    st.markdown(
        """
        <div class="section-title">
        👤 Profil Peneliti & Akademik
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="profile-box">

        <div class="profile-item">
        <b>Nama Peneliti:</b> Huriyatul Firdausi
        </div>

        <div class="profile-item">
        <b>NIM:</b> 06111382328074
        </div>

        <div class="profile-item">
        <b>Dosen Pembimbing:</b> Dr. Melly Ariska, S.Pd., M.Sc.
        </div>

        <div class="profile-item">
        <b>Program Studi:</b> Pendidikan Fisika
        </div>

        <div class="profile-item">
        <b>Fakultas:</b> Keguruan dan Ilmu Pendidikan
        </div>

        <div class="profile-item">
        <b>Universitas:</b> Universitas Sriwijaya
        </div>

        <div class="profile-item">
        <b>Tahun:</b> 2026
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-title">
        📚 Judul Penelitian
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "MACHINE LEARNING UNTUK MEMPREDIKSI PERUBAHAN IKLIM "
        "WILAYAH PESISIR PANTAI PULAU SUMATERA"
    )

    st.markdown(
        """
        <div class="section-title">
        🧪 Metodologi Penelitian
        </div>
        """,
        unsafe_allow_html=True
    )

    methodology = pd.DataFrame({
        "Komponen": [
            "Data Historis",
            "Periode Historis",
            "Model",
            "Feature Engineering",
            "Scaling",
            "Pembagian Data",
            "Evaluasi",
            "Forecast"
        ],
        "Keterangan": [
            "Data iklim tiga stasiun pesisir",
            "1985–2025",
            "Random Forest Regressor",
            "Lag 1–3, Rolling Mean 3, Month Sin/Cos",
            "MinMaxScaler",
            "80% Training : 20% Testing",
            "RMSE, MAE, R²",
            "2026–2055"
        ]
    })

    st.dataframe(
        methodology,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer-box">
    🌍 Wilayah Pesisir Pulau Sumatera
    &nbsp; | &nbsp;
    📅 Data Historis 1985–2025
    &nbsp; | &nbsp;
    🤖 Random Forest
    &nbsp; | &nbsp;
    🔮 Forecast 2026–2055
    </div>
    """,
    unsafe_allow_html=True
)
