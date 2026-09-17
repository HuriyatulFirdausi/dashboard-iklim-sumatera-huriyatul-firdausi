import streamlit as st
import pandas as pd
import numpy as np
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
    text-align:center;color:#12467a;font-size:30px;
    font-weight:800;line-height:1.25;margin-bottom:4px;
}
.subtitle {
    text-align:center;color:#58718b;font-size:15px;margin-bottom:18px;
}
.blue-line { height:2px;background:#d5e7f8;margin:14px 0 22px; }
.section-title { color:#123f72;font-size:21px;font-weight:750;margin:12px 0 10px; }
.profile-box {
    background:linear-gradient(135deg,#f2f8ff,#eaf4fd);
    border:1px solid #cfe3f6;border-radius:10px;
    padding:17px 21px;margin-bottom:18px;
}
.profile-item { font-size:14px;color:#173b60;line-height:1.9; }
.metric-card {
    border-radius:10px;padding:15px 19px;min-height:105px;
    border:1px solid #dce8f4;margin-bottom:10px;
}
.metric-blue { background:#edf6ff; }
.metric-green { background:#eefaf3; }
.metric-yellow { background:#fff9e9; }
.metric-red { background:#fff0f1; }
.metric-label { color:#3f5872;font-size:13px; }
.metric-value { color:#123d72;font-size:25px;font-weight:800; }
.metric-unit { color:#71849a;font-size:12px; }
.footer-box {
    background:#edf6ff;border:1px solid #d0e5fa;border-radius:9px;
    padding:12px 17px;color:#285581;font-size:13px;margin-top:20px;
}
.stButton > button {
    width:100%;border-radius:8px;border:none;
    background:#147de5;color:white;font-weight:700;
}
.stButton > button:hover { background:#0d69c7;color:white; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATASET & PARAMETER
# ============================================================
DATASETS = {
    "Stasiun Minangkabau": "minang kabau data FIX.xlsx",
    "Stasiun Pesawaran": "pesawaran data FIX.xlsx",
    "Stasiun Maritim Panjang": "Maritim panjang data FIX.xlsx"
}

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
        "YEAR","DOY","TN","TX","TAVG",
        "RH_AVG","RR","SS","FF_X","FF_AVG"
    ]

    header_row = None

    for i in range(min(40, len(raw))):
        row = (
            raw.iloc[i].astype(str)
            .str.strip().str.upper().tolist()
        )
        if sum(col in row for col in required) >= 5:
            header_row = i
            break

    df = pd.read_excel(
        file_path,
        header=header_row if header_row is not None else 0
    )
    df.columns = (
        df.columns.astype(str)
        .str.strip().str.upper()
    )
    return df

# ============================================================
# DATE
# ============================================================
def create_date(df):
    df = df.copy()

    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        return df

    if "TANGGAL" in df.columns:
        df["DATE"] = pd.to_datetime(df["TANGGAL"], errors="coerce")
        return df

    if "YEAR" in df.columns and "DOY" in df.columns:
        year = pd.to_numeric(df["YEAR"], errors="coerce")
        doy = pd.to_numeric(df["DOY"], errors="coerce")
        df["DATE"] = pd.NaT
        valid = year.notna() & doy.notna()
        df.loc[valid, "DATE"] = (
            pd.to_datetime(
                year.loc[valid].astype(int).astype(str),
                format="%Y", errors="coerce"
            ) +
            pd.to_timedelta(doy.loc[valid] - 1, unit="D")
        )
    else:
        df["DATE"] = pd.NaT

    return df

def clean_data(df):
    df = create_date(df)

    for col in PARAMETER_INFO:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["DATE"])
    return df.sort_values("DATE")

# ============================================================
# BULANAN
# ============================================================
def monthly_data(df):
    df = df.copy()
    df["MONTH"] = (
        df["DATE"].dt.to_period("M").dt.to_timestamp()
    )

    agg = {}
    for col in ["TN","TX","TAVG","RH_AVG","FF_X","FF_AVG"]:
        if col in df.columns:
            agg[col] = "mean"

    if "RR" in df.columns:
        agg["RR"] = "sum"

    if "SS" in df.columns:
        agg["SS"] = "sum"

    return (
        df.groupby("MONTH")
        .agg(agg)
        .reset_index()
        .sort_values("MONTH")
    )

# ============================================================
# FEATURE ENGINEERING
# ============================================================
def create_features(df, target):
    data = df[["MONTH", target]].copy()

    data["lag1"] = data[target].shift(1)
    data["lag2"] = data[target].shift(2)
    data["lag3"] = data[target].shift(3)

    data["rolling_mean_3"] = (
        data[target].shift(1).rolling(3).mean()
    )

    month = data["MONTH"].dt.month
    data["month_sin"] = np.sin(2 * np.pi * month / 12)
    data["month_cos"] = np.cos(2 * np.pi * month / 12)

    return data.dropna()

# ============================================================
# TRAIN RANDOM FOREST
# ============================================================
@st.cache_data(show_spinner=False)
def train_model_cached(monthly, target):
    data = create_features(monthly, target)

    features = [
        "lag1","lag2","lag3",
        "rolling_mean_3",
        "month_sin","month_cos"
    ]

    X = data[features].values
    y = data[[target]].values

    split = int(len(data) * TRAIN_RATIO)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_s = scaler_x.fit_transform(X_train)
    X_test_s = scaler_x.transform(X_test)
    y_train_s = scaler_y.fit_transform(y_train).ravel()

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X_train_s, y_train_s)

    pred_s = model.predict(X_test_s)
    pred = scaler_y.inverse_transform(
        pred_s.reshape(-1,1)
    ).ravel()

    actual = y_test.ravel()

    metrics = {
        "RMSE": float(np.sqrt(
            mean_squared_error(actual, pred)
        )),
        "MAE": float(
            mean_absolute_error(actual, pred)
        ),
        "R2": float(
            r2_score(actual, pred)
        )
    }

    evaluation = pd.DataFrame({
        "MONTH": data.iloc[split:]["MONTH"].values,
        "Aktual": actual,
        "Prediksi": pred
    })

    return (
        model, scaler_x, scaler_y,
        features, metrics, evaluation
    )

# ============================================================
# FORECAST
# ============================================================
def make_forecast(monthly, target, model, scaler_x, scaler_y):
    history = (
        monthly[["MONTH", target]]
        .dropna()
        .sort_values("MONTH")
    )

    values = history[target].astype(float).tolist()
    last_month = history["MONTH"].max()

    dates = pd.date_range(
        start=last_month + pd.offsets.MonthBegin(1),
        periods=FORECAST_MONTHS,
        freq="MS"
    )

    predictions = []

    for date in dates:
        lag1, lag2, lag3 = values[-1], values[-2], values[-3]
        rolling = float(np.mean(values[-3:]))

        month = date.month
        sin_month = np.sin(2 * np.pi * month / 12)
        cos_month = np.cos(2 * np.pi * month / 12)

        X = np.array([[
            lag1, lag2, lag3,
            rolling,
            sin_month, cos_month
        ]])

        X_s = scaler_x.transform(X)
        pred_s = model.predict(X_s)

        pred = float(
            scaler_y.inverse_transform(
                pred_s.reshape(-1,1)
            )[0,0]
        )

        predictions.append(pred)
        values.append(pred)

    return pd.DataFrame({
        "MONTH": dates,
        target: predictions
    })

def annual_forecast(forecast, target):
    temp = forecast.copy()
    temp["YEAR"] = temp["MONTH"].dt.year

    if target in ["RR", "SS"]:
        result = (
            temp.groupby("YEAR")[target]
            .sum()
            .reset_index()
        )
    else:
        result = (
            temp.groupby("YEAR")[target]
            .mean()
            .reset_index()
        )

    return result

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown(
    '<div style="font-size:20px;font-weight:800;color:#123f72;">📌 Menu Navigasi</div>',
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
    '<div style="font-size:18px;font-weight:750;color:#123f72;">🌍 Wilayah Pesisir</div>',
    unsafe_allow_html=True
)

station = st.sidebar.selectbox(
    "Pilih stasiun:",
    list(DATASETS.keys())
)

st.sidebar.markdown(
    '<div style="font-size:18px;font-weight:750;color:#123f72;margin-top:14px;">📅 Rentang Waktu</div>',
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
    index=len(years)-1
)

st.sidebar.markdown(
    '<div style="font-size:18px;font-weight:750;color:#123f72;margin-top:14px;">⚙️ Parameter Iklim</div>',
    unsafe_allow_html=True
)

parameter_selected = st.sidebar.selectbox(
    "Pilih parameter:",
    [
        f"{k} — {v}"
        for k,v in PARAMETER_INFO.items()
    ]
)

parameter = parameter_selected.split(" — ")[0]

# ============================================================
# LOAD DATA
# ============================================================
file_path = Path(DATASETS[station])

if not file_path.exists():
    st.error(f"File {file_path.name} tidak ditemukan.")
    st.stop()

with st.spinner("Memuat dan mengolah dataset..."):
    raw = load_excel(file_path)
    daily = clean_data(raw)

daily = daily[
    (daily["DATE"] >= pd.Timestamp(f"{start_year}-01-01")) &
    (daily["DATE"] <= pd.Timestamp(f"{end_year}-12-31"))
].copy()

monthly = monthly_data(daily)

if parameter not in monthly.columns:
    st.error(f"Parameter {parameter} tidak tersedia pada dataset.")
    st.stop()

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="main-title">
🌦️ DASHBOARD MACHINE LEARNING UNTUK MEMPREDIKSI<br>
PERUBAHAN IKLIM WILAYAH PESISIR PANTAI PULAU SUMATERA
</div>
<div class="subtitle">
Analisis Temporal Jangka Panjang Berbasis Random Forest — 1985–2025
</div>
<div class="blue-line"></div>
""", unsafe_allow_html=True)

# ============================================================
# DASHBOARD
# ============================================================
if page == "🏠 Dashboard":

    st.markdown("""
    <div class="profile-box">
    <div class="section-title">👤 Profil Peneliti & Akademik</div>
    <div class="profile-item">
    <b>Nama Peneliti:</b> Huriyatul Firdausi
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b>Dosen Pembimbing:</b> Dr. Melly Ariska, S.Pd., M.Sc.
    </div>
    <div class="profile-item">
    <b>NIM:</b> 06111382328074
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b>Program Studi:</b> Pendidikan Fisika
    </div>
    <div class="profile-item">
    <b>Fakultas:</b> Keguruan dan Ilmu Pendidikan
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b>Universitas:</b> Universitas Sriwijaya
    </div>
    <div class="profile-item">
    <b>Tahun:</b> 2026
    </div>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)

    cards = [
        ("🗄️ Data Harian", f"{len(daily):,}", "baris", "metric-blue"),
        ("📅 Data Bulanan", f"{len(monthly):,}", "bulan", "metric-green"),
        ("🗓️ Periode Awal",
         monthly["MONTH"].min().strftime("%Y-%m"),
         "tahun-bulan", "metric-yellow"),
        ("🗓️ Periode Akhir",
         monthly["MONTH"].max().strftime("%Y-%m"),
         "tahun-bulan", "metric-red")
    ]

    for col, card in zip([c1,c2,c3,c4], cards):
        with col:
            st.markdown(f"""
            <div class="metric-card {card[3]}">
            <div class="metric-label">{card[0]}</div>
            <div class="metric-value">{card[1]}</div>
            <div class="metric-unit">{card[2]}</div>
            </div>
            """, unsafe_allow_html=True)

    left,right = st.columns([2.5,1])

    with left:
        st.markdown(
            '<div class="section-title">📊 Visualisasi Data Iklim Bulanan</div>',
            unsafe_allow_html=True
        )

        chart = monthly[["MONTH",parameter]].dropna()

        fig = px.line(
            chart,
            x="MONTH",
            y=parameter,
            title=f"{PARAMETER_INFO[parameter]} — {station}"
        )
        fig.update_layout(
            height=430,
            xaxis_title="Waktu",
            yaxis_title=parameter,
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown(
            '<div class="section-title">📈 Statistik Deskriptif</div>',
            unsafe_allow_html=True
        )

        values = monthly[parameter].dropna()

        stats = pd.DataFrame({
            "Statistik": [
                "Rata-rata",
                "Minimum",
                "Maksimum",
                "Std. Deviasi",
                "Jumlah Data"
            ],
            "Nilai": [
                round(values.mean(),2),
                round(values.min(),2),
                round(values.max(),2),
                round(values.std(),2),
                len(values)
            ]
        })

        st.dataframe(
            stats,
            use_container_width=True,
            hide_index=True
        )

    st.markdown(
        '<div class="section-title">📋 Data Bulanan</div>',
        unsafe_allow_html=True
    )

    display = monthly.copy()
    display["MONTH"] = display["MONTH"].dt.strftime("%Y-%m")

    st.dataframe(
        display,
        use_container_width=True,
        height=320
    )

    st.markdown(
        '<div class="section-title">🤖 Analisis Random Forest</div>',
        unsafe_allow_html=True
    )

    st.info(
        f"Parameter: {parameter} — {PARAMETER_INFO[parameter]} | "
        "Feature: lag 1–3 bulan, rolling mean 3 bulan, month sin/cos | "
        "Training 80% dan testing 20%."
    )

    if st.button(
        "▶ Jalankan Analisis Random Forest",
        type="primary"
    ):
        with st.spinner("Melatih Random Forest dan membuat forecast..."):
            (
                model, scaler_x, scaler_y,
                features, metrics, evaluation
            ) = train_model_cached(
                monthly,
                parameter
            )

            forecast = make_forecast(
                monthly,
                parameter,
                model,
                scaler_x,
                scaler_y
            )

        st.session_state["result"] = {
            "station": station,
            "parameter": parameter,
            "metrics": metrics,
            "evaluation": evaluation,
            "forecast": forecast,
            "features": features
        }

        st.success("Analisis berhasil dijalankan.")

# ============================================================
# VALIDASI & EVALUASI
# ============================================================
elif page == "📊 Validasi & Evaluasi":

    st.markdown(
        '<div class="section-title">📊 Validasi & Evaluasi Model</div>',
        unsafe_allow_html=True
    )

    if "result" not in st.session_state:
        st.info(
            "Jalankan Random Forest terlebih dahulu pada halaman Dashboard."
        )
    else:
        result = st.session_state["result"]

        if result["station"] != station or result["parameter"] != parameter:
            st.warning(
                "Hasil model yang tersimpan berasal dari kombinasi "
                "stasiun/parameter berbeda. Jalankan analisis kembali "
                "pada Dashboard untuk pilihan saat ini."
            )
        else:
            metrics = result["metrics"]

            m1,m2,m3 = st.columns(3)

            with m1:
                st.metric("RMSE", f"{metrics['RMSE']:.4f}")

            with m2:
                st.metric("MAE", f"{metrics['MAE']:.4f}")

            with m3:
                st.metric("R²", f"{metrics['R2']:.4f}")

            st.markdown(
                '<div class="section-title">📈 Validasi Data Uji 1985–2025</div>',
                unsafe_allow_html=True
            )

            evaluation = result["evaluation"].copy()

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
                "RMSE, MAE, dan R² dihitung menggunakan data testing "
                "20% terakhir dari data historis."
            )

            st.markdown(
                '<div class="section-title">🔮 Gabungan Historis dan Forecast 2026–2055</div>',
                unsafe_allow_html=True
            )

            historical = monthly[
                ["MONTH", parameter]
            ].dropna()

            forecast = result["forecast"]

            fig_all = go.Figure()

            fig_all.add_trace(
                go.Scatter(
                    x=historical["MONTH"],
                    y=historical[parameter],
                    mode="lines",
                    name="Data Historis 1985–2025"
                )
            )

            fig_all.add_trace(
                go.Scatter(
                    x=forecast["MONTH"],
                    y=forecast[parameter],
                    mode="lines",
                    name="Forecast 2026–2055"
                )
            )

            fig_all.add_vline(
                x=pd.Timestamp("2026-01-01"),
                line_dash="dash",
                annotation_text="Mulai Forecast 2026",
                annotation_position="top"
            )

            fig_all.update_layout(
                height=470,
                xaxis_title="Waktu",
                yaxis_title=parameter,
                hovermode="x unified"
            )

            st.plotly_chart(
                fig_all,
                use_container_width=True
            )

            st.markdown(
                '<div class="section-title">📋 Tabel Hasil Validasi</div>',
                unsafe_allow_html=True
            )

            st.dataframe(
                evaluation,
                use_container_width=True,
                height=400
            )

# ============================================================
# PROFIL
# ============================================================
else:

    st.markdown(
        '<div class="section-title">👤 Profil Peneliti & Akademik</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="profile-box">
    <div class="profile-item"><b>Nama:</b> Huriyatul Firdausi</div>
    <div class="profile-item"><b>NIM:</b> 06111382328074</div>
    <div class="profile-item"><b>Dosen Pembimbing:</b> Dr. Melly Ariska, S.Pd., M.Sc.</div>
    <div class="profile-item"><b>Program Studi:</b> Pendidikan Fisika</div>
    <div class="profile-item"><b>Fakultas:</b> Keguruan dan Ilmu Pendidikan</div>
    <div class="profile-item"><b>Universitas:</b> Universitas Sriwijaya</div>
    <div class="profile-item"><b>Tahun:</b> 2026</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">📚 Judul Penelitian</div>',
        unsafe_allow_html=True
    )

    st.info(
        "MACHINE LEARNING UNTUK MEMPREDIKSI PERUBAHAN IKLIM "
        "WILAYAH PESISIR PANTAI PULAU SUMATERA"
    )

    st.markdown(
        '<div class="section-title">🧪 Metodologi Penelitian</div>',
        unsafe_allow_html=True
    )

    method = pd.DataFrame({
        "Komponen": [
            "Data Historis",
            "Periode Historis",
            "Model",
            "Feature Engineering",
            "Scaling",
            "Evaluasi",
            "Periode Forecast"
        ],
        "Keterangan": [
            "Data iklim tiga stasiun pesisir",
            "1985–2025",
            "Random Forest Regressor",
            "Lag 1–3, Rolling Mean 3, Month Sin/Cos",
            "MinMaxScaler",
            "RMSE, MAE, R²",
            "2026–2055"
        ]
    })

    st.dataframe(
        method,
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer-box">
🌍 Wilayah Pesisir Pulau Sumatera &nbsp; | &nbsp;
📅 Data Historis 1985–2025 &nbsp; | &nbsp;
🤖 Random Forest &nbsp; | &nbsp;
🔮 Forecast 2026–2055
</div>
""", unsafe_allow_html=True)
