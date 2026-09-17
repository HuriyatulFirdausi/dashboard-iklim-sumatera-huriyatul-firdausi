import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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
# CSS — TAMPILAN DASHBOARD
# ============================================================

st.markdown("""
<style>

    /* Background utama */
    .stApp {
        background-color: #ffffff;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #f4f8fc 0%,
            #eef4fa 100%
        );
        border-right: 1px solid #dce7f2;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }

    /* Judul utama */
    .main-title {
        text-align: center;
        color: #12467a;
        font-size: 31px;
        font-weight: 800;
        line-height: 1.25;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #58718b;
        font-size: 15px;
        margin-bottom: 20px;
    }

    /* Garis */
    .blue-line {
        height: 2px;
        background: #d5e7f8;
        margin: 15px 0 22px 0;
    }

    /* Section */
    .section-title {
        color: #123f72;
        font-size: 21px;
        font-weight: 750;
        margin-bottom: 10px;
    }

    /* Profile box */
    .profile-box {
        background: linear-gradient(
            135deg,
            #f2f8ff,
            #eaf4fd
        );
        border: 1px solid #cfe3f6;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 18px;
    }

    .profile-item {
        font-size: 14px;
        color: #173b60;
        line-height: 1.9;
    }

    /* Card */
    .metric-card {
        border-radius: 10px;
        padding: 17px 20px;
        min-height: 115px;
        border: 1px solid #dce8f4;
        margin-bottom: 12px;
    }

    .metric-blue {
        background: #edf6ff;
    }

    .metric-green {
        background: #eefaf3;
    }

    .metric-yellow {
        background: #fff9e9;
    }

    .metric-red {
        background: #fff0f1;
    }

    .metric-label {
        color: #3f5872;
        font-size: 13px;
        margin-bottom: 5px;
    }

    .metric-value {
        color: #123d72;
        font-size: 26px;
        font-weight: 800;
    }

    .metric-unit {
        color: #71849a;
        font-size: 12px;
    }

    /* Panel */
    .panel {
        border: 1px solid #dbe7f2;
        border-radius: 10px;
        padding: 17px;
        background: white;
    }

    /* Footer */
    .footer-box {
        background: #edf6ff;
        border: 1px solid #d0e5fa;
        border-radius: 9px;
        padding: 13px 18px;
        color: #285581;
        font-size: 13px;
        margin-top: 20px;
    }

    /* Sidebar title */
    .sidebar-title {
        color: #123f72;
        font-size: 20px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .sidebar-section {
        color: #123f72;
        font-size: 18px;
        font-weight: 750;
        margin-top: 18px;
        margin-bottom: 8px;
    }

    /* Tombol */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        border: none;
        background: #147de5;
        color: white;
        font-weight: 700;
        padding: 10px 15px;
    }

    .stButton > button:hover {
        background: #0d69c7;
        color: white;
    }

    /* Radio */
    div[role="radiogroup"] label {
        padding: 3px 0;
    }

    /* Dataframe */
    .stDataFrame {
        border-radius: 8px;
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
# BACA EXCEL
# ============================================================

@st.cache_data
def load_excel(file_path):

    raw = pd.read_excel(
        file_path,
        header=None
    )

    required = [
        "YEAR", "DOY", "TN", "TX",
        "TAVG", "RH_AVG", "RR",
        "SS", "FF_X", "FF_AVG"
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

        matches = sum(
            col in row
            for col in required
        )

        if matches >= 5:
            header_row = i
            break

    if header_row is not None:

        df = pd.read_excel(
            file_path,
            header=header_row
        )

    else:

        df = pd.read_excel(
            file_path
        )

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.upper()
    )

    return df


# ============================================================
# BUAT TANGGAL
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
            &
            doy.notna()
        )

        df.loc[valid, "DATE"] = (
            pd.to_datetime(
                year.loc[valid]
                .astype(int)
                .astype(str),
                format="%Y",
                errors="coerce"
            )
            +
            pd.to_timedelta(
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

    df = df.dropna(
        subset=["DATE"]
    )

    return df.sort_values(
        "DATE"
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

    agg = {}

    for col in [
        "TN",
        "TX",
        "TAVG",
        "RH_AVG",
        "FF_X",
        "FF_AVG"
    ]:

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

    return data.dropna()


# ============================================================
# RANDOM FOREST
# ============================================================

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

    X_train_s = scaler_x.fit_transform(
        X_train
    )

    X_test_s = scaler_x.transform(
        X_test
    )

    y_train_s = scaler_y.fit_transform(
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
        X_train_s,
        y_train_s
    )

    pred_s = model.predict(
        X_test_s
    )

    pred = scaler_y.inverse_transform(
        pred_s.reshape(-1, 1)
    ).ravel()

    actual = y_test.ravel()

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            pred
        )
    )

    mae = mean_absolute_error(
        actual,
        pred
    )

    r2 = r2_score(
        actual,
        pred
    )

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
        {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        },
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

    history = monthly[
        ["MONTH", target]
    ].dropna().copy()

    values = history[target].tolist()

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

        rolling = np.mean(
            values[-3:]
        )

        month = date.month

        sin_month = np.sin(
            2 * np.pi * month / 12
        )

        cos_month = np.cos(
            2 * np.pi * month / 12
        )

        X = np.array([[
            lag1,
            lag2,
            lag3,
            rolling,
            sin_month,
            cos_month
        ]])

        X_s = scaler_x.transform(X)

        pred_s = model.predict(X_s)

        pred = scaler_y.inverse_transform(
            pred_s.reshape(-1, 1)
        )[0, 0]

        predictions.append(pred)

        values.append(pred)

    return pd.DataFrame({
        "MONTH": dates,
        target: predictions
    })


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    '<div class="sidebar-title">📌 Menu Navigasi</div>',
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Pilih Tampilan:",
    [
        "🏠 Dashboard",
        "📊 Validasi & Evaluasi",
        "👤 Profil Peneliti"
    ],
    label_visibility="visible"
)


st.sidebar.markdown("---")


# ============================================================
# WILAYAH
# ============================================================

st.sidebar.markdown(
    '<div class="sidebar-section">🌍 Wilayah Pesisir</div>',
    unsafe_allow_html=True
)

station = st.sidebar.selectbox(
    "Pilih wilayah:",
    list(DATASETS.keys())
)


# ============================================================
# RENTANG WAKTU
# ============================================================

st.sidebar.markdown(
    '<div class="sidebar-section">📅 Rentang Waktu</div>',
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


# ============================================================
# PARAMETER
# ============================================================

st.sidebar.markdown(
    '<div class="sidebar-section">⚙️ Parameter Iklim</div>',
    unsafe_allow_html=True
)

parameter_options = [
    f"{key} — {value}"
    for key, value in PARAMETER_INFO.items()
]

parameter_selected = st.sidebar.selectbox(
    "Pilih Parameter:",
    parameter_options
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


with st.spinner("Memuat data..."):

    raw = load_excel(
        file_path
    )

    daily = clean_data(
        raw
    )


daily = daily[
    (daily["DATE"] >= pd.Timestamp(
        f"{start_year}-01-01"
    ))
    &
    (daily["DATE"] <= pd.Timestamp(
        f"{end_year}-12-31"
    ))
].copy()


monthly = monthly_data(
    daily
)


if parameter not in monthly.columns:

    st.error(
        f"Parameter {parameter} tidak tersedia pada dataset."
    )

    st.stop()


# ============================================================
# HEADER UTAMA
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

    # --------------------------------------------------------
    # PROFIL RINGKAS
    # --------------------------------------------------------

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
        <b>Universitas:</b> Universitas Sriwijaya
        </div>

        <div class="profile-item">
        <b>Program Studi:</b> Pendidikan Fisika
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
        <b>Tahun:</b> 2026
        </div>

        <div class="profile-item">
        <b>Fakultas:</b> Keguruan dan Ilmu Pendidikan
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # METRIC CARDS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.markdown(
            f"""
            <div class="metric-card metric-blue">
            <div class="metric-label">🗄️ Data Harian</div>
            <div class="metric-value">
            {len(daily):,}
            </div>
            <div class="metric-unit">baris</div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with c2:

        st.markdown(
            f"""
            <div class="metric-card metric-green">
            <div class="metric-label">📅 Data Bulanan</div>
            <div class="metric-value">
            {len(monthly):,}
            </div>
            <div class="metric-unit">bulan</div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with c3:

        st.markdown(
            f"""
            <div class="metric-card metric-yellow">
            <div class="metric-label">🗓️ Periode Awal</div>
            <div class="metric-value">
            {monthly["MONTH"].min().strftime("%Y-%m")}
            </div>
            <div class="metric-unit">
            Tahun-Bulan
            </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with c4:

        st.markdown(
            f"""
            <div class="metric-card metric-red">
            <div class="metric-label">🗓️ Periode Akhir</div>
            <div class="metric-value">
            {monthly["MONTH"].max().strftime("%Y-%m")}
            </div>
            <div class="metric-unit">
            Tahun-Bulan
            </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # GRAFIK + STATISTIK
    # --------------------------------------------------------

    left, right = st.columns(
        [2.5, 1]
    )


    with left:

        st.markdown(
            '<div class="section-title">📊 Visualisasi Data Iklim Bulanan</div>',
            unsafe_allow_html=True
        )

        chart = monthly[
            [
                "MONTH",
                parameter
            ]
        ].dropna()


        fig = px.line(
            chart,
            x="MONTH",
            y=parameter,
            title=(
                f"{PARAMETER_INFO[parameter]} Bulanan — "
                f"{station}"
            )
        )


        fig.update_layout(
            height=430,
            margin=dict(
                l=20,
                r=20,
                t=55,
                b=20
            ),
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
            '<div class="section-title">📈 Statistik Deskriptif</div>',
            unsafe_allow_html=True
        )

        values = monthly[
            parameter
        ].dropna()


        stats_table = pd.DataFrame({
            "Statistik": [
                "Rata-rata",
                "Minimum",
                "Maksimum",
                "Standar Deviasi",
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
            stats_table,
            use_container_width=True,
            hide_index=True,
            height=245
        )


    # --------------------------------------------------------
    # DATA BULANAN
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">📋 Data Bulanan</div>',
        unsafe_allow_html=True
    )

    display_monthly = monthly.copy()

    display_monthly["MONTH"] = (
        display_monthly["MONTH"]
        .dt.strftime("%Y-%m")
    )


    st.dataframe(
        display_monthly,
        use_container_width=True,
        height=350
    )


    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">🤖 Analisis Random Forest</div>',
        unsafe_allow_html=True
    )


    st.info(
        f"""
        Parameter yang dipilih: **{parameter} — {PARAMETER_INFO[parameter]}**

        Model menggunakan lag 1, 2, 3 bulan, rolling mean 3 bulan,
        serta fitur siklus bulan. Pembagian data dilakukan secara
        temporal dengan **80% data training dan 20% data testing**.
        """
    )


    if st.button(
        "▶ Jalankan Analisis Random Forest",
        type="primary"
    ):

        with st.spinner(
            "Sedang melatih Random Forest..."
        ):

            (
                model,
                scaler_x,
                scaler_y,
                features,
                metrics,
                evaluation
            ) = train_model(
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


        st.success(
            "Analisis Random Forest berhasil dijalankan."
        )


    # --------------------------------------------------------
    # FORECAST
    # --------------------------------------------------------

    if "result" in st.session_state:

        result = st.session_state["result"]

        if (
            result["station"] == station
            and
            result["parameter"] == parameter
        ):

            st.markdown(
                '<div class="section-title">🔮 Forecast 2026–2055</div>',
                unsafe_allow_html=True
            )


            forecast = result["forecast"]


            fig_future = px.line(
                forecast,
                x="MONTH",
                y=parameter,
                title=(
                    f"Prediksi {PARAMETER_INFO[parameter]} "
                    f"2026–2055"
                )
            )


            fig_future.update_layout(
                height=420,
                xaxis_title="Tahun",
                yaxis_title=parameter,
                hovermode="x unified"
            )


            st.plotly_chart(
                fig_future,
                use_container_width=True
            )


            st.dataframe(
                forecast,
                use_container_width=True,
                height=300
            )


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
            """
            Model belum dijalankan.

            Silakan kembali ke **Dashboard**, pilih parameter,
            kemudian klik **Jalankan Analisis Random Forest**.
            """
        )

    else:

        result = st.session_state["result"]


        if result["station"] != station:

            st.warning(
                "Hasil model yang tersedia berasal dari stasiun berbeda."
            )

        else:

            metrics = result["metrics"]


            c1, c2, c3 = st.columns(3)


            with c1:

                st.metric(
                    "RMSE",
                    f"{metrics['RMSE']:.4f}"
                )


            with c2:

                st.metric(
                    "MAE",
                    f"{metrics['MAE']:.4f}"
                )


            with c3:

                st.metric(
                    "R²",
                    f"{metrics['R2']:.4f}"
                )


            st.markdown(
                '<div class="section-title">📈 Aktual vs Prediksi</div>',
                unsafe_allow_html=True
            )


            evaluation = result[
                "evaluation"
            ]


            long_eval = evaluation.melt(
                id_vars="MONTH",
                value_vars=[
                    "Aktual",
                    "Prediksi"
                ],
                var_name="Jenis",
                value_name="Nilai"
            )


            fig = px.line(
                long_eval,
                x="MONTH",
                y="Nilai",
                color="Jenis",
                title=(
                    f"Aktual vs Prediksi — "
                    f"{result['parameter']}"
                )
            )


            fig.update_layout(
                height=450,
                hovermode="x unified"
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


            st.markdown(
                '<div class="section-title">📋 Detail Evaluasi</div>',
                unsafe_allow_html=True
            )


            st.dataframe(
                evaluation,
                use_container_width=True,
                height=400
            )


# ============================================================
# PROFIL PENELITI
# ============================================================

elif page == "👤 Profil Peneliti":

    st.markdown(
        '<div class="section-title">👤 Profil Peneliti & Akademik</div>',
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
        '<div class="section-title">📚 Judul Penelitian</div>',
        unsafe_allow_html=True
    )


    st.info(
        """
        **MACHINE LEARNING UNTUK MEMPREDIKSI PERUBAHAN IKLIM
        WILAYAH PESISIR PANTAI PULAU SUMATERA**
        """
    )


    st.markdown(
        '<div class="section-title">🧪 Metodologi</div>',
        unsafe_allow_html=True
    )


    methodology = pd.DataFrame({
        "Komponen": [
            "Data Historis",
            "Periode",
            "Model",
            "Feature Engineering",
            "Scaling",
            "Evaluasi",
            "Forecast"
        ],
        "Keterangan": [
            "Data iklim tiga stasiun pesisir",
            "1985–2025",
            "Random Forest Regressor",
            "Lag 1–3 dan Rolling Mean 3",
            "MinMaxScaler",
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
    ℹ️ Dashboard ini dikembangkan sebagai bagian dari penelitian skripsi.
    &nbsp;&nbsp;|&nbsp;&nbsp;
    🌍 Wilayah: Pesisir Pulau Sumatera
    &nbsp;&nbsp;|&nbsp;&nbsp;
    📅 Data historis: 1985–2025
    </div>
    """,
    unsafe_allow_html=True
)
