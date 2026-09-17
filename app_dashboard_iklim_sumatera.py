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
    layout="wide"
)


# ============================================================
# DATASET
# ============================================================

DATASETS = {
    "Stasiun Minangkabau": "minang kabau data FIX.xlsx",
    "Stasiun Pesawaran": "pesawaran data FIX.xlsx",
    "Stasiun Maritim Panjang": "Maritim panjang data FIX.xlsx"
}


# ============================================================
# PARAMETER IKLIM
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
# KONFIGURASI MODEL
# ============================================================

RANDOM_STATE = 42

N_ESTIMATORS = 100

MAX_DEPTH = 12

MIN_SAMPLES_LEAF = 2

TRAIN_RATIO = 0.80

FORECAST_MONTHS = 360


# ============================================================
# JUDUL
# ============================================================

st.title(
    "🌦️ Dashboard Iklim Wilayah Pesisir Pulau Sumatera"
)

st.markdown(
    """
    **Machine Learning untuk Memprediksi Perubahan Iklim
    Wilayah Pesisir Pantai Pulau Sumatera**

    Analisis Temporal Jangka Panjang Berbasis
    **Random Forest (1985–2025)**
    """
)


# ============================================================
# FUNGSI MEMBACA EXCEL
# ============================================================

@st.cache_data
def load_excel(file_path):

    raw = pd.read_excel(
        file_path,
        header=None
    )

    required_columns = [
        "YEAR",
        "DOY",
        "TN",
        "TX",
        "TAVG",
        "RH_AVG",
        "RR",
        "SS",
        "FF_X",
        "FF_AVG"
    ]

    header_row = None

    for i in range(min(40, len(raw))):

        row_values = (
            raw.iloc[i]
            .astype(str)
            .str.strip()
            .str.upper()
            .tolist()
        )

        matches = sum(
            1
            for col in required_columns
            if col in row_values
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
# MEMBUAT TANGGAL
# ============================================================

def create_date_column(df):

    df = df.copy()

    # DATE
    if "DATE" in df.columns:

        df["DATE"] = pd.to_datetime(
            df["DATE"],
            errors="coerce"
        )

        return df

    # TANGGAL
    if "TANGGAL" in df.columns:

        df["DATE"] = pd.to_datetime(
            df["TANGGAL"],
            errors="coerce"
        )

        return df

    # YEAR + DOY
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

        return df

    df["DATE"] = pd.NaT

    return df


# ============================================================
# MEMBERSIHKAN DATA
# ============================================================

def clean_data(df):

    df = df.copy()

    df = create_date_column(df)

    for col in PARAMETER_INFO:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    df = df.dropna(
        subset=["DATE"]
    )

    df = df.sort_values(
        "DATE"
    )

    return df


# ============================================================
# AGREGASI BULANAN
# ============================================================

def monthly_aggregation(df):

    df = df.copy()

    df["MONTH"] = (
        df["DATE"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    aggregation = {}

    # Temperatur
    for col in [
        "TN",
        "TX",
        "TAVG",
        "RH_AVG",
        "FF_X",
        "FF_AVG"
    ]:

        if col in df.columns:

            aggregation[col] = "mean"

    # Curah hujan
    if "RR" in df.columns:

        aggregation["RR"] = "sum"

    # Penyinaran matahari
    if "SS" in df.columns:

        aggregation["SS"] = "sum"

    monthly = (
        df.groupby("MONTH")
        .agg(aggregation)
        .reset_index()
    )

    monthly = monthly.sort_values(
        "MONTH"
    )

    return monthly


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(monthly, target):

    data = monthly[
        [
            "MONTH",
            target
        ]
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

    data["month_number"] = (
        data["MONTH"].dt.month
    )

    data["month_sin"] = np.sin(
        2 * np.pi *
        data["month_number"] / 12
    )

    data["month_cos"] = np.cos(
        2 * np.pi *
        data["month_number"] / 12
    )

    data = data.dropna()

    return data


# ============================================================
# MEMBUAT MODEL
# ============================================================

def train_random_forest(monthly, target):

    data = create_features(
        monthly,
        target
    )

    feature_columns = [
        "lag1",
        "lag2",
        "lag3",
        "rolling_mean_3",
        "month_sin",
        "month_cos"
    ]

    X = data[
        feature_columns
    ].values

    y = data[
        [target]
    ].values

    # --------------------------------------------------------
    # Split time series 80 : 20
    # --------------------------------------------------------

    split_index = int(
        len(data) * TRAIN_RATIO
    )

    X_train = X[:split_index]

    X_test = X[split_index:]

    y_train = y[:split_index]

    y_test = y[split_index:]

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    scaler_X = MinMaxScaler()

    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(
        X_train
    )

    X_test_scaled = scaler_X.transform(
        X_test
    )

    y_train_scaled = scaler_y.fit_transform(
        y_train
    ).ravel()

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Prediksi test
    # --------------------------------------------------------

    y_pred_scaled = model.predict(
        X_test_scaled
    )

    y_pred = scaler_y.inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    ).ravel()

    y_test_original = y_test.ravel()

    # --------------------------------------------------------
    # Evaluasi
    # --------------------------------------------------------

    rmse = np.sqrt(
        mean_squared_error(
            y_test_original,
            y_pred
        )
    )

    mae = mean_absolute_error(
        y_test_original,
        y_pred
    )

    r2 = r2_score(
        y_test_original,
        y_pred
    )

    evaluation = {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }

    # --------------------------------------------------------
    # Data aktual vs prediksi
    # --------------------------------------------------------

    evaluation_data = data.iloc[
        split_index:
    ].copy()

    evaluation_data["Actual"] = (
        y_test_original
    )

    evaluation_data["Predicted"] = (
        y_pred
    )

    return (
        model,
        scaler_X,
        scaler_y,
        evaluation,
        evaluation_data,
        feature_columns
    )


# ============================================================
# FORECAST 2026–2055
# ============================================================

def forecast_future(
    monthly,
    target,
    model,
    scaler_X,
    scaler_y,
    months=360
):

    history = monthly[
        [
            "MONTH",
            target
        ]
    ].dropna().copy()

    history = history.sort_values(
        "MONTH"
    )

    values = (
        history[target]
        .astype(float)
        .tolist()
    )

    last_month = (
        history["MONTH"]
        .max()
    )

    future_dates = pd.date_range(
        start=last_month + pd.offsets.MonthBegin(1),
        periods=months,
        freq="MS"
    )

    predictions = []

    for date in future_dates:

        lag1 = values[-1]

        lag2 = values[-2]

        lag3 = values[-3]

        rolling_mean_3 = np.mean(
            values[-3:]
        )

        month_number = date.month

        month_sin = np.sin(
            2 * np.pi *
            month_number / 12
        )

        month_cos = np.cos(
            2 * np.pi *
            month_number / 12
        )

        X_future = np.array(
            [
                [
                    lag1,
                    lag2,
                    lag3,
                    rolling_mean_3,
                    month_sin,
                    month_cos
                ]
            ]
        )

        X_future_scaled = scaler_X.transform(
            X_future
        )

        pred_scaled = model.predict(
            X_future_scaled
        )

        prediction = scaler_y.inverse_transform(
            pred_scaled.reshape(-1, 1)
        )[0, 0]

        predictions.append(
            prediction
        )

        values.append(
            prediction
        )

    forecast = pd.DataFrame(
        {
            "MONTH": future_dates,
            target: predictions
        }
    )

    return forecast


# ============================================================
# FORECAST TAHUNAN
# ============================================================

def annual_forecast(forecast, target):

    data = forecast.copy()

    data["YEAR"] = (
        data["MONTH"].dt.year
    )

    # RR dan SS merupakan akumulasi
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
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Pengaturan"
)


station = st.sidebar.selectbox(
    "Pilih Stasiun",
    list(DATASETS.keys())
)


start_date = st.sidebar.date_input(
    "Tanggal Awal",
    value=pd.Timestamp(
        "1985-01-01"
    ).date()
)


end_date = st.sidebar.date_input(
    "Tanggal Akhir",
    value=pd.Timestamp(
        "2025-12-31"
    ).date()
)


# ============================================================
# VALIDASI TANGGAL
# ============================================================

if start_date >= end_date:

    st.error(
        "Tanggal awal harus lebih kecil dari tanggal akhir."
    )

    st.stop()


# ============================================================
# BACA DATA
# ============================================================

file_path = Path(
    DATASETS[station]
)


if not file_path.exists():

    st.error(
        f"File {file_path.name} tidak ditemukan."
    )

    st.stop()


with st.spinner(
    "Membaca dataset..."
):

    df_raw = load_excel(
        file_path
    )


# ============================================================
# CLEANING
# ============================================================

df = clean_data(
    df_raw
)


df = df[
    (df["DATE"] >= pd.Timestamp(start_date))
    &
    (df["DATE"] <= pd.Timestamp(end_date))
].copy()


if df.empty:

    st.error(
        "Tidak ada data pada rentang tanggal tersebut."
    )

    st.stop()


# ============================================================
# DATA BULANAN
# ============================================================

df_monthly = monthly_aggregation(
    df
)


available_parameters = [
    col
    for col in PARAMETER_INFO
    if col in df_monthly.columns
]


if not available_parameters:

    st.error(
        "Parameter iklim tidak ditemukan."
    )

    st.stop()


# ============================================================
# INFORMASI DATA
# ============================================================

st.success(
    f"Dataset **{station}** berhasil diproses."
)


c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "Data Harian",
        f"{len(df):,}"
    )


with c2:

    st.metric(
        "Data Bulanan",
        f"{len(df_monthly):,}"
    )


with c3:

    st.metric(
        "Periode Awal",
        df_monthly["MONTH"]
        .min()
        .strftime("%Y-%m")
    )


with c4:

    st.metric(
        "Periode Akhir",
        df_monthly["MONTH"]
        .max()
        .strftime("%Y-%m")
    )


# ============================================================
# MENU TAB
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Dashboard",
        "🤖 Random Forest",
        "📈 Evaluasi",
        "🔮 Forecast 2026–2055",
        "📋 Data"
    ]
)


# ============================================================
# TAB 1 — DASHBOARD
# ============================================================

with tab1:

    st.subheader(
        "📊 Data Iklim Bulanan"
    )

    parameter_dashboard = st.selectbox(
        "Pilih Parameter",
        available_parameters,
        format_func=lambda x:
            f"{x} — {PARAMETER_INFO[x]}",
        key="dashboard_parameter"
    )


    chart_data = df_monthly[
        [
            "MONTH",
            parameter_dashboard
        ]
    ].dropna()


    fig = px.line(
        chart_data,
        x="MONTH",
        y=parameter_dashboard,
        title=(
            f"{PARAMETER_INFO[parameter_dashboard]} "
            f"— {station}"
        )
    )


    fig.update_layout(
        xaxis_title="Waktu",
        yaxis_title=parameter_dashboard,
        hovermode="x unified"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # Statistik
    st.subheader(
        "📌 Statistik Deskriptif"
    )


    stats = chart_data[
        parameter_dashboard
    ].describe()


    s1, s2, s3, s4 = st.columns(4)


    with s1:

        st.metric(
            "Rata-rata",
            f"{stats['mean']:.2f}"
        )


    with s2:

        st.metric(
            "Minimum",
            f"{stats['min']:.2f}"
        )


    with s3:

        st.metric(
            "Maksimum",
            f"{stats['max']:.2f}"
        )


    with s4:

        st.metric(
            "Std. Deviasi",
            f"{stats['std']:.2f}"
        )


# ============================================================
# TAB 2 — RANDOM FOREST
# ============================================================

with tab2:

    st.subheader(
        "🤖 Pemodelan Random Forest"
    )

    st.write(
        """
        Model Random Forest digunakan untuk mempelajari
        pola temporal data iklim bulanan periode 1985–2025.
        """
    )


    st.info(
        """
        **Feature Engineering**

        • Lag 1 bulan  
        • Lag 2 bulan  
        • Lag 3 bulan  
        • Rolling Mean 3 bulan  
        • Month Sin  
        • Month Cos
        """
    )


    parameter_model = st.selectbox(
        "Pilih Parameter untuk Pemodelan",
        available_parameters,
        format_func=lambda x:
            f"{x} — {PARAMETER_INFO[x]}",
        key="model_parameter"
    )


    st.write(
        "**Konfigurasi Random Forest**"
    )


    config1, config2, config3, config4 = st.columns(4)


    with config1:

        st.metric(
            "Jumlah Trees",
            N_ESTIMATORS
        )


    with config2:

        st.metric(
            "Max Depth",
            MAX_DEPTH
        )


    with config3:

        st.metric(
            "Train",
            "80%"
        )


    with config4:

        st.metric(
            "Test",
            "20%"
        )


    train_button = st.button(
        "🚀 Jalankan Random Forest",
        type="primary"
    )


    if train_button:

        with st.spinner(
            f"Melatih Random Forest untuk {parameter_model}..."
        ):

            (
                model,
                scaler_X,
                scaler_y,
                evaluation,
                evaluation_data,
                feature_columns
            ) = train_random_forest(
                df_monthly,
                parameter_model
            )


            forecast = forecast_future(
                df_monthly,
                parameter_model,
                model,
                scaler_X,
                scaler_y,
                FORECAST_MONTHS
            )


        # Menyimpan hasil ke session
        st.session_state[
            "model_result"
        ] = {

            "station": station,

            "parameter": parameter_model,

            "model": model,

            "scaler_X": scaler_X,

            "scaler_y": scaler_y,

            "evaluation": evaluation,

            "evaluation_data": evaluation_data,

            "forecast": forecast,

            "feature_columns": feature_columns
        }


        st.success(
            f"Model Random Forest {parameter_model} selesai dilatih."
        )


    # --------------------------------------------------------
    # HASIL MODEL
    # --------------------------------------------------------

    if "model_result" in st.session_state:

        result = st.session_state[
            "model_result"
        ]


        if (
            result["station"] == station
            and
            result["parameter"] == parameter_model
        ):

            st.success(
                "Hasil model tersedia."
            )


            st.write(
                "**Feature yang digunakan:**"
            )

            st.code(
                ", ".join(
                    result["feature_columns"]
                )
            )


            evaluation = result[
                "evaluation"
            ]


            e1, e2, e3 = st.columns(3)


            with e1:

                st.metric(
                    "RMSE",
                    f"{evaluation['RMSE']:.4f}"
                )


            with e2:

                st.metric(
                    "MAE",
                    f"{evaluation['MAE']:.4f}"
                )


            with e3:

                st.metric(
                    "R²",
                    f"{evaluation['R2']:.4f}"
                )


        else:

            st.info(
                "Klik tombol Random Forest untuk melatih parameter yang dipilih."
            )


# ============================================================
# TAB 3 — EVALUASI
# ============================================================

with tab3:

    st.subheader(
        "📈 Validasi dan Evaluasi Model"
    )


    if "model_result" not in st.session_state:

        st.info(
            """
            Belum ada model yang dijalankan.

            Masuk ke tab **🤖 Random Forest**, pilih parameter,
            kemudian klik **Jalankan Random Forest**.
            """
        )

    else:

        result = st.session_state[
            "model_result"
        ]


        if result["station"] != station:

            st.info(
                "Hasil model yang tersedia berasal dari stasiun lain."
            )

        else:

            evaluation = result[
                "evaluation"
            ]

            evaluation_data = result[
                "evaluation_data"
            ].copy()


            st.write(
                f"### {result['parameter']} — {station}"
            )


            a1, a2, a3 = st.columns(3)


            with a1:

                st.metric(
                    "RMSE",
                    f"{evaluation['RMSE']:.4f}"
                )


            with a2:

                st.metric(
                    "MAE",
                    f"{evaluation['MAE']:.4f}"
                )


            with a3:

                st.metric(
                    "R²",
                    f"{evaluation['R2']:.4f}"
                )


            # ------------------------------------------------
            # Grafik aktual vs prediksi
            # ------------------------------------------------

            plot_eval = evaluation_data[
                [
                    "MONTH",
                    "Actual",
                    "Predicted"
                ]
            ].copy()


            plot_long = plot_eval.melt(
                id_vars="MONTH",
                value_vars=[
                    "Actual",
                    "Predicted"
                ],
                var_name="Jenis",
                value_name="Nilai"
            )


            fig_eval = px.line(
                plot_long,
                x="MONTH",
                y="Nilai",
                color="Jenis",
                title="Aktual vs Prediksi Data Uji"
            )


            fig_eval.update_layout(
                xaxis_title="Waktu",
                yaxis_title=result["parameter"],
                hovermode="x unified"
            )


            st.plotly_chart(
                fig_eval,
                use_container_width=True
            )


            # ------------------------------------------------
            # Tabel
            # ------------------------------------------------

            st.subheader(
                "📋 Data Aktual dan Prediksi"
            )


            st.dataframe(
                evaluation_data[
                    [
                        "MONTH",
                        "Actual",
                        "Predicted"
                    ]
                ],
                use_container_width=True
            )


# ============================================================
# TAB 4 — FORECAST
# ============================================================

with tab4:

    st.subheader(
        "🔮 Prediksi Iklim 2026–2055"
    )


    if "model_result" not in st.session_state:

        st.info(
            "Jalankan model terlebih dahulu pada tab Random Forest."
        )

    else:

        result = st.session_state[
            "model_result"
        ]


        if result["station"] != station:

            st.info(
                "Model yang tersedia berasal dari stasiun lain."
            )

        else:

            forecast = result[
                "forecast"
            ].copy()


            target = result[
                "parameter"
            ]


            st.success(
                f"Forecast {target} untuk "
                f"{station} tersedia sebanyak "
                f"{len(forecast)} bulan."
            )


            # ------------------------------------------------
            # Grafik forecast
            # ------------------------------------------------

            fig_forecast = px.line(
                forecast,
                x="MONTH",
                y=target,
                title=(
                    f"Forecast {PARAMETER_INFO[target]} "
                    f"2026–2055"
                )
            )


            fig_forecast.update_layout(
                xaxis_title="Tahun",
                yaxis_title=target,
                hovermode="x unified"
            )


            st.plotly_chart(
                fig_forecast,
                use_container_width=True
            )


            # ------------------------------------------------
            # Forecast tahunan
            # ------------------------------------------------

            st.subheader(
                "📅 Forecast Tahunan"
            )


            annual = annual_forecast(
                forecast,
                target
            )


            st.dataframe(
                annual,
                use_container_width=True
            )


            # ------------------------------------------------
            # Download
            # ------------------------------------------------

            csv_forecast = forecast.copy()

            csv_forecast["MONTH"] = (
                csv_forecast["MONTH"]
                .dt.strftime("%Y-%m-%d")
            )


            csv_bytes = (
                csv_forecast
                .to_csv(index=False)
                .encode("utf-8")
            )


            st.download_button(
                "📥 Download Forecast Bulanan CSV",
                data=csv_bytes,
                file_name=(
                    station.lower()
                    .replace(" ", "_")
                    + "_"
                    + target.lower()
                    + "_forecast_2026_2055.csv"
                ),
                mime="text/csv"
            )


            csv_annual = (
                annual
                .to_csv(index=False)
                .encode("utf-8")
            )


            st.download_button(
                "📥 Download Forecast Tahunan CSV",
                data=csv_annual,
                file_name=(
                    station.lower()
                    .replace(" ", "_")
                    + "_"
                    + target.lower()
                    + "_forecast_tahunan_2026_2055.csv"
                ),
                mime="text/csv"
            )


# ============================================================
# TAB 5 — DATA
# ============================================================

with tab5:

    st.subheader(
        "📋 Data Bulanan 1985–2025"
    )


    display_data = df_monthly.copy()


    display_data["MONTH"] = (
        display_data["MONTH"]
        .dt.strftime("%Y-%m")
    )


    st.dataframe(
        display_data,
        use_container_width=True,
        height=500
    )


    # --------------------------------------------------------
    # Missing value
    # --------------------------------------------------------

    st.subheader(
        "🔎 Pemeriksaan Missing Value"
    )


    missing = pd.DataFrame(
        {
            "Kolom": df_monthly.columns,
            "Missing": [
                df_monthly[col].isna().sum()
                for col in df_monthly.columns
            ]
        }
    )


    missing["Persentase (%)"] = (
        missing["Missing"]
        / len(df_monthly)
        * 100
    ).round(2)


    st.dataframe(
        missing,
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    """
    Dashboard Iklim Wilayah Pesisir Pulau Sumatera |
    Data historis 1985–2025 |
    Random Forest |
    Forecast 2026–2055
    """
)
