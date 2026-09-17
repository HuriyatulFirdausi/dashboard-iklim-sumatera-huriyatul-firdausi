import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path


# ============================================================
# KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(
    page_title="Dashboard Iklim Sumatera",
    page_icon="🌦️",
    layout="wide"
)


# ============================================================
# JUDUL APLIKASI
# ============================================================

st.title("🌦️ Dashboard Iklim Wilayah Pesisir Pulau Sumatera")

st.markdown(
    """
    **Machine Learning untuk Memprediksi Perubahan Iklim
    Wilayah Pesisir Pantai Pulau Sumatera**

    Analisis Temporal Jangka Panjang Berbasis **Random Forest (1985–2025)**
    """
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
# FUNGSI MEMBACA EXCEL
# ============================================================

@st.cache_data
def load_excel(file_path):

    try:

        # Membaca file tanpa menentukan header
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

        # Mencari baris header
        for i in range(min(40, len(raw))):

            row_values = (
                raw.iloc[i]
                .astype(str)
                .str.strip()
                .str.upper()
                .tolist()
            )

            jumlah_cocok = sum(
                1
                for col in required_columns
                if col in row_values
            )

            if jumlah_cocok >= 5:
                header_row = i
                break

        # Membaca kembali dengan header yang ditemukan
        if header_row is not None:

            df = pd.read_excel(
                file_path,
                header=header_row
            )

        else:

            df = pd.read_excel(
                file_path
            )

        # Membersihkan nama kolom
        df.columns = (
            df.columns
            .astype(str)
            .str.strip()
            .str.upper()
        )

        return df, None

    except Exception as e:

        return None, str(e)


# ============================================================
# FUNGSI MEMBUAT KOLOM TANGGAL
# ============================================================

def create_date_column(df):

    df = df.copy()

    # --------------------------------------------------------
    # Jika sudah terdapat DATE
    # --------------------------------------------------------

    if "DATE" in df.columns:

        df["DATE"] = pd.to_datetime(
            df["DATE"],
            errors="coerce"
        )

        return df


    # --------------------------------------------------------
    # Jika terdapat TANGGAL
    # --------------------------------------------------------

    if "TANGGAL" in df.columns:

        df["DATE"] = pd.to_datetime(
            df["TANGGAL"],
            errors="coerce"
        )

        return df


    # --------------------------------------------------------
    # YEAR + DOY
    # --------------------------------------------------------

    if "YEAR" in df.columns and "DOY" in df.columns:

        df["YEAR"] = pd.to_numeric(
            df["YEAR"],
            errors="coerce"
        )

        df["DOY"] = pd.to_numeric(
            df["DOY"],
            errors="coerce"
        )

        valid = (
            df["YEAR"].notna()
            &
            df["DOY"].notna()
        )

        df["DATE"] = pd.NaT

        df.loc[valid, "DATE"] = (
            pd.to_datetime(
                df.loc[valid, "YEAR"].astype(int).astype(str),
                format="%Y",
                errors="coerce"
            )
            +
            pd.to_timedelta(
                df.loc[valid, "DOY"] - 1,
                unit="D"
            )
        )

        return df


    # --------------------------------------------------------
    # Jika tidak ditemukan
    # --------------------------------------------------------

    df["DATE"] = pd.NaT

    return df


# ============================================================
# FUNGSI PEMBERSIHAN DATA
# ============================================================

def clean_data(df):

    df = df.copy()

    # Membuat tanggal
    df = create_date_column(df)

    # Parameter yang digunakan
    parameters = list(PARAMETER_INFO.keys())

    # Mengubah parameter menjadi numerik
    for col in parameters:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    # Menghapus baris tanpa tanggal
    df = df.dropna(
        subset=["DATE"]
    )

    # Mengurutkan tanggal
    df = df.sort_values(
        "DATE"
    )

    # Menghapus tanggal duplikat jika ada
    # Tidak langsung menghapus seluruh data,
    # hanya digunakan sebagai informasi kualitas data.

    return df


# ============================================================
# FUNGSI AGREGASI BULANAN
# ============================================================

def monthly_aggregation(df):

    df = df.copy()

    df["MONTH"] = (
        df["DATE"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    aggregation = {}

    # --------------------------------------------------------
    # Temperatur
    # --------------------------------------------------------

    if "TN" in df.columns:
        aggregation["TN"] = "mean"

    if "TX" in df.columns:
        aggregation["TX"] = "mean"

    if "TAVG" in df.columns:
        aggregation["TAVG"] = "mean"

    # --------------------------------------------------------
    # Kelembapan
    # --------------------------------------------------------

    if "RH_AVG" in df.columns:
        aggregation["RH_AVG"] = "mean"

    # --------------------------------------------------------
    # Curah hujan
    # --------------------------------------------------------

    if "RR" in df.columns:
        aggregation["RR"] = "sum"

    # --------------------------------------------------------
    # Penyinaran matahari
    # --------------------------------------------------------

    if "SS" in df.columns:
        aggregation["SS"] = "sum"

    # --------------------------------------------------------
    # Angin
    # --------------------------------------------------------

    if "FF_X" in df.columns:
        aggregation["FF_X"] = "mean"

    if "FF_AVG" in df.columns:
        aggregation["FF_AVG"] = "mean"

    # --------------------------------------------------------
    # Agregasi
    # --------------------------------------------------------

    if not aggregation:

        return pd.DataFrame()

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
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Pengaturan")


# ------------------------------------------------------------
# PILIH STASIUN
# ------------------------------------------------------------

station = st.sidebar.selectbox(
    "Pilih Stasiun",
    list(DATASETS.keys())
)


# ------------------------------------------------------------
# TANGGAL
# ------------------------------------------------------------

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

if start_date > end_date:

    st.error(
        "Tanggal awal tidak boleh lebih besar dari tanggal akhir."
    )

    st.stop()


# ============================================================
# MEMBACA DATASET
# ============================================================

file_name = DATASETS[station]

file_path = Path(file_name)


if not file_path.exists():

    st.error(
        f"File **{file_name}** tidak ditemukan."
    )

    st.info(
        "Pastikan file Excel berada di folder utama repository."
    )

    st.stop()


df_raw, error = load_excel(
    file_path
)


# ============================================================
# ERROR DATASET
# ============================================================

if error is not None:

    st.error(
        f"Gagal membaca dataset: {error}"
    )

    st.stop()


# ============================================================
# MEMBERSIHKAN DATA
# ============================================================

df = clean_data(
    df_raw
)


# ============================================================
# FILTER TANGGAL
# ============================================================

df = df[
    (df["DATE"] >= pd.Timestamp(start_date))
    &
    (df["DATE"] <= pd.Timestamp(end_date))
].copy()


# ============================================================
# CEK DATA
# ============================================================

if df.empty:

    st.warning(
        "Tidak terdapat data pada rentang tanggal yang dipilih."
    )

    st.stop()


# ============================================================
# AGREGASI BULANAN
# ============================================================

df_monthly = monthly_aggregation(
    df
)


# ============================================================
# HEADER INFORMASI
# ============================================================

st.success(
    f"Dataset **{station}** berhasil diproses."
)


# ============================================================
# METRIK
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Data Harian",
        f"{len(df):,}"
    )


with col2:

    st.metric(
        "Data Bulanan",
        f"{len(df_monthly):,}"
    )


with col3:

    st.metric(
        "Tanggal Awal",
        df["DATE"].min().strftime("%d-%m-%Y")
    )


with col4:

    st.metric(
        "Tanggal Akhir",
        df["DATE"].max().strftime("%d-%m-%Y")
    )


# ============================================================
# TAB
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Dashboard",
        "📋 Data Bulanan",
        "🔎 Pemeriksaan Data",
        "📥 Download"
    ]
)


# ============================================================
# TAB 1 — DASHBOARD
# ============================================================

with tab1:

    st.subheader(
        "📈 Visualisasi Data Iklim Bulanan"
    )

    available_parameters = [
        col
        for col in PARAMETER_INFO.keys()
        if col in df_monthly.columns
    ]


    if not available_parameters:

        st.warning(
            "Tidak ditemukan parameter iklim."
        )

    else:

        parameter = st.selectbox(
            "Pilih Parameter Iklim",
            available_parameters,
            format_func=lambda x:
                f"{x} — {PARAMETER_INFO[x]}"
        )


        chart_data = df_monthly[
            [
                "MONTH",
                parameter
            ]
        ].dropna()


        if not chart_data.empty:

            fig = px.line(
                chart_data,
                x="MONTH",
                y=parameter,
                title=(
                    f"{PARAMETER_INFO[parameter]} "
                    f"Bulanan — {station}"
                ),
                markers=False
            )

            fig.update_layout(
                xaxis_title="Waktu",
                yaxis_title=(
                    f"{parameter} "
                    f"({PARAMETER_INFO[parameter]})"
                ),
                hovermode="x unified"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


            # ------------------------------------------------
            # STATISTIK
            # ------------------------------------------------

            st.subheader(
                "📌 Statistik Deskriptif"
            )

            stat_data = chart_data[
                parameter
            ].describe()

            stat_col1, stat_col2, stat_col3, stat_col4 = (
                st.columns(4)
            )

            with stat_col1:

                st.metric(
                    "Rata-rata",
                    f"{stat_data['mean']:.2f}"
                )

            with stat_col2:

                st.metric(
                    "Minimum",
                    f"{stat_data['min']:.2f}"
                )

            with stat_col3:

                st.metric(
                    "Maksimum",
                    f"{stat_data['max']:.2f}"
                )

            with stat_col4:

                st.metric(
                    "Standar Deviasi",
                    f"{stat_data['std']:.2f}"
                )


# ============================================================
# TAB 2 — DATA BULANAN
# ============================================================

with tab2:

    st.subheader(
        "📋 Data Iklim Bulanan"
    )

    st.write(
        "Data harian telah diagregasi menjadi data bulanan."
    )

    display_monthly = df_monthly.copy()

    display_monthly["MONTH"] = (
        display_monthly["MONTH"]
        .dt.strftime("%Y-%m")
    )

    st.dataframe(
        display_monthly,
        use_container_width=True,
        height=500
    )


# ============================================================
# TAB 3 — PEMERIKSAAN DATA
# ============================================================

with tab3:

    st.subheader(
        "🔎 Pemeriksaan Dataset"
    )


    # --------------------------------------------------------
    # Kolom
    # --------------------------------------------------------

    st.write(
        "**Kolom yang terbaca:**"
    )

    st.write(
        list(df.columns)
    )


    # --------------------------------------------------------
    # Parameter
    # --------------------------------------------------------

    st.write(
        "**Parameter iklim yang tersedia:**"
    )

    available_parameters = [
        col
        for col in PARAMETER_INFO.keys()
        if col in df.columns
    ]

    st.write(
        available_parameters
    )


    # --------------------------------------------------------
    # Missing Value
    # --------------------------------------------------------

    st.subheader(
        "Missing Value"
    )

    missing_data = pd.DataFrame(
        {
            "Kolom": df.columns,
            "Jumlah Missing": [
                df[col].isna().sum()
                for col in df.columns
            ],
            "Persentase Missing (%)": [
                round(
                    df[col].isna().mean() * 100,
                    2
                )
                for col in df.columns
            ]
        }
    )

    st.dataframe(
        missing_data,
        use_container_width=True
    )


    # --------------------------------------------------------
    # Duplikat
    # --------------------------------------------------------

    st.subheader(
        "Duplikasi Data"
    )

    duplicate_count = df.duplicated(
        subset=["DATE"]
    ).sum()

    st.write(
        f"Jumlah tanggal duplikat: **{duplicate_count:,}**"
    )


    # --------------------------------------------------------
    # Preview data bersih
    # --------------------------------------------------------

    st.subheader(
        "Preview Data Bersih"
    )

    st.dataframe(
        df.head(20),
        use_container_width=True
    )


# ============================================================
# TAB 4 — DOWNLOAD
# ============================================================

with tab4:

    st.subheader(
        "📥 Download Data"
    )


    # --------------------------------------------------------
    # CSV Data Bulanan
    # --------------------------------------------------------

    csv_monthly = df_monthly.copy()

    csv_monthly["MONTH"] = (
        csv_monthly["MONTH"]
        .dt.strftime("%Y-%m-%d")
    )

    csv_data = csv_monthly.to_csv(
        index=False
    ).encode(
        "utf-8"
    )


    st.download_button(
        label="📥 Download Data Bulanan CSV",
        data=csv_data,
        file_name=(
            station
            .lower()
            .replace(" ", "_")
            + "_data_bulanan.csv"
        ),
        mime="text/csv"
    )


    # --------------------------------------------------------
    # CSV Data Harian
    # --------------------------------------------------------

    csv_daily = df.copy()

    csv_daily["DATE"] = (
        csv_daily["DATE"]
        .dt.strftime("%Y-%m-%d")
    )

    csv_daily_data = csv_daily.to_csv(
        index=False
    ).encode(
        "utf-8"
    )


    st.download_button(
        label="📥 Download Data Harian CSV",
        data=csv_daily_data,
        file_name=(
            station
            .lower()
            .replace(" ", "_")
            + "_data_harian.csv"
        ),
        mime="text/csv"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Tahap 3 — Pengolahan data harian menjadi data bulanan "
    "periode 1985–2025. Pemodelan Random Forest akan "
    "ditambahkan setelah struktur data tervalidasi."
)
