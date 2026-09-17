import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# =========================================================
# KONFIGURASI
# =========================================================

st.set_page_config(
    page_title="Dashboard Iklim Sumatera",
    page_icon="🌦️",
    layout="wide"
)

# =========================================================
# JUDUL
# =========================================================

st.title("🌦️ Dashboard Iklim Wilayah Pesisir Pulau Sumatera")

st.markdown(
    """
    **Machine Learning untuk Memprediksi Perubahan Iklim Wilayah Pesisir Pantai Pulau Sumatera**

    Analisis Temporal Jangka Panjang Berbasis **Random Forest (1985–2025)**
    """
)

# =========================================================
# DATASET
# =========================================================

DATASETS = {
    "Stasiun Minangkabau": "minang kabau data FIX.xlsx",
    "Stasiun Pesawaran": "pesawaran data FIX.xlsx",
    "Stasiun Maritim Panjang": "Maritim panjang data FIX.xlsx",
}

# =========================================================
# FUNGSI MEMBACA EXCEL
# =========================================================

@st.cache_data
def load_excel(file_path):

    try:
        # Membaca seluruh sheet pertama tanpa langsung menentukan header
        raw = pd.read_excel(file_path, header=None)

        # Mencari baris yang kemungkinan merupakan header
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
            "FF_AVG",
        ]

        header_row = None

        for i in range(min(30, len(raw))):
            row_values = raw.iloc[i].astype(str).str.upper().tolist()

            matches = sum(
                1 for col in required_columns
                if col in row_values
            )

            if matches >= 5:
                header_row = i
                break

        # Jika header ditemukan
        if header_row is not None:
            df = pd.read_excel(file_path, header=header_row)
        else:
            # Jika tidak ditemukan, coba membaca header pertama
            df = pd.read_excel(file_path)

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


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ Pengaturan")

station = st.sidebar.selectbox(
    "Pilih Stasiun",
    list(DATASETS.keys())
)

start_date = st.sidebar.date_input(
    "Tanggal Awal",
    value=pd.Timestamp("1985-01-01").date()
)

end_date = st.sidebar.date_input(
    "Tanggal Akhir",
    value=pd.Timestamp("2025-12-31").date()
)

# =========================================================
# MEMBACA DATA
# =========================================================

file_name = DATASETS[station]

file_path = Path(file_name)

if not file_path.exists():

    st.error(
        f"File **{file_name}** tidak ditemukan di repository."
    )

    st.info(
        "Pastikan nama file Excel di GitHub sama persis dengan nama yang digunakan aplikasi."
    )

    st.stop()

df, error = load_excel(file_path)

if error is not None:

    st.error(f"Gagal membaca dataset: {error}")

    st.stop()

# =========================================================
# INFORMASI DATA
# =========================================================

st.success(
    f"Dataset **{station}** berhasil dibaca."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Jumlah Baris",
        f"{len(df):,}"
    )

with col2:
    st.metric(
        "Jumlah Kolom",
        len(df.columns)
    )

with col3:
    st.metric(
        "Stasiun",
        station
    )

# =========================================================
# KOLOM DATA
# =========================================================

st.subheader("📋 Struktur Dataset")

st.write(
    "Kolom yang terbaca dari dataset:"
)

st.write(
    list(df.columns)
)

# =========================================================
# PREVIEW DATA
# =========================================================

st.subheader("🔎 Preview Data")

st.dataframe(
    df.head(20),
    use_container_width=True
)

# =========================================================
# MEMBUAT TANGGAL
# =========================================================

df_plot = df.copy()

try:

    if "DATE" in df_plot.columns:

        df_plot["DATE"] = pd.to_datetime(
            df_plot["DATE"],
            errors="coerce"
        )

    elif "TANGGAL" in df_plot.columns:

        df_plot["DATE"] = pd.to_datetime(
            df_plot["TANGGAL"],
            errors="coerce"
        )

    elif "YEAR" in df_plot.columns and "DOY" in df_plot.columns:

        df_plot["YEAR"] = pd.to_numeric(
            df_plot["YEAR"],
            errors="coerce"
        )

        df_plot["DOY"] = pd.to_numeric(
            df_plot["DOY"],
            errors="coerce"
        )

        df_plot["DATE"] = pd.to_datetime(
            df_plot["YEAR"].astype("Int64").astype(str),
            format="%Y",
            errors="coerce"
        ) + pd.to_timedelta(
            df_plot["DOY"] - 1,
            unit="D"
        )

    else:

        df_plot["DATE"] = pd.NaT

except Exception:

    df_plot["DATE"] = pd.NaT


# =========================================================
# FILTER TANGGAL
# =========================================================

if df_plot["DATE"].notna().any():

    df_plot = df_plot[
        (df_plot["DATE"] >= pd.Timestamp(start_date))
        &
        (df_plot["DATE"] <= pd.Timestamp(end_date))
    ]

# =========================================================
# PILIH PARAMETER
# =========================================================

st.subheader("📈 Visualisasi Parameter Iklim")

available_parameters = [
    "TN",
    "TX",
    "TAVG",
    "RH_AVG",
    "RR",
    "SS",
    "FF_X",
    "FF_AVG",
]

available_parameters = [
    col for col in available_parameters
    if col in df_plot.columns
]

if not available_parameters:

    st.warning(
        "Belum ditemukan kolom parameter iklim yang sesuai."
    )

else:

    parameter = st.selectbox(
        "Pilih Parameter",
        available_parameters
    )

    df_chart = df_plot[
        ["DATE", parameter]
    ].copy()

    df_chart[parameter] = pd.to_numeric(
        df_chart[parameter],
        errors="coerce"
    )

    df_chart = df_chart.dropna()

    if len(df_chart) > 0:

        fig = px.line(
            df_chart,
            x="DATE",
            y=parameter,
            title=f"{parameter} - {station}"
        )

        fig.update_layout(
            xaxis_title="Tanggal",
            yaxis_title=parameter
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning(
            f"Tidak ada data numerik yang dapat divisualisasikan untuk {parameter}."
        )

# =========================================================
# CATATAN
# =========================================================

st.divider()

st.caption(
    "Tahap 1: pembacaan dan pemeriksaan dataset. "
    "Pemodelan Random Forest akan ditambahkan pada tahap berikutnya."
)
