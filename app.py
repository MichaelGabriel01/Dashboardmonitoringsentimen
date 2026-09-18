import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from google_play_scraper import reviews_all, reviews, Sort
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm


# === PATH FILE CSV ===
RESULTS_PATH = "indobert_sentiment_test_results.csv"
METRICS_PATH = "indobert_sentiment_metrics.csv"

# Dataset Real (Gold + Pseudo)
MASTER_PATH = "total_dataset.csv"
SCRAPE_FILE = "all_reviews_mybca.csv"
MODEL_DIR = "mikeyygb/finetuningmybca"
APP_ID = "com.bca.mybca.omni.android"


# ========== CONFIG & BASIC STYLE ==========
st.set_page_config(page_title="IndoBERT Live Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f7fb;
    }
    .metric-card {
        padding: 1rem 1.2rem;
        border-radius: 0.75rem;
        background-color: white;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08);
        border: 1px solid #e2e8f0;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0f172a;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# ========== HELPER: SCRAPING & DATA ==========
def load_existing_scrape():
    if Path(SCRAPE_FILE).exists():
        df = pd.read_csv(SCRAPE_FILE)
        expected_cols = ["reviewId", "content", "at"]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = None
        df = df[expected_cols].copy()
        df["reviewId"] = df["reviewId"].astype(str)
        return df
    else:
        return pd.DataFrame(columns=["reviewId", "content", "at"])


def fetch_all_reviews():
    raw = reviews_all(APP_ID, lang="id", country="id")
    df = pd.DataFrame(raw)[["reviewId", "content", "at"]].copy()
    df["reviewId"] = df["reviewId"].astype(str)
    return df


def fetch_latest(limit=300):
    raw, _ = reviews(APP_ID, lang="id", country="id", sort=Sort.NEWEST, count=limit)
    df = pd.DataFrame(raw)[["reviewId", "content", "at"]].copy()
    df["reviewId"] = df["reviewId"].astype(str)
    return df


def auto_update(limit_latest=300):
    df_old = load_existing_scrape()
    if df_old.empty:
        df_all = fetch_all_reviews()
        df_new = df_all.copy()
        df_all.to_csv(SCRAPE_FILE, index=False, encoding="utf-8")
        return df_all, df_new

    df_latest = fetch_latest(limit=limit_latest)
    existing_ids = set(df_old["reviewId"].tolist())
    df_new = df_latest[~df_latest["reviewId"].isin(existing_ids)].copy()

    if df_new.empty:
        return df_old, df_new

    df_all = pd.concat([df_old, df_new], ignore_index=True)
    df_all.to_csv(SCRAPE_FILE, index=False, encoding="utf-8")
    return df_all, df_new


# ========== HELPER: MODEL ==========
@st.cache_resource
def load_teacher_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def pseudo_label_batch(df_new, tokenizer, model, device, batch_size=16):
    texts = df_new["content"].astype(str).tolist()
    enc = tokenizer(
        texts, truncation=True, padding=True, max_length=128, return_tensors="pt"
    )
    dataset = TensorDataset(enc["input_ids"], enc["attention_mask"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Pseudo-labeling", leave=False):
            input_ids, attention_mask = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())

    df_new = df_new.copy()
    df_new["pseudo_label_id"] = all_preds
    id2label = {0: "negatif", 1: "netral", 2: "positif"}
    df_new["pseudo_label"] = df_new["pseudo_label_id"].map(id2label)
    return df_new


def merge_to_master(df_new_labeled):
    if os.path.exists(MASTER_PATH):
        df_master = pd.read_csv(MASTER_PATH)
    else:
        df_master = pd.DataFrame(
            columns=["reviewId", "content", "at", "label_id", "label_text", "source"]
        )

    if not df_master.empty:
        df_master["reviewId"] = df_master["reviewId"].astype(str)

    df_new = df_new_labeled.copy()
    df_new["reviewId"] = df_new["reviewId"].astype(str)
    df_new["label_id"] = df_new["pseudo_label_id"].astype(int)
    df_new["label_text"] = df_new["pseudo_label"].astype(str)
    df_new["source"] = "pseudo"

    if "at" not in df_new.columns:
        df_new["at"] = pd.NaT

    df_new = df_new[["reviewId", "content", "at", "label_id", "label_text", "source"]]

    if not df_master.empty:
        existing_ids = set(df_master["reviewId"].tolist())
        df_new = df_new[~df_new["reviewId"].isin(existing_ids)].copy()

    df_total = pd.concat([df_master, df_new], ignore_index=True)
    df_total = df_total.drop_duplicates(subset=["reviewId"]).reset_index(drop=True)
    df_total.to_csv(MASTER_PATH, index=False, encoding="utf-8")
    return df_total


def load_master_dataset():
    if os.path.exists(MASTER_PATH):
        df = pd.read_csv(MASTER_PATH)
        expected = ["reviewId", "content", "at", "label_id", "label_text", "source"]
        for c in expected:
            if c not in df.columns:
                df[c] = None
        return df[expected]
    else:
        return pd.DataFrame(
            columns=["reviewId", "content", "at", "label_id", "label_text", "source"]
        )


# ========== LOAD METRICS ==========
@st.cache_data
def load_test_metrics():
    if os.path.exists(METRICS_PATH):
        return pd.read_csv(METRICS_PATH)
    return pd.DataFrame()


metrics_df = load_test_metrics()
acc_val = 0.0
f1_val = 0.0

if not metrics_df.empty:
    macro_row = metrics_df[metrics_df["label_id"] == "macro"]
    acc_row = metrics_df[metrics_df["label_id"] == "overall"]
    if not macro_row.empty:
        f1_val = macro_row.iloc[0]["f1_score"]
    if not acc_row.empty:
        acc_val = acc_row.iloc[0]["precision"]


# ========== SIDEBAR ==========
st.sidebar.image("ayam unhas.png", width=60)


st.sidebar.markdown("---")
st.sidebar.markdown("**Performa Model (Test Set)**")
st.sidebar.write(f"Accuracy: `{acc_val:.4f}`")
st.sidebar.write(f"Macro F1: `{f1_val:.4f}`")

st.sidebar.markdown("---")
st.sidebar.markdown("**Update Data Play Store**")

if st.sidebar.button("Update Dataset", key="update_btn_sidebar"):
    with st.spinner("Mengambil review baru dan memperbarui dataset..."):
        tokenizer, model, device = load_teacher_model()
        df_all, df_new = auto_update(limit_latest=300)

        if df_new.empty:
            st.sidebar.warning("Tidak ada review baru.")
        else:
            df_new_labeled = pseudo_label_batch(df_new, tokenizer, model, device)
            df_total = merge_to_master(df_new_labeled)
            st.sidebar.success(f"Berhasil update! Total data: {len(df_total)}")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Dashboard IndoBERT Sentiment · v1.2")


# =========================================================
#  MAIN PAGE: FULL LIVE DASHBOARD
# =========================================================

st.title("📡 Dashboard Sistem Monitoring Sentimen Publik – myBCA")


# --- load master dataset ---
df_total = load_master_dataset()
if len(df_total) > 0:
    df_total["at"] = pd.to_datetime(df_total["at"], errors="coerce")

# 1. FILTER BULAN
month_options = []
if len(df_total) > 0 and df_total["at"].notna().any():
    min_period = df_total["at"].dt.to_period("M").min()
    max_period = df_total["at"].dt.to_period("M").max()
    months_range = pd.period_range(min_period, max_period, freq="M")
    month_options = [str(m) for m in months_range]

st.markdown("#### 📅 Filter Periode")

if len(month_options) == 0:
    st.info(
        "Dataset kosong atau kolom tanggal (`at`) belum tersedia. Silakan Update Dataset di Sidebar."
    )
    start_period = end_period = None
else:
    start_idx = 0
    end_idx = len(month_options) - 1

    if "month_start_idx" in st.session_state:
        start_idx = st.session_state["month_start_idx"]
    if "month_end_idx" in st.session_state:
        end_idx = st.session_state["month_end_idx"]

    c_sel1, c_sel2 = st.columns([1, 1])
    with c_sel1:
        start_sel = st.selectbox(
            "Mulai bulan", month_options, index=start_idx, key="start_month_select"
        )
    with c_sel2:
        end_sel = st.selectbox(
            "Sampai bulan", month_options, index=end_idx, key="end_month_select"
        )

    st.session_state["month_start_idx"] = month_options.index(start_sel)
    st.session_state["month_end_idx"] = month_options.index(end_sel)

    start_period = pd.Period(start_sel, freq="M")
    end_period = pd.Period(end_sel, freq="M")

    if start_period > end_period:
        start_period, end_period = end_period, start_period

st.markdown("---")

# 2. GRAFIK TREN & KARTU METRIK
if (
    "start_period" in locals()
    and start_period is not None
    and "end_period" in locals()
    and end_period is not None
    and len(df_total) > 0
):
    df_month = df_total.dropna(subset=["at"]).copy()
    df_month["year_month"] = df_month["at"].dt.to_period("M")
    df_month = df_month[df_month["year_month"].between(start_period, end_period)]

    if not df_month.empty:
        monthly_counts = (
            df_month.groupby(["year_month", "label_text"]).size().unstack(fill_value=0)
        )
        for c in ["negatif", "netral", "positif"]:
            if c not in monthly_counts.columns:
                monthly_counts[c] = 0
        monthly_counts = monthly_counts[["negatif", "netral", "positif"]].sort_index()
        x_vals = monthly_counts.index.to_timestamp()

        # METRIC CARDS
        total_neg = int(monthly_counts["negatif"].sum())
        total_net = int(monthly_counts["netral"].sum())
        total_pos = int(monthly_counts["positif"].sum())

        st.markdown("#### 📈 Tren & Ringkasan")

        # Grafik Line Chart
        fig_line, ax_line = plt.subplots(figsize=(10, 3))
        ax_line.plot(
            x_vals,
            monthly_counts["negatif"],
            marker="o",
            color="#ef4444",
            label="negatif",
        )
        ax_line.plot(
            x_vals,
            monthly_counts["netral"],
            marker="o",
            color="#94a3b8",
            label="netral",
        )
        ax_line.plot(
            x_vals,
            monthly_counts["positif"],
            marker="o",
            color="#22c55e",
            label="positif",
        )
        ax_line.set_ylabel("Jumlah Ulasan")
        ax_line.grid(True, linestyle="--", alpha=0.3)
        ax_line.legend()
        plt.xticks(rotation=0)
        st.pyplot(fig_line, use_container_width=True)

        # Kartu Metrik di bawah grafik
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.markdown(
                f"""<div class="metric-card"><div class="metric-title">TOTAL NEGATIF</div><div class="metric-value" style="color:#ef4444">{total_neg}</div></div>""",
                unsafe_allow_html=True,
            )
        with mc2:
            st.markdown(
                f"""<div class="metric-card"><div class="metric-title">TOTAL NETRAL</div><div class="metric-value" style="color:#64748b">{total_net}</div></div>""",
                unsafe_allow_html=True,
            )
        with mc3:
            st.markdown(
                f"""<div class="metric-card"><div class="metric-title">TOTAL POSITIF</div><div class="metric-value" style="color:#22c55e">{total_pos}</div></div>""",
                unsafe_allow_html=True,
            )

    else:
        st.info("Tidak ada data pada rentang bulan yang dipilih.")
else:
    st.info("Menunggu data...")

st.markdown("---")

# 3. BAGIAN BAWAH: FILTER, BAR CHART, & TABEL (Layout Split)
if len(df_total) > 0:
    st.markdown("#### 🎯 Analisis Detail")

    label_filter = st.multiselect(
        "Pilih Label Sentimen:",
        options=["negatif", "netral", "positif"],
        default=["negatif", "netral", "positif"],
        key="label_filter_live",
    )

    # Filter Data berdasarkan Label
    df_view_live = df_total[df_total["label_text"].isin(label_filter)].copy()

    # Apply date filter lagi
    if (
        "start_period" in locals()
        and start_period is not None
        and "end_period" in locals()
        and end_period is not None
    ):
        df_view_live["year_month"] = df_view_live["at"].dt.to_period("M")
        df_view_live = df_view_live[
            df_view_live["year_month"].between(start_period, end_period)
        ]

    st.write(f"Menampilkan **{len(df_view_live)}** review (setelah filter).")

    # === LAYOUT DUA KOLOM (KEMBALI KE ASAL) ===
    col_g1, col_g2 = st.columns(2)

    # KIRI: Bar Chart Distribusi
    with col_g1:
        st.markdown("**Distribusi Sentimen (Filtered)**")

        label_order = [
            lab for lab in ["negatif", "netral", "positif"] if lab in label_filter
        ]

        if len(label_order) == 0:
            st.info("Pilih label dulu.")
        else:
            label_counts = (
                df_view_live["label_text"].value_counts().reindex(label_order).fillna(0)
            )

            fig_live, ax_live = plt.subplots(figsize=(4, 3))
            ax_live.bar(
                label_counts.index,
                label_counts.values,
                color=[
                    (
                        "#ef4444"
                        if x == "negatif"
                        else "#94a3b8" if x == "netral" else "#22c55e"
                    )
                    for x in label_counts.index
                ],
            )
            ax_live.set_xlabel("Label")
            ax_live.set_ylabel("Jumlah")
            plt.tight_layout()
            st.pyplot(fig_live, use_container_width=False)

    # KANAN: Tabel Data & Download
    with col_g2:
        st.markdown("**Review**")

        display_df = df_view_live.sort_values("at", ascending=False).reset_index(
            drop=True
        )
        display_df.index = display_df.index + 1

        st.dataframe(
            display_df[["at", "label_text", "content"]],
            use_container_width=True,
            height=300,
        )

        csv_live = df_view_live.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Download CSV Filtered",
            data=csv_live,
            file_name="filtered_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )
