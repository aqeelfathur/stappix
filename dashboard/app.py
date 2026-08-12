import os
import sys

# --- ANTI-DEADLOCK MACOS (Wajib di baris paling atas) ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import pandas as pd
import shap
from streamlit_shap import st_shap
import torch

# Batasi penggunaan thread untuk mencegah stuck di Mac
torch.set_num_threads(1)

# Injeksi path agar Streamlit bisa membaca folder 'evaluation_models'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluation_models.shap_explainer import TriageExplainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Konfigurasi Halaman Dasar
st.set_page_config(page_title="STAPPIX | Risk Triage", page_icon="🚨", layout="wide")

# 2. Fungsi Memuat Data (Di-cache agar ringan)
@st.cache_data
def load_data():
    file_path = "../evaluation_models/output/scored_test_results.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df = df.sort_values(by="Risk_Score", ascending=False).reset_index(drop=True)
        return df
    return None

# 3. Fungsi Memuat Model AI & SHAP (Di-cache agar hanya dimuat 1x saat aplikasi jalan)
@st.cache_resource
def load_ai_explainer():
    model_path = "../classification_models/best_model_checkpoint"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return TriageExplainer(model, tokenizer)

def main():
    st.title("🚨 STAPPIX: Risk Triage Dashboard")
    st.markdown("Sistem Pendukung Keputusan untuk Memprioritaskan Mitigasi Risiko Hoaks pada Kebijakan Publik")
    st.divider()

    df = load_data()

    if df is not None:
        # Ringkasan Metrik
        col1, col2, col3, col4 = st.columns(4)
        count_merah = len(df[df['Zona_Triase'] == 'Merah'])
        count_kuning = len(df[df['Zona_Triase'] == 'Kuning'])
        count_hijau = len(df[df['Zona_Triase'] == 'Hijau'])

        with col1:
            st.metric(label="Total Antrean Triage", value=len(df))
        with col2:
            st.metric(label="🔴 Zona Merah (Immediate)", value=count_merah)
        with col3:
            st.metric(label="🟡 Zona Kuning (Priority)", value=count_kuning)
        with col4:
            st.metric(label="🟢 Zona Hijau (Routine)", value=count_hijau)

        st.write("---")
        st.subheader("Daftar Antrean Penanganan Konten")
        
        # Filter Zona
        pilihan_zona = st.selectbox("Filter berdasarkan Zona Triase:", ["Semua Zona", "Merah", "Kuning", "Hijau"])
        df_tampil = df[df['Zona_Triase'] == pilihan_zona] if pilihan_zona != "Semua Zona" else df

        kolom_tampil = ['text', 'P_hoax', 'Amplification_norm', 'Velocity_norm', 'Risk_Score', 'Zona_Triase', 'Target_Respons']
        st.dataframe(df_tampil[kolom_tampil], use_container_width=True, height=400, hide_index=True)

        st.write("---")
        st.subheader("🔍 Analisis Mendalam (Explainable AI)")
        
        # Pilihan Teks untuk Dianalisis
        pilihan_teks = st.selectbox(
            "Pilih cuitan dari tabel di atas untuk membedah kontribusi kata pembentuk hoaks:",
            df_tampil['text'].tolist()
        )
        
        if pilihan_teks:
            baris_terpilih = df[df['text'] == pilihan_teks].iloc[0]
            
            st.markdown("**Detail Kalkulasi Risiko Eksak:**")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Probabilitas Teks (IndoBERT)", f"{baris_terpilih['P_hoax']*100:.2f}%")
            col_b.metric("Bobot Amplifikasi Aktual", f"{baris_terpilih['Amplification_norm']:.2f}")
            col_c.metric("Bobot Velositas Aktual", f"{baris_terpilih['Velocity_norm']:.2f}")
            
            st.markdown("**Visualisasi SHAP (Token-Level Explanation):**")
            
            # Tombol Eksekusi On-Demand
            if st.button("Jalankan Mesin Interpretasi SHAP", type="primary"):
                with st.spinner("Memuat AI dan menghitung kontribusi linguistik... (Ini memakan waktu beberapa detik)"):
                    try:
                        explainer_engine = load_ai_explainer()
                        shap_values = explainer_engine.explain_text_token_level(pilihan_teks)
                        
                        # Merender visualisasi teks dari SHAP ke Streamlit
                        st_shap(shap.plots.text(shap_values))
                        st.success("Komputasi selesai! Warna merah menandakan kata yang mendorong AI memprediksi teks sebagai hoaks.")
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat komputasi AI: {e}")
    else:
        st.error("File 'scored_test_results.csv' tidak ditemukan. Pastikan sudah ada di folder 'evaluation_models/output/'.")

if __name__ == "__main__":
    main()