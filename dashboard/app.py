import os
import sys
import re # Tambahan untuk membersihkan token mesin

# --- ULTIMATE ANTI-DEADLOCK MACOS ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES" 
os.environ["USE_TF"] = "0" 
os.environ["USE_TORCH"] = "1" 

import streamlit as st
import pandas as pd
import altair as alt

# 1. Konfigurasi Halaman Dasar
st.set_page_config(page_title="STAPPIX | Risk Triage", page_icon="🚨", layout="wide")

# 2. Fungsi Memuat Data
@st.cache_data
def load_data():
    file_path = "../evaluation_models/output/scored_test_results.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df = df.sort_values(by="Risk_Score", ascending=False).reset_index(drop=True)
        return df
    return None

# 3. Fungsi Memuat Model AI (Lazy Loading)
@st.cache_resource
def load_ai_explainer():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from evaluation_models.shap_explainer import TriageExplainer
    
    torch.set_num_threads(1)
    
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
            st.metric(label="Total Antrean", value=len(df))
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

        kolom_tampil = ['text', 'P_hoax', 'amplification_norm', 'velocity_norm', 'Risk_Score', 'Zona_Triase', 'Target_Respons']
        st.dataframe(df_tampil[kolom_tampil], use_container_width=True, height=350, hide_index=True)

        st.write("---")
        st.subheader("🔍 Analisis Mendalam (Explainable AI)")
        
        # Pilihan Teks untuk Dianalisis
        pilihan_teks = st.selectbox(
            "Pilih cuitan dari tabel di atas untuk membedah kontribusi kata:",
            df_tampil['text'].tolist()
        )
        
        if pilihan_teks:
            baris_terpilih = df[df['text'] == pilihan_teks].iloc[0]
            
            # UX FIX 1: Tampilkan teks asli agar analis ingat konteksnya
            st.info(f"**Konteks Cuitan:**\n\n_{baris_terpilih['text']}_")
            
            st.markdown("**Detail Kalkulasi Risiko Eksak:**")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Probabilitas Teks (IndoBERT)", f"{baris_terpilih['P_hoax']*100:.2f}%")
            col_b.metric("Bobot Amplifikasi Aktual", f"{baris_terpilih['amplification_norm']:.2f}")
            col_c.metric("Bobot Velositas Aktual", f"{baris_terpilih['velocity_norm']:.2f}")
            
            st.markdown("**Visualisasi SHAP (Interpretasi Semantik):**")
            
            if st.button("Jalankan Mesin Interpretasi SHAP", type="primary"):
                with st.spinner("Memuat AI dan membedah struktur linguistik..."):
                    try:
                        import shap
                        from streamlit_shap import st_shap
                        
                        explainer_engine = load_ai_explainer()
                        shap_values = explainer_engine.explain_text_token_level(pilihan_teks)
                        
                        tokens = shap_values.data
                        if len(shap_values.values.shape) == 2:
                            val = shap_values.values[:, 1]
                        else:
                            val = shap_values.values
                            
                        # UX FIX 2: Pembersihan Token Mesin (Humanizing Machine Output)
                        cleaned_pairs = []
                        for t, v in zip(tokens, val):
                            w = str(t).strip()
                            # 1. Buang token spesial bawaan arsitektur Transformer
                            if w in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']:
                                continue
                            # 2. Hapus imbuhan pemisah sub-kata (##) dari tokenizer
                            w = w.replace('##', '')
                            # 3. Abaikan token yang hanya berisi tanda baca murni
                            if not re.search('[a-zA-Z0-9]', w):
                                continue
                            
                            cleaned_pairs.append({"Kata": w.lower(), "Beban": float(v)})
                            
                        df_shap = pd.DataFrame(cleaned_pairs)
                        
                        if not df_shap.empty:
                            # Gabungkan kata yang sama (misal ada 2 kata 'rakyat' di kalimat)
                            df_shap = df_shap.groupby("Kata", as_index=False).sum()
                            df_shap = df_shap[df_shap["Beban"] > 0]
                            # Ambil 5 teratas
                            df_top = df_shap.sort_values(by="Beban", ascending=False).head(5)
                            
                            if not df_top.empty:
                                st.markdown("#### 🚩 Top 5 Kata Dominan Pemicu Hoaks")
                                chart = alt.Chart(df_top).mark_bar(color='#ff4b4b', cornerRadiusEnd=4).encode(
                                    x=alt.X('Beban:Q', title='Besaran Pengaruh (%)', axis=alt.Axis(format='%')),
                                    y=alt.Y('Kata:N', sort='-x', title=''),
                                    tooltip=['Kata', alt.Tooltip('Beban:Q', format='.2%')]
                                ).properties(height=280)
                                st.altair_chart(chart, use_container_width=True)
                            else:
                                st.info("AI mendeteksi sentimen secara keseluruhan, tidak ada kata pemicu tunggal yang menonjol.")
                        else:
                            st.info("Tidak dapat mengekstrak kata bermakna dari teks ini.")
                        
                        with st.expander("Lihat Visualisasi Lanjutan (SHAP Force Plot)"):
                            st.caption("Mode peneliti: Melihat kontribusi token sebelum dibersihkan (merah = pendorong hoaks, biru = penahan/fakta).")
                            st_shap(shap.plots.text(shap_values))
                            
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat komputasi AI: {e}")
    else:
        st.error("File 'scored_test_results.csv' tidak ditemukan. Pastikan sudah ada di folder 'evaluation_models/output/'.")

if __name__ == "__main__":
    main()