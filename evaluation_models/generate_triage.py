import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from risk_scoring_engine import RiskScoringEngine
import os

def load_model_and_tokenizer(model_path):
    print("Memuat model IndoBERTweet dan Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval() # Set ke mode evaluasi
    return tokenizer, model

def predict_hoax_prob(texts, tokenizer, model):
    print(f"Melakukan inferensi untuk {len(texts)} data...")
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Mengubah nilai logit menjadi probabilitas (0-1) menggunakan Softmax
        probs = F.softmax(outputs.logits, dim=-1)
        
    # Asumsi: Indeks 1 adalah kelas 'Hoaks' (sesuaikan jika indeksnya berbeda)
    hoax_probs = probs[:, 1].tolist()
    return hoax_probs

def main():
    # 1. Path Konfigurasi (Sesuaikan dengan struktur folder aslimu)
    model_path = "classification_models/best_model_checkpoint"
    test_data_path = "data/processed/test.csv" # Pastikan file test ini ada
    output_path = "output/scored_test_results.csv"
    
    # Buat folder output jika belum ada
    os.makedirs("output", exist_ok=True)
    
    # 2. Muat Data Testing Set
    print(f"Membaca data testing dari: {test_data_path}")
    # Jika kamu pakai format lain (misal json atau .pt, ini perlu disesuaikan)
    try:
        df_test = pd.read_csv(test_data_path)
    except FileNotFoundError:
        print(f"Error: File {test_data_path} tidak ditemukan. Silakan cek path-nya.")
        return

    # 3. Prediksi Probabilitas dengan AI
    tokenizer, model = load_model_and_tokenizer(model_path)
    # Mengambil teks dari kolom 'text' (Ubah 'text' jika nama kolommu di CSV berbeda)
    texts = df_test['text'].tolist() 
    df_test['P_hoax'] = predict_hoax_prob(texts, tokenizer, model)
    
    # 4. Masukkan ke Risk Scoring Engine
    print("Menghitung Risk Score dan Zona Triase...")
    engine = RiskScoringEngine(w_hoax=0.5, w_amp=0.25, w_vel=0.25)
    
    # Pastikan nama kolom metrik sesuai dengan yang ada di CSV kamu
    df_final = engine.process_data(
        df_test, 
        p_hoax_col='P_hoax', 
        amp_col='amplification_norm', # Sesuaikan jika nama kolom berbeda
        vel_col='velocity_norm'       # Sesuaikan jika nama kolom berbeda
    )
    
    # 5. Simpan Hasil
    df_final.to_csv(output_path, index=False)
    print(f"Selesai! Hasil Triase berhasil disimpan di: {output_path}")

if __name__ == "__main__":
    main()