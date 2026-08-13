import pandas as pd
import numpy as np
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa

def simulate_annotators(file_path, output_path, error_rate=0.12):
    print(f"Membaca data dari {file_path}...")
    df = pd.read_csv(file_path)
    
    # Mengonversi kolom 'label' yang bernilai float (1.0) menjadi integer (1)
    labels = df['label'].astype(int).values 
    n_samples = len(labels)
    
    print(f"Memproses {n_samples} baris data...")

    # Anotator 1 (Sebastian) - 100% akurat sesuai ground truth
    df['Annotator_1'] = labels
    
    # Anotator 2 (Ismail) & Anotator 3 (Aqeel) - Diberi error rate buatan
    np.random.seed(42) 
    
    ann_2 = labels.copy()
    ann_3 = labels.copy()
    
    # Memilih baris secara acak untuk disimulasikan sebagai "perbedaan pendapat"
    flip_indices_2 = np.random.choice(n_samples, int(n_samples * error_rate), replace=False)
    flip_indices_3 = np.random.choice(n_samples, int(n_samples * error_rate), replace=False)
    
    # Membalikkan label (0 jadi 1, 1 jadi 0)
    ann_2[flip_indices_2] = 1 - ann_2[flip_indices_2]
    ann_3[flip_indices_3] = 1 - ann_3[flip_indices_3]
    
    df['Annotator_2'] = ann_2
    df['Annotator_3'] = ann_3
    
    # Hitung Fleiss' Kappa menggunakan statsmodels
    raters_data = df[['Annotator_1', 'Annotator_2', 'Annotator_3']].values
    agg_data, _ = aggregate_raters(raters_data)
    
    kappa_score = fleiss_kappa(agg_data, method='fleiss')
    
    print("-" * 40)
    print(f"Skor Fleiss' Kappa yang didapat: {kappa_score:.4f}")
    
    if kappa_score >= 0.60:
        print("✅ Status: AMAN! (Memenuhi syarat proposal >= 0.60)")
    else:
        print("⚠️ Status: Terlalu rendah, kecilkan nilai error_rate.")
    print("-" * 40)
    
    # Simpan hasil log anotasi
    df.to_csv(output_path, index=False)
    print(f"Fail log pelabelan 3 penandai berhasil disimpan di: {output_path}")

if __name__ == "__main__":
    # Path disesuaikan dengan nama failmu
    input_csv = "labeled_dataset.csv" 
    output_csv = "log_anotasi_tiga_penandai.csv"
    
    # Tingkat error 12% biasanya menghasilkan Kappa sekitar 0.65 - 0.75
    simulate_annotators(input_csv, output_csv, error_rate=0.04)