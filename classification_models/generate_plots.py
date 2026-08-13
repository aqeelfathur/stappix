import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Pengaturan font agar terlihat profesional seperti di jurnal
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10})

def plot_confusion_matrix():
    # Data dari log model IndoBERTweet milikmu
    # [True Negative (213), False Positive (72)]
    # [False Negative (8),  True Positive (16)]
    cm = np.array([[213, 72], 
                   [8, 16]])
    
    plt.figure(figsize=(6, 5))
    
    # Menggunakan colormap 'Blues' agar jika dicetak hitam-putih tetap terlihat kontras gradasinya
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                     xticklabels=['Non-Hoaks (0)', 'Hoaks (1)'], 
                     yticklabels=['Non-Hoaks (0)', 'Hoaks (1)'],
                     cbar=False, annot_kws={"size": 14, "weight": "bold"})
    
    plt.title('Confusion Matrix IndoBERTweet (Threshold 0.20)', pad=15, fontweight='bold')
    plt.ylabel('Label Aktual (Ground Truth)', fontweight='bold')
    plt.xlabel('Label Prediksi Model', fontweight='bold')
    
    # Simpan dengan resolusi tinggi (300 ppi) untuk makalah
    plt.tight_layout()
    plt.savefig('gambar_2_confusion_matrix.png', dpi=300)
    print("✅ Gambar Confusion Matrix berhasil disimpan: 'gambar_2_confusion_matrix.png'")
    plt.close()

def plot_threshold_curve():
    # Data dari hasil Sweep Table tuning milikmu
    thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    recall_hoax = [0.7391, 0.7391, 0.6522, 0.6087, 0.5652, 0.4783, 0.4783, 0.4783, 0.4348]
    precision_hoax = [0.1382, 0.1848, 0.1899, 0.2373, 0.2600, 0.2558, 0.2683, 0.2821, 0.2778]
    macro_f1 = [0.4978, 0.5673, 0.5773, 0.6204, 0.6350, 0.6268, 0.6340, 0.6414, 0.6346]

    plt.figure(figsize=(8, 5))
    
    # Menggunakan marker dan linestyle yang berbeda agar jelas saat dicetak B/W
    plt.plot(thresholds, recall_hoax, label='Recall (Hoaks)', marker='o', linestyle='-', color='#1f77b4', linewidth=2)
    plt.plot(thresholds, precision_hoax, label='Precision (Hoaks)', marker='s', linestyle='--', color='#ff7f0e', linewidth=2)
    plt.plot(thresholds, macro_f1, label='Macro F1-Score', marker='^', linestyle=':', color='#2ca02c', linewidth=2)
    
    # Menandai titik Threshold terpilih (0.20)
    plt.axvline(x=0.20, color='red', linestyle='-.', alpha=0.7, label='Chosen Threshold (0.20)')
    
    # Menambahkan anotasi teks di titik temu 0.20 untuk Recall
    plt.text(0.205, 0.66, 'Rec: 0.65', color='#1f77b4', fontweight='bold')
    
    plt.title('Kurva Threshold Tuning vs Performa Model', pad=15, fontweight='bold')
    plt.xlabel('Batas Probabilitas (Threshold)', fontweight='bold')
    plt.ylabel('Nilai Metrik', fontweight='bold')
    plt.xticks(thresholds)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='center right')
    
    # Simpan dengan resolusi tinggi (300 ppi) untuk makalah
    plt.tight_layout()
    plt.savefig('gambar_3_threshold_curve.png', dpi=300)
    print("✅ Gambar Kurva Threshold berhasil disimpan: 'gambar_3_threshold_curve.png'")
    plt.close()

if __name__ == '__main__':
    print("Membuat visualisasi untuk makalah GEMASTIK...")
    plot_confusion_matrix()
    plot_threshold_curve()
    print("Selesai!")