import shap
from transformers import pipeline

class TriageExplainer:
    def __init__(self, model, tokenizer):
        """
        Inisialisasi SHAP Explainer menggunakan pipeline HuggingFace.
        """
        self.model = model
        self.tokenizer = tokenizer
        
        # Menggunakan pipeline agar otomatis menangani tokenisasi dan output Softmax
        # return_all_scores=True penting agar SHAP tahu probabilitas semua kelas
        self.pipe = pipeline(
            "text-classification", 
            model=self.model, 
            tokenizer=self.tokenizer, 
            return_all_scores=True, 
            device="cpu" # Ubah ke "cuda" atau "mps" jika menggunakan GPU/Apple Silicon
        )
        
        # Partition explainer sangat optimal untuk model teks/Transformer
        self.explainer = shap.Explainer(self.pipe)

    def explain_text_token_level(self, text):
        """
        Mengekstrak nilai SHAP per kata (token) untuk satu cuitan spesifik.
        Ini dipanggil secara on-demand dari Streamlit.
        """
        # SHAP memproses teks dalam bentuk list
        shap_values = self.explainer([text])
        return shap_values

# Blok pengujian lokal
if __name__ == "__main__":
    print("Modul SHAP Explainer siap diintegrasikan ke Streamlit!")
    # Catatan: Pengujian langsung di sini membutuhkan model yang dimuat (load), 
    # lebih ideal langsung diuji di dalam skrip Streamlit nanti.