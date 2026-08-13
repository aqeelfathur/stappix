import shap
from transformers import pipeline

class TriageExplainer:
    def __init__(self, model, tokenizer):
        """
        Inisialisasi SHAP Explainer menggunakan pipeline HuggingFace.
        """
        self.model = model
        self.tokenizer = tokenizer
        
        self.pipe = pipeline(
            "text-classification", 
            model=self.model, 
            tokenizer=self.tokenizer, 
            return_all_scores=True, 
            device="cpu" 
        )
        
        # Partition explainer sangat optimal untuk model teks Transformer
        self.explainer = shap.Explainer(self.pipe)

    def explain_text_token_level(self, text):
        """
        Mengekstrak nilai SHAP per kata untuk satu cuitan spesifik.
        """
        shap_values = self.explainer([text])
        # Mengembalikan indeks [0] karena kita hanya memasukkan 1 teks
        return shap_values[0]