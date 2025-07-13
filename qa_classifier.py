# qa_classifier.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict

class QAClassifier:
    """
    Memuat model klasifikasi teks dari Hugging Face Hub dan melakukan prediksi.
    """
    def __init__(self, model_id: str):
        """
        Menginisialisasi tokenizer dan model dari Hugging Face Hub.

        Args:
            model_id (str): Nama repositori di Hugging Face (contoh: "nama-anda/nama-model-anda").
        """
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Menggunakan perangkat: {self.device}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(self.device)
            
            # Ambil pemetaan label dari konfigurasi model (penting!)
            self.id2label: Dict[int, str] = self.model.config.id2label
            
            print(f"Model klasifikasi '{model_id}' berhasil dimuat.")
            print(f"Label yang terdeteksi: {self.id2label}")

        except Exception as e:
            raise RuntimeError(f"Gagal memuat model dari Hugging Face: {e}")

    def predict(self, query: str) -> str:
        """
        Memprediksi kelas dari sebuah query.

        Args:
            query (str): Teks pertanyaan dari pengguna.

        Returns:
            str: Label kelas yang diprediksi (contoh: "Kompleks" atau "Simple").
        """
        try:
            # 1. Tokenisasi input
            inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True).to(self.device)
            
            # 2. Lakukan inferensi tanpa menghitung gradien untuk efisiensi
            with torch.no_grad():
                logits = self.model(**inputs).logits
            
            # 3. Dapatkan ID kelas dengan probabilitas tertinggi
            predicted_class_id = torch.argmax(logits, dim=-1).item()
            
            # 4. Konversi ID ke label string menggunakan pemetaan dari config
            result = self.id2label[predicted_class_id]
            
            print(f"Query: '{query}' -> Klasifikasi: {result}")
            return result
            
        except Exception as e:
            print(f"Error saat klasifikasi query: {e}")
            # Default ke 'Simple' jika terjadi error
            return "Simple"