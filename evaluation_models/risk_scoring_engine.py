import pandas as pd

class RiskScoringEngine:
    def __init__(self, w_hoax=0.5, w_amp=0.25, w_vel=0.25):
        """
        Inisialisasi bobot heuristik sesuai proposal TRIAXIS/STAPPIX.
        Total bobot harus sama dengan 1.0
        """
        self.w_hoax = w_hoax
        self.w_amp = w_amp
        self.w_vel = w_vel

    def calculate_score(self, p_hoax, amp_norm, vel_norm):
        """
        Mengonversi P_hoax dan metrik interaksi menjadi Risk Score (0-100).
        """
        risk_score = 100 * ((self.w_hoax * p_hoax) + 
                            (self.w_amp * amp_norm) + 
                            (self.w_vel * vel_norm))
        return round(risk_score, 2)

    def determine_zone(self, risk_score):
        """
        Memetakan Risk Score ke dalam 3 Zona Triase.
        """
        if risk_score >= 70:
            return "Merah", "Immediate (< 1 jam)"
        elif risk_score >= 40:
            return "Kuning", "Priority (< 6 jam)"
        else:
            return "Hijau", "Routine/Monitoring (< 24 jam)"

    def process_data(self, df, p_hoax_col='P_hoax', amp_col='Amplification_norm', vel_col='Velocity_norm'):
        """
        Mengeksekusi perhitungan untuk seluruh baris di dalam DataFrame testing set.
        """
        # Hitung skor risiko
        df['Risk_Score'] = df.apply(
            lambda row: self.calculate_score(row[p_hoax_col], row[amp_col], row[vel_col]), 
            axis=1
        )
        
        # Tentukan zona dan target respons
        zone_info = df['Risk_Score'].apply(self.determine_zone)
        df['Zona_Triase'] = [info[0] for info in zone_info]
        df['Target_Respons'] = [info[1] for info in zone_info]
        
        # Hitung kontribusi eksak (untuk interpretasi Dashboard/SHAP Feature-level)
        df['Kontribusi_Teks'] = df[p_hoax_col] * self.w_hoax * 100
        df['Kontribusi_Amplifikasi'] = df[amp_col] * self.w_amp * 100
        df['Kontribusi_Velositas'] = df[vel_col] * self.w_vel * 100
        
        return df

# Blok pengujian lokal (opsional, hanya jalan jika script di-run langsung)
if __name__ == "__main__":
    print("Mencoba Risk Scoring Engine dengan data dummy...")
    engine = RiskScoringEngine()
    
    # Contoh kasus: Teks diyakini hoaks 80%, viralitas moderat
    skor = engine.calculate_score(p_hoax=0.80, amp_norm=0.50, vel_norm=0.40)
    zona, respons = engine.determine_zone(skor)
    
    print(f"Risk Score: {skor}")
    print(f"Zona Triase: {zona} (Target: {respons})")