import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

class DataSanitizer:
    """
    Pipeline di sanificazione end-to-end. Rimuove il rumore indotto dall'hardware,
    previene l'overfitting e fa emergere il trend reale per il modello PyTorch.
    """
    def __init__(self, z_threshold=3.0, smooth_window=11, smooth_poly=2):
        self.z_threshold = z_threshold
        self.smooth_window = smooth_window
        self.smooth_poly = smooth_poly

    def remove_anomalies(self, df, columns):
        """Usa lo Z-score per intercettare e rimuovere i picchi impossibili (glitches)."""
        df_clean = df.copy()
        for col in columns:
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            
            # Calcola lo Z-score, ignorando i NaN
            z_scores = np.abs((df_clean[col] - mean) / std)
            
            # Sostituisce gli outlier estremi con NaN per poi interpolarli
            is_outlier = z_scores > self.z_threshold
            df_clean.loc[is_outlier, col] = np.nan
        return df_clean

    def impute_missing(self, df, columns, method='linear'):
        """Riempie i buchi (disconnessioni) tramite interpolazione."""
        df_clean = df.copy()
        for col in columns:
            # L'interpolazione lineare unisce il punto prima e dopo la disconnessione
            df_clean[col] = df_clean[col].interpolate(method=method, limit_direction='both')
            # Fallback se le prime righe sono NaN
            df_clean[col] = df_clean[col].bfill().ffill() 
        return df_clean

    def apply_smoothing(self, df, columns):
        """Applica il filtro Savitzky-Golay per eliminare il micro-rumore e i 'freezes'."""
        df_clean = df.copy()
        for col in columns:
            df_clean[col] = savgol_filter(
                df_clean[col], 
                window_length=self.smooth_window, 
                polyorder=self.smooth_poly
            )
        return df_clean

    def run_pipeline(self, df, columns):
        """Esegue l'intera tecnica di sanitizzazione in ordine logico."""
        # 1. Rimuove i picchi assurdi (li trasforma in NaN)
        df_step1 = self.remove_anomalies(df, columns)
        
        # 2. Riempie i NaN (sia quelli originali che quelli creati dallo step 1)
        df_step2 = self.impute_missing(df_step1, columns)
        
        # 3. Ammorbidisce la curva per aiutare la LSTM a trovare i trend
        df_final = self.apply_smoothing(df_step2, columns)
        
        return df_final