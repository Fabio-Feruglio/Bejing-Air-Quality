import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

class DataSanitizer:
    """
    Pipeline di sanificazione end-to-end. Rimuove il rumore indotto dall'hardware,
    previene l'overfitting e fa emergere il trend reale per il modello PyTorch.

    Segue il pattern fit/transform di scikit-learn: fit() stima le statistiche
    (media, deviazione standard per lo Z-score) SOLO sui dati che gli passi
    (in pratica: solo il training set); transform() riusa quelle statistiche
    su qualsiasi altra partizione (val, test), senza mai ricalcolarle su di
    essa. Cosi' la sanificazione di una riga di validazione non dipende da
    nessuna informazione vista solo in validazione o test.
    """
    def __init__(self, z_threshold=3.0, smooth_window=11, smooth_poly=2):
        self.z_threshold = z_threshold
        self.smooth_window = smooth_window
        self.smooth_poly = smooth_poly
        self.stats_ = {}

    def fit(self, df, columns):
        """Stima media e deviazione standard per colonna, da riusare in transform()."""
        self.stats_ = {col: (df[col].mean(), df[col].std()) for col in columns}
        return self

    def remove_anomalies(self, df, columns):
        """Usa lo Z-score (con le statistiche stimate in fit) per intercettare
        e rimuovere i picchi impossibili (glitches)."""
        if not self.stats_:
            raise RuntimeError("Chiama fit(df_train, columns) prima di transform().")
        df_clean = df.copy()
        for col in columns:
            mean, std = self.stats_[col]
            if not std or pd.isna(std):
                continue

            # Calcola lo Z-score usando media/std del training set, ignorando i NaN
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
        """Applica il filtro Savitzky-Golay per eliminare il micro-rumore e i 'freezes'.

        Nota: e' un filtro NON causale (guarda anche punti "futuri" dentro la
        finestra), quindi va sempre applicato SEPARATAMENTE su train/val/test
        dopo lo split -- mai su tutta la serie combinata, altrimenti lo
        smoothing di righe di training userebbe implicitamente dati di
        validazione/test e viceversa.
        """
        df_clean = df.copy()
        window = self.smooth_window
        if window >= len(df_clean):
            # Partizione troppo piccola per la finestra configurata: usa la
            # finestra dispari piu' grande possibile, invece di alzare un errore.
            window = len(df_clean) - 1 if (len(df_clean) - 1) % 2 == 1 else len(df_clean) - 2
        if window <= self.smooth_poly:
            return df_clean
        for col in columns:
            df_clean[col] = savgol_filter(
                df_clean[col],
                window_length=window,
                polyorder=self.smooth_poly
            )
        return df_clean

    def transform(self, df, columns):
        """Esegue l'intera pipeline di sanificazione usando le statistiche di fit()."""
        # 1. Rimuove i picchi assurdi (li trasforma in NaN)
        df_step1 = self.remove_anomalies(df, columns)

        # 2. Riempie i NaN (sia quelli originali che quelli creati dallo step 1)
        df_step2 = self.impute_missing(df_step1, columns)

        # 3. Ammorbidisce la curva per aiutare la LSTM a trovare i trend
        df_final = self.apply_smoothing(df_step2, columns)

        return df_final

    def fit_transform(self, df, columns):
        """Comodita': fit() + transform() sullo stesso dataframe (solo per il training set)."""
        return self.fit(df, columns).transform(df, columns)