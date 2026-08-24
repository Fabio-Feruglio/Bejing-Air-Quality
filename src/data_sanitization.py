import pandas as pd
import numpy as np

class DataSanitizer:
    """
    Pipeline di sanificazione end-to-end, causale: il valore pulito al tempo t
    dipende solo da informazioni fino a t incluso, mai dal futuro. Segue il
    pattern fit/transform di scikit-learn: fit() stima le statistiche SOLO
    sul training set, transform() le riusa su qualunque altra partizione.
    """
    def __init__(self, z_threshold=3.0, smooth_span=6):
        self.z_threshold = z_threshold
        self.smooth_span = smooth_span
        self.stats_ = {}

    def fit(self, df, columns):
        self.stats_ = {col: (df[col].mean(), df[col].std()) for col in columns}
        return self

    def transform(self, df, columns):
        if not self.stats_:
            raise RuntimeError("Chiama fit(df_train, columns) prima di transform().")
        df_clean = df.copy()
        for col in columns:
            mean, std = self.stats_[col]

            # Outlier hardware (glitch) -> NaN, con statistiche del training
            if std and not pd.isna(std):
                z_scores = (df_clean[col] - mean).abs() / std
                df_clean.loc[z_scores > self.z_threshold, col] = np.nan

            # Imputazione CAUSALE dei buchi: solo forward-fill, mai guarda al
            # futuro. Le righe che restano NaN (nessun valore precedente
            # disponibile, tipicamente solo l'inizio della serie) vengono
            # scartate a valle da dropna() in process_split, invece di essere
            # "indovinate" con dati non ancora osservati.
            df_clean[col] = df_clean[col].ffill()

            df_clean[col] = df_clean[col].ewm(span=self.smooth_span, adjust=False).mean()

        return df_clean

    def fit_transform(self, df, columns):
        return self.fit(df, columns).transform(df, columns)
