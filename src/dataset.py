import pandas as pd
import numpy as np

class SensorErrorInjector:
    """
    Simula guasti hardware e problemi di rete sui dati grezzi dei sensori.
    Questo modulo e' essenziale per ricreare le condizioni di un "highly-noisy raw dataset".
    """
    def __init__(self, random_seed=42):
        self.seed = random_seed
        np.random.seed(self.seed)

    def inject_disconnections(self, df, column, n_blocks=50, max_block_size=24):
        """Simula disconnessioni di rete inserendo blocchi di NaN (dati mancanti)."""
        df_noisy = df.copy()
        col_pos = df_noisy.columns.get_loc(column)
        for _ in range(n_blocks):
            start_idx = np.random.randint(0, len(df_noisy) - max_block_size)
            block_size = np.random.randint(5, max_block_size)
            end_idx = min(start_idx + block_size, len(df_noisy))
            df_noisy.iloc[start_idx:end_idx, col_pos] = np.nan
        return df_noisy

    def inject_glitches(self, df, column, n_glitches=50):
        """Simula cortocircuiti o cali di batteria inserendo picchi anomali (es. PM2.5 a 9999)."""
        df_noisy = df.copy()
        col_pos = df_noisy.columns.get_loc(column)
        max_val = df_noisy[column].max(skipna=True)
        for _ in range(n_glitches):
            spike_idx = np.random.randint(0, len(df_noisy))
            # Inserisce un valore fisicamente impossibile
            df_noisy.iloc[spike_idx, col_pos] = max_val * np.random.uniform(5, 10)
        return df_noisy

    def inject_freezes(self, df, column, n_freezes=30, freeze_length=12):
        """Simula un sensore 'congelato' che ripete lo stesso identico valore per ore."""
        df_noisy = df.copy()
        col_pos = df_noisy.columns.get_loc(column)
        for _ in range(n_freezes):
            start_idx = np.random.randint(0, len(df_noisy) - freeze_length)
            end_idx = min(start_idx + freeze_length, len(df_noisy))
            frozen_value = df_noisy.iloc[start_idx, col_pos]
            df_noisy.iloc[start_idx:end_idx, col_pos] = frozen_value
        return df_noisy

    def corrupt_dataset(self, df, columns):
        """Applica tutti gli errori hardware alle colonne specificate.

        Nota: l'indice viene resettato prima di iniettare rumore. Le tre
        funzioni sopra generano posizioni casuali con np.random.randint,
        pensate come *posizioni* (0..len-1) e ora usate con .iloc, che e'
        posizionale. Se df arriva da un filtro (es. df[df['station']==x])
        il suo indice originale puo' non partire da 0 e non essere
        contiguo: senza reset_index, .loc (usato in precedenza) avrebbe
        potuto non trovare le righe giuste, iniettando rumore nel posto
        sbagliato o, silenziosamente, da nessuna parte.
        """
        df_corrupted = df.reset_index(drop=True).copy()
        for col in columns:
            df_corrupted = self.inject_disconnections(df_corrupted, col)
            df_corrupted = self.inject_glitches(df_corrupted, col)
            df_corrupted = self.inject_freezes(df_corrupted, col)
        return df_corrupted

# Funzione helper per caricare il dataset
def load_and_corrupt_data(filepath, target_columns):
    # Nota: il tuo file 'prova.xlsx' ha i dati uniti in una singola colonna stile CSV.
    # Se il dataset reale di 420k righe e' un CSV standard, usa pd.read_csv(filepath)
    df = pd.read_csv(filepath)
    injector = SensorErrorInjector()
    df_noisy = injector.corrupt_dataset(df, target_columns)
    return df, df_noisy