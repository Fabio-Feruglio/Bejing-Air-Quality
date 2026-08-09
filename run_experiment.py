import argparse
import numpy as np
import torch
import wandb
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

# Import dai tuoi moduli
from src.dataset import SensorErrorInjector
from src.data_sanitization import DataSanitizer
from src.model import AirQualityLSTM
from src.train import train_model
from src.evaluate import persistence_baseline_mse, report_original_scale_metrics


def create_sequences(data, target_col_idx, seq_length=24):
    """Trasforma un array 2D in sequenze 3D (Batch, Sequence, Features)"""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length, target_col_idx]
        xs.append(x)
        ys.append(y)
    return torch.tensor(np.array(xs), dtype=torch.float32), torch.tensor(np.array(ys), dtype=torch.float32).unsqueeze(1)


def process_split(df_raw, features, injector, sanitizer, scaler, fit=False):
    """
    Applica iniezione rumore + sanificazione + scaling a UNA partizione
    (train, val o test).

    Se fit=True (va usato solo per il training set) stima le statistiche
    di sanitizer e scaler su questa partizione; altrimenti (val/test)
    riusa quelle gia' stimate sul training, cosi' nessuna informazione
    su val/test influenza il preprocessing del training (e viceversa).
    """
    df_noisy = injector.corrupt_dataset(df_raw, features)

    if fit:
        df_clean = sanitizer.fit_transform(df_noisy, features)
    else:
        df_clean = sanitizer.transform(df_noisy, features)
    df_clean = df_clean.dropna(subset=features).reset_index(drop=True)

    if fit:
        scaled = scaler.fit_transform(df_clean[features])
    else:
        scaled = scaler.transform(df_clean[features])
    return scaled


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline di sanificazione dati e training LSTM.")

    # Parametri File e Dati
    parser.add_argument("--data_path", type=str, default="data/raw/beijing_data.csv", help="Percorso del dataset raw")
    parser.add_argument("--station", type=str, default="Aotizhongxin", help="Stazione da usare per il training")
    parser.add_argument("--target_col", type=str, default="PM2.5", help="Colonna da prevedere")

    # Parametri Modello
    parser.add_argument("--seq_length", type=int, default=24, help="Ore di storico per la predizione")
    parser.add_argument("--hidden_size", type=int, default=64, help="Dimensione hidden layer LSTM")
    parser.add_argument("--num_layers", type=int, default=2, help="Numero di layer LSTM")
    parser.add_argument("--dropout", type=float, default=0.3, help="Probabilita' di Dropout")

    # Parametri Training
    parser.add_argument("--batch_size", type=int, default=64, help="Dimensione del batch")
    parser.add_argument("--epochs", type=int, default=20, help="Numero di epoche")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    # Parametri Split (cronologico, no shuffle: preserva l'ordine temporale)
    parser.add_argument("--val_size", type=float, default=0.15, help="Frazione finale del training riservata a validation")
    parser.add_argument("--test_size", type=float, default=0.15, help="Frazione finale riservata a test, mai vista in training/model selection")

    # Parametri Wandb
    parser.add_argument("--wandb_project", type=str, default="beijing-air-quality", help="Nome progetto W&B")
    parser.add_argument("--run_name", type=str, default="experiment_1", help="Nome della run su W&B")

    return parser.parse_args()

def main():
    # Inizializza i parser
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Dispositivo in uso: {device}")

    # Inizializza W&B passando direttamente l'oggetto args
    wandb.init(project=args.wandb_project, config=vars(args), name=args.run_name)

    # 1. Caricamento Dati
    print(f"Caricamento dati da {args.data_path} per la stazione {args.station}...")
    df = pd.read_csv(args.data_path)
    df_station = df[df['station'] == args.station].reset_index(drop=True)

    features = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP']
    target_idx = features.index(args.target_col)

    # 2. Split CRONOLOGICO in tre parti, PRIMA di iniettare rumore o sanificare.
    #    E' il punto chiave: cosi' nessuna statistica di sanificazione o di
    #    scaling stimata su dati futuri (val/test) puo' "trapelare" nel training.
    n = len(df_station)
    val_start = int(n * (1 - args.val_size - args.test_size))
    test_start = int(n * (1 - args.test_size))

    df_train_raw = df_station.iloc[:val_start].reset_index(drop=True)
    df_val_raw = df_station.iloc[val_start:test_start].reset_index(drop=True)
    df_test_raw = df_station.iloc[test_start:].reset_index(drop=True)
    print(f"Righe -> train: {len(df_train_raw)} | val: {len(df_val_raw)} | test: {len(df_test_raw)}")

    # 3. Iniezione rumore + sanificazione + scaling: fit SOLO sul training set
    injector = SensorErrorInjector()
    sanitizer = DataSanitizer()
    scaler = MinMaxScaler()

    train_scaled = process_split(df_train_raw, features, injector, sanitizer, scaler, fit=True)
    val_scaled = process_split(df_val_raw, features, injector, sanitizer, scaler, fit=False)
    test_scaled = process_split(df_test_raw, features, injector, sanitizer, scaler, fit=False)

    # 4. Creazione sequenze. Per val/test si antepone la coda (seq_length
    #    righe) della partizione precedente, gia' scalata, come "contesto"
    #    storico: senza, si perderebbero le prime seq_length ore di ciascuna
    #    partizione, che altrimenti non avrebbero abbastanza storico a monte.
    X_train, y_train = create_sequences(train_scaled, target_idx, seq_length=args.seq_length)

    val_with_context = np.vstack([train_scaled[-args.seq_length:], val_scaled])
    X_val, y_val = create_sequences(val_with_context, target_idx, seq_length=args.seq_length)

    test_with_context = np.vstack([val_scaled[-args.seq_length:], test_scaled])
    X_test, y_test = create_sequences(test_with_context, target_idx, seq_length=args.seq_length)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)

    # 5. Baseline di persistenza: y_hat(t) = ultimo valore osservato in input.
    #    Riferimento per capire se l'LSTM impara davvero un pattern o si
    #    limita a "copiare" l'ultimo valore di un segnale gia' smussato.
    baseline_val = persistence_baseline_mse(X_val, y_val, target_idx)
    print(f"Baseline di persistenza (val, scala 0-1): {baseline_val:.4f}")
    wandb.log({"baseline_persistence_val_mse": baseline_val})

    # 6. Inizializzazione Modello
    model = AirQualityLSTM(
        input_size=len(features),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout_prob=args.dropout
    )

    # 7. Addestramento (la scelta del checkpoint migliore si basa su val_loss)
    print(f"\n--- Inizio Training: {args.epochs} Epoche ---")
    save_path = 'models/best_lstm_model.pth'
    best_loss = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_path=save_path
    )
    print(f"\nTraining completato. Miglior Val Loss: {best_loss:.4f} (baseline persistenza: {baseline_val:.4f})")

    # 8. Valutazione finale, UNA SOLA VOLTA, sul test set: non e' mai stato
    #    usato ne' in training ne' per scegliere il checkpoint (quello ha
    #    usato solo il validation set). Ricarica i pesi del miglior checkpoint
    #    prima di valutare, dato che a fine training il modello in memoria
    #    e' quello dell'ultima epoca, non necessariamente il migliore.
    model.load_state_dict(torch.load(save_path, map_location=device))
    baseline_test = persistence_baseline_mse(X_test, y_test, target_idx)
    rmse, mae = report_original_scale_metrics(
        model, test_loader, scaler, target_idx, len(features), device, target_name=args.target_col
    )
    print(f"Baseline di persistenza (test, scala 0-1): {baseline_test:.4f}")
    wandb.log({
        "baseline_persistence_test_mse": baseline_test,
        "test_rmse_original_scale": rmse,
        "test_mae_original_scale": mae,
    })

    wandb.finish()

if __name__ == "__main__":
    main()