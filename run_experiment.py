import argparse
import numpy as np
import torch
import wandb
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

from src.dataset import SensorErrorInjector
from src.data_sanitization import DataSanitizer
from src.model import AirQualityLSTM
from src.train import train_model
from src.evaluate import persistence_baseline_mse, report_original_scale_metrics
from src.evaluate import compute_mse, skill_ratio


def create_sequences(data, target_col_idx, seq_length=24):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length, target_col_idx]
        xs.append(x)
        ys.append(y)
    return torch.tensor(np.array(xs), dtype=torch.float32), torch.tensor(np.array(ys), dtype=torch.float32).unsqueeze(1)


def process_split(df_raw, features, injector, sanitizer, scaler, fit=False):
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


def evaluate_cross_station(df_full, station_name, features, injector, sanitizer, scaler,
                            model, target_idx, seq_length, batch_size, device):
    """
    Valuta il modello (addestrato su UNA stazione) su una stazione mai vista
    in training: e' il vero test di generalizzazione "out-of-domain".
    """
    df_station = df_full[df_full['station'] == station_name].reset_index(drop=True)
    scaled = process_split(df_station, features, injector, sanitizer, scaler, fit=False)
    X, y = create_sequences(scaled, target_idx, seq_length=seq_length)
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)

    baseline_mse = persistence_baseline_mse(X, y, target_idx)
    model_mse = compute_mse(model, loader, device)
    rmse, mae = report_original_scale_metrics(
        model, loader, scaler, target_idx, len(features), device, target_name=station_name
    )
    return {
        "station": station_name,
        "baseline_mse": baseline_mse,
        "model_mse": model_mse,
        "skill_ratio": skill_ratio(model_mse, baseline_mse),
        "rmse": rmse,
        "mae": mae,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline di sanificazione dati e training LSTM.")
    parser.add_argument("--data_path", type=str, default="data/raw/beijing_data.csv")
    parser.add_argument("--station", type=str, default="Aotizhongxin")
    parser.add_argument("--target_col", type=str, default="PM2.5")
    parser.add_argument("--seq_length", type=int, default=24)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--wandb_project", type=str, default="beijing-air-quality")
    parser.add_argument("--run_name", type=str, default="experiment_1")
    parser.add_argument("--generalization_stations", type=str, default="",
        help="Stazioni (separate da virgola) MAI usate in training, per testare la generalizzazione")
    return parser.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Dispositivo in uso: {device}")

    wandb.init(project=args.wandb_project, config=vars(args), name=args.run_name)

    print(f"Caricamento dati da {args.data_path} per la stazione {args.station}...")
    df = pd.read_csv(args.data_path)
    df_station = df[df['station'] == args.station].reset_index(drop=True)

    features = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP']
    target_idx = features.index(args.target_col)

    n = len(df_station)
    val_start = int(n * (1 - args.val_size - args.test_size))
    test_start = int(n * (1 - args.test_size))

    df_train_raw = df_station.iloc[:val_start].reset_index(drop=True)
    df_val_raw = df_station.iloc[val_start:test_start].reset_index(drop=True)
    df_test_raw = df_station.iloc[test_start:].reset_index(drop=True)
    print(f"Righe -> train: {len(df_train_raw)} | val: {len(df_val_raw)} | test: {len(df_test_raw)}")

    injector = SensorErrorInjector()
    sanitizer = DataSanitizer()
    scaler = MinMaxScaler()

    train_scaled = process_split(df_train_raw, features, injector, sanitizer, scaler, fit=True)
    val_scaled = process_split(df_val_raw, features, injector, sanitizer, scaler, fit=False)
    test_scaled = process_split(df_test_raw, features, injector, sanitizer, scaler, fit=False)

    X_train, y_train = create_sequences(train_scaled, target_idx, seq_length=args.seq_length)

    val_with_context = np.vstack([train_scaled[-args.seq_length:], val_scaled])
    X_val, y_val = create_sequences(val_with_context, target_idx, seq_length=args.seq_length)

    test_with_context = np.vstack([val_scaled[-args.seq_length:], test_scaled])
    X_test, y_test = create_sequences(test_with_context, target_idx, seq_length=args.seq_length)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)

    baseline_val = persistence_baseline_mse(X_val, y_val, target_idx)
    print(f"Baseline di persistenza (val, scala 0-1): {baseline_val:.4f}")
    wandb.log({"baseline_persistence_val_mse": baseline_val})

    model = AirQualityLSTM(
        input_size=len(features),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout_prob=args.dropout
    )

    print(f"\n--- Inizio Training: {args.epochs} Epoche ---")
    save_path = 'models/best_lstm_model.pth'
    best_loss = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_path=save_path,
        baseline_val_mse=baseline_val,
    )
    print(f"\nTraining completato. Miglior Val Loss: {best_loss:.4f} (baseline persistenza: {baseline_val:.4f})")

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

    # Generalizzazione su stazioni MAI viste in training -- va dentro main(),
    # dopo l'inizializzazione di df/injector/sanitizer/scaler/model, e prima
    # di wandb.finish() cosi' i log arrivano sulla stessa run.
    if args.generalization_stations:
        print("\n--- Generalizzazione su stazioni mai viste ---")
        for station in [s.strip() for s in args.generalization_stations.split(",")]:
            r = evaluate_cross_station(df, station, features, injector, sanitizer, scaler,
                                        model, target_idx, args.seq_length, args.batch_size, device)
            print(f"{station}: skill_ratio={r['skill_ratio']:.2f} | RMSE={r['rmse']:.2f} | MAE={r['mae']:.2f}")
            wandb.log({
                f"generalization/{station}_baseline_mse": r["baseline_mse"],
                f"generalization/{station}_skill_ratio": r["skill_ratio"],
                f"generalization/{station}_rmse": r["rmse"],
                f"generalization/{station}_mae": r["mae"],
            })

    wandb.finish()


if __name__ == "__main__":
    main()