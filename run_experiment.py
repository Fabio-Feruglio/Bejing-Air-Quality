import os
import argparse
import torch
import wandb
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Import dai tuoi moduli
from src.dataset import SensorErrorInjector
from src.data_sanitization import DataSanitizer
from src.model import AirQualityLSTM
from src.train import train_model

def create_sequences(data, target_col_idx, seq_length=24):
    """Trasforma un array 2D in sequenze 3D (Batch, Sequence, Features)"""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length, target_col_idx]
        xs.append(x)
        ys.append(y)
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32).unsqueeze(1)

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
    parser.add_argument("--dropout", type=float, default=0.3, help="Probabilità di Dropout")
    
    # Parametri Training
    parser.add_argument("--batch_size", type=int, default=64, help="Dimensione del batch")
    parser.add_argument("--epochs", type=int, default=20, help="Numero di epoche")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    
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
    df_station = df[df['station'] == args.station].copy()
    
    features = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP']
    
    # 2. Iniezione Rumore e Sanificazione
    injector = SensorErrorInjector()
    df_noisy = injector.corrupt_dataset(df_station, features)
    
    sanitizer = DataSanitizer()
    df_clean = sanitizer.run_pipeline(df_noisy, features)
    df_clean = df_clean.dropna(subset=features)
    
    # 3. Preprocessing
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_clean[features])
    target_idx = features.index(args.target_col)
    
    X, y = create_sequences(scaled_data, target_idx, seq_length=args.seq_length)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)
    
    # 4. Inizializzazione Modello
    model = AirQualityLSTM(
        input_size=len(features), 
        hidden_size=args.hidden_size, 
        num_layers=args.num_layers, 
        dropout_prob=args.dropout
    )
    
    # 5. Addestramento
    print(f"\n--- Inizio Training: {args.epochs} Epoche ---")
    best_loss = train_model(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        epochs=args.epochs, 
        lr=args.lr, 
        device=device
    )
    
    print(f"\nTraining completato. Miglior Val Loss: {best_loss:.4f}")
    wandb.finish()

if __name__ == "__main__":
    main()