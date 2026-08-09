# src/train.py (AGGIORNATO)
import torch
import torch.nn as nn
import torch.optim as optim
import os
import wandb # <--- AGGIUNTO

def add_gaussian_noise(tensor, mean=0.0, std=0.05):
    noise = torch.randn(tensor.size()) * std + mean
    return tensor + noise.to(tensor.device)

def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, device='cpu', save_path='models/best_lstm_model.pth'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    best_val_loss = float('inf')
    model.to(device)

    # wandb.watch traccia i gradienti e i pesi del modello (opzionale ma fantastico)
    wandb.watch(model, criterion, log="all", log_freq=10)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            X_batch_noisy = add_gaussian_noise(X_batch, std=0.02)

            optimizer.zero_grad()
            predictions = model(X_batch_noisy)
            loss = criterion(predictions, y_batch)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)

        print(f'Epoch {epoch+1:03d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

        # ---> LOG SU WANDB <---
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f" ---> Miglioramento! Modello salvato in {save_path}")
            # Opzionale: puoi salvare anche il modello come artefatto su wandb

    return best_val_loss