"""
Funzioni di valutazione: baseline di persistenza e metriche in scala
originale (invece della sola MSE su dati normalizzati 0-1).
"""
import numpy as np
import torch


def persistence_baseline_mse(X, y, target_idx):
    """
    Baseline banale: prevede che il valore al tempo t sia uguale
    all'ultimo valore osservato nella sequenza di input (t-1).

    Se l'LSTM non fa sensibilmente meglio di questa baseline, vuol dire
    che non sta imparando molto oltre al "copia l'ultimo valore" -- un
    rischio concreto quando il target e' stato smussato a monte (vedi
    DataSanitizer.apply_smoothing), perche' rende la serie molto
    autocorrelata e quindi facile da "prevedere" per persistenza.
    """
    y_pred_baseline = X[:, -1, target_idx].unsqueeze(1)
    return torch.mean((y_pred_baseline - y) ** 2).item()


def get_predictions(model, data_loader, device):
    """Esegue il modello su un intero DataLoader e concatena predizioni e target."""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu()
            all_preds.append(preds)
            all_targets.append(y_batch)
    return torch.cat(all_preds), torch.cat(all_targets)


def inverse_transform_target(values, scaler, target_idx, num_features):
    """
    Inverte il MinMaxScaler solo per la colonna target. Lo scaler e' stato
    fittato su tutte le feature insieme, quindi serve un array con lo
    stesso numero di colonne: le altre vengono riempite con zeri, dato
    che il MinMaxScaler inverte ogni colonna in modo indipendente dalle altre.
    """
    values = np.asarray(values).reshape(-1, 1)
    dummy = np.zeros((values.shape[0], num_features))
    dummy[:, target_idx] = values[:, 0]
    inverted = scaler.inverse_transform(dummy)
    return inverted[:, target_idx]


def report_original_scale_metrics(model, data_loader, scaler, target_idx, num_features, device, target_name="target"):
    """
    Calcola RMSE e MAE nell'unita' di misura originale (es. ug/m3 per il
    PM2.5), molto piu' interpretabili della sola MSE su scala 0-1 -- ed e'
    quello che ha senso mostrare/discutere in un colloquio.
    """
    preds_scaled, targets_scaled = get_predictions(model, data_loader, device)
    preds_orig = inverse_transform_target(preds_scaled.detach().numpy(), scaler, target_idx, num_features)
    targets_orig = inverse_transform_target(targets_scaled.detach().numpy(), scaler, target_idx, num_features)

    rmse = float(np.sqrt(np.mean((preds_orig - targets_orig) ** 2)))
    mae = float(np.mean(np.abs(preds_orig - targets_orig)))

    print(f"\nMetriche in scala originale ({target_name}):")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE:  {mae:.2f}")

    return rmse, mae

def compute_mse(model, data_loader, device):
    """MSE del modello in scala 0-1 (stesse unita' della baseline), utile per
    calcolare lo skill ratio senza dover invertire lo scaler."""
    preds, targets = get_predictions(model, data_loader, device)
    return torch.mean((preds - targets) ** 2).item()


def skill_ratio(model_mse, baseline_mse):
    """
    < 1  -> il modello batte la baseline di persistenza (sta imparando qualcosa)
    ~ 1  -> il modello non fa meglio del semplice "copia l'ultimo valore"
    > 1  -> il modello e' peggio della baseline banale
    """
    return model_mse / baseline_mse if baseline_mse > 0 else float('nan')