import torch
import torch.nn as nn

class AirQualityLSTM(nn.Module):
    
    
    def __init__(self, input_size, hidden_size, num_layers=2, output_size=1, dropout_prob=0.4):
        super(AirQualityLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Layer LSTM: batch_first=True indica un input di forma (Batch, Sequenza, Features)
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout_prob if num_layers > 1 else 0.0
        )
        
        # Layer di Dropout Aggressivo prima dell'output
        self.dropout = nn.Dropout(dropout_prob)
        
        # Layer lineare per prevedere il valore continuo (es. inquinamento o temperatura futura)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        
        
        
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Estraiamo solo l'output dell'ultimo step temporale della sequenza
        last_time_step_out = lstm_out[:, -1, :]
        
        # Applichiamo il dropout e passiamo al layer finale
        out = self.dropout(last_time_step_out)
        out = self.fc(out)
        
        return out
