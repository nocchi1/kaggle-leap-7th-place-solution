from torch import nn


class LSTMBlock(nn.Module):
    def __init__(self, hidden_dim: int, scaler_num: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.fc_h = nn.Sequential(
            nn.Linear(scaler_num, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
        )
        self.fc_c = nn.Sequential(
            nn.Linear(scaler_num, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
        )
        self.fc_1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.fc_2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.ln_1 = nn.LayerNorm([60, hidden_dim])
        self.ln_2 = nn.LayerNorm([60, hidden_dim])
        self.act = nn.GELU()
        self._reinitialize()

    # Tensorflow/Keras-like initialization
    def _reinitialize(self):
        for name, p in self.named_parameters():
            if "lstm" in name:
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(p.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(p.data)
                elif "bias_ih" in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4) : (n // 2)].fill_(1)
                elif "bias_hh" in name:
                    p.data.fill_(0)

    def forward(self, x, s_x):
        x_s = x
        h0 = self.fc_h(s_x).unsqueeze(0).repeat(2, 1, 1)  # (bidirectional * n_layer, batch, hidden_dim)
        c0 = self.fc_c(s_x).unsqueeze(0).repeat(2, 1, 1)
        x, _ = self.lstm(x, (h0, c0))
        x = self.ln_1(x)
        x = x + x_s
        x_s = x
        x = self.fc_1(x)
        x = self.act(x)
        x = self.fc_2(x)
        x = self.ln_2(x)
        x = x + x_s
        return x
