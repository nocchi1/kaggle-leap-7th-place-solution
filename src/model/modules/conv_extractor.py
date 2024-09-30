from torch import nn


class ConvExtractor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.fc_1 = nn.Linear(in_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln_1 = nn.LayerNorm([60, hidden_dim])
        self.ln_2 = nn.LayerNorm([60, hidden_dim])
        self.ln_3 = nn.LayerNorm([60, hidden_dim])
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.LayerNorm([hidden_dim, 60]),
            nn.ELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.LayerNorm([hidden_dim, 60]),
            nn.ELU(),
        )
        self.act = nn.ELU()

    def forward(self, x):
        x = self.fc_1(x)
        x = self.ln_1(x)
        x_s = self.act(x)
        x = self.conv(x_s.transpose(1, 2)).transpose(1, 2)
        x = x + x_s
        x = self.ln_2(x)
        x = self.fc_2(x)
        x = self.ln_3(x)
        return x
