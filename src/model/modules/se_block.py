from torch import nn


class SEBlock1D(nn.Module):
    def __init__(self, channel: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False), nn.ReLU(), nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channel, _ = x.size()
        scaler = self.avg_pool(x).view(batch_size, channel)
        scaler = self.fc(scaler).view(batch_size, channel, 1)
        return x * scaler
