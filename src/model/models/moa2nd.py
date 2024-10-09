import torch.nn.functional as F
from torch import nn


# original ref: https://www.kaggle.com/competitions/lish-moa/discussion/202256
# another ref: https://www.kaggle.com/code/nyanpn/1st-place-public-2nd-place-solution
class MoA2ndModel(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_size: int):
        super().__init__()
        self.cha_1 = 64  # 256
        self.cha_2 = 256  # 512
        self.cha_3 = 256  # 512

        self.cha_1_reshape = int(hidden_size / self.cha_1)
        self.cha_po_1 = int(hidden_size / self.cha_1 / 2)
        self.cha_po_2 = int(hidden_size / self.cha_1 / 2 / 2)

        self.batch_norm1 = nn.BatchNorm1d(num_features=in_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.linear1 = nn.utils.weight_norm(nn.Linear(in_dim, hidden_size))

        self.batch_norm_c1 = nn.BatchNorm1d(self.cha_1)
        self.dropout_c1 = nn.Dropout(0.1)
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(
                self.cha_1,
                self.cha_2,
                kernel_size=5,
                stride=1,
                padding=2,
                bias=False,
            ),
            dim=None,
        )
        self.po_c1 = nn.AdaptiveAvgPool1d(output_size=self.cha_po_1)

        self.batch_norm_c2 = nn.BatchNorm1d(self.cha_2)
        self.dropout_c2 = nn.Dropout(0.1)
        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(self.cha_2, self.cha_2, kernel_size=3, stride=1, padding=1, bias=True),
            dim=None,
        )

        self.batch_norm_c2_1 = nn.BatchNorm1d(self.cha_2)
        self.dropout_c2_1 = nn.Dropout(0.3)
        self.conv2_1 = nn.utils.weight_norm(
            nn.Conv1d(self.cha_2, self.cha_2, kernel_size=3, stride=1, padding=1, bias=True),
            dim=None,
        )

        self.batch_norm_c2_2 = nn.BatchNorm1d(self.cha_2)
        self.dropout_c2_2 = nn.Dropout(0.2)
        self.conv2_2 = nn.utils.weight_norm(
            nn.Conv1d(self.cha_2, self.cha_3, kernel_size=5, stride=1, padding=2, bias=True),
            dim=None,
        )

        # self.po_c2 = nn.AdaptiveMaxPool1d(output_size=self.cha_po_2)
        self.po_c2 = nn.AdaptiveAvgPool1d(output_size=self.cha_po_2)
        self.flatten = nn.Flatten()

        self.batch_norm3 = nn.BatchNorm1d(self.cha_3 * self.cha_po_2)
        self.dropout3 = nn.Dropout(0.2)
        self.linear3 = nn.utils.weight_norm(nn.Linear(self.cha_3 * self.cha_po_2, out_dim))

    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.celu(self.linear1(x), alpha=0.06)
        x = x.reshape(x.shape[0], self.cha_1, self.cha_1_reshape)

        x = self.batch_norm_c1(x)
        x = self.dropout_c1(x)
        x = F.relu(self.conv1(x))
        x = self.po_c1(x)

        x = self.batch_norm_c2(x)
        x = self.dropout_c2(x)
        x = F.relu(self.conv2(x))
        x_s = x

        x = self.batch_norm_c2_1(x)
        x = self.dropout_c2_1(x)
        x = F.relu(self.conv2_1(x))

        x = self.batch_norm_c2_2(x)
        x = self.dropout_c2_2(x)
        x = F.relu(self.conv2_2(x))
        x = x + x_s  # x * x_s

        x = self.po_c2(x)
        x = self.flatten(x)  # (batch, cha_2 * cha_po_2)

        x = self.batch_norm3(x)
        # x = self.dropout3(x)
        x = self.linear3(x)
        return x
