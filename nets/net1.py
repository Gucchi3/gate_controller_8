import torch
import torch.nn as nn


class net1_ex(nn.Module):
    def __init__(self):
        super(net1_ex, self).__init__()
        class SEBlock(nn.Module):
            """
            Squeeze-and-Excitation Blockを実装したPyTorchモジュール。
            """
            def __init__(self, channels: int, reduction_ratio: int = 16):
                """
                Args:
                    channels (int): 入力および出力の特徴マップのチャンネル数。
                    reduction_ratio (int, optional): 中間層のチャンネル削減率。デフォルトは16。
                """
                super().__init__()
                
                # 中間層のチャンネル数が最低でも1になるように設定
                squeezed_channels = max(1, channels // reduction_ratio)

                self.squeeze = nn.AdaptiveAvgPool2d(1)
                self.excitation = nn.Sequential(
                    nn.Linear(channels, squeezed_channels, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Linear(squeezed_channels, channels, bias=False),
                    nn.Sigmoid()
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                batch_size, num_channels, _, _ = x.shape
                
                # Squeeze
                y = self.squeeze(x).view(batch_size, num_channels)
                
                # Excitation
                y = self.excitation(y).view(batch_size, num_channels, 1, 1)
                
                # Scale (入力テンソルに重みを乗算)
                return x * y.expand_as(x)

        self.SEBlock1 = SEBlock(channels=32, reduction_ratio=8)
        self.SEBlock2 = SEBlock(channels=48, reduction_ratio=8)
        self.SEBlock3 = SEBlock(channels=48, reduction_ratio=8)
        # conv1ブロック
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1b = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn1b = nn.BatchNorm2d(16)
        self.relu1b = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2)  # 160 -> 80

        # conv2ブロック
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2b = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2b = nn.BatchNorm2d(32)
        self.relu2b = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2)  # 80 -> 40

        # conv3ブロック
        self.conv3 = nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(48)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3b = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1)
        self.bn3b = nn.BatchNorm2d(48)
        self.relu3b = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2)  # 40 -> 20

        # conv4ブロック
        self.conv4 = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(48)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv4b = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1)
        self.bn4b = nn.BatchNorm2d(48)
        self.relu4b = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2)  # 20 -> 10

        self.fc = nn.Linear(10*10*48, 8)  # 1x9

    def forward(self, x):
        # conv1ブロック
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        # x = self.conv1b(x)
        # x = self.bn1b(x)
        # x = self.relu1b(x)
        x = self.pool1(x)

        # conv2ブロック
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.SEBlock1(x)
        # x = self.conv2b(x)
        # x = self.bn2b(x)
        # x = self.relu2b(x)
        x = self.pool2(x)

        # conv3ブロック
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.SEBlock2(x)
        # x = self.conv3b(x)
        # x = self.bn3b(x)
        # x = self.relu3b(x)
        x = self.pool3(x)

        # conv4ブロック
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.SEBlock3(x)
        # x = self.conv4b(x)
        # x = self.bn4b(x)
        # x = self.relu4b(x)
        x = self.pool4(x)

        x = torch.flatten(x, 1)
        out = self.fc(x)
        #print(out)
        return out
        


