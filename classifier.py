from torch import nn


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding='same'
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding='same'
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding='same'
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2),
              nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding='same'
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2)
        )

        self.flatten = nn.Flatten()

        self.dense_stack = nn.Sequential(
            nn.LazyLinear(out_features=128),
            nn.ReLU(),
            nn.LazyLinear(out_features=1),
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        return self.dense_stack(x)