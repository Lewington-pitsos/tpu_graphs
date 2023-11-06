import torch
import torch.nn as nn

class FeatureConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_width, stride_width=1):
        super(FeatureConv, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_width,),
            padding=(kernel_width // 2,),
          )


    def forward(self, x):
        return self.conv(x)


class OneDModel(nn.modules.Module):
    def __init__(self, num_features):
        super(OneDModel, self).__init__()
        self.conv1 = FeatureConv(in_channels=num_features, out_channels=128, kernel_width=5, stride_width=1)
        self.conv2 = FeatureConv(in_channels=128, out_channels=128, kernel_width=5, stride_width=1)
        self.conv3 = FeatureConv(in_channels=128, out_channels=128, kernel_width=5, stride_width=1)

        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = torch.mean(x, dim=2)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

model = OneDModel(num_features=141)

input_tensor = torch.randn(2, 141, 27)

output = model(input_tensor)

print(output)
