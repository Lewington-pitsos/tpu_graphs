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

        self.fc1 = nn.Linear(152, 128)
        self.fc2 = nn.Linear(128, 1)

        self.relu = nn.ReLU()

    def forward(self, node_feat, config_feat):
        node_feat = node_feat.permute(0, 2, 1)

        node_feat = self.conv1(node_feat)
        node_feat = self.relu(node_feat)

        node_feat = self.conv2(node_feat)
        node_feat = self.relu(node_feat)

        node_feat = self.conv3(node_feat)
        node_feat = self.relu(node_feat)

        node_feat = torch.mean(node_feat, dim=2)
        combined = torch.concat([node_feat, config_feat], dim=1)

        combined = self.relu(self.fc1(combined))
        combined = self.fc2(combined)

        return combined

if __name__ == '__main__':
    model = OneDModel(num_features=141)

    input_tensor = torch.randn(2, 27, 141)
    config_feat = torch.randn(2, 24)

    output = model(input_tensor, config_feat)

    print(output)
