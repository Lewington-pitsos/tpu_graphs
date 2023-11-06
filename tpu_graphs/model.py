import torch
import torch.nn as nn

class FeatureConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_width):
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
        self.conv1 = FeatureConv(in_channels=num_features, out_channels=256, kernel_width=5)
        self.conv2 = FeatureConv(in_channels=256, out_channels=512, kernel_width=5)
        self.conv3 = FeatureConv(in_channels=512, out_channels=512, kernel_width=3)
        self.conv4 = FeatureConv(in_channels=512, out_channels=1024, kernel_width=3)
        self.conv5 = FeatureConv(in_channels=1024, out_channels=1024, kernel_width=3)


        self.fc1 = nn.Linear(1048, 1048)
        self.fc2 = nn.Linear(1048, 512)
        self.fc3 = nn.Linear(512, 512)

        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 1)

        self.activation = nn.Tanh()

    def forward(self, node_feat, config_feat):
        node_feat = node_feat.permute(0, 2, 1)

        node_feat = self.conv1(node_feat)
        node_feat = self.activation(node_feat)

        node_feat = self.conv2(node_feat)
        node_feat = self.activation(node_feat)

        node_feat = self.conv3(node_feat)
        node_feat = self.activation(node_feat)

        node_feat = self.conv4(node_feat)
        node_feat = self.activation(node_feat)

        node_feat = self.conv5(node_feat)
        node_feat = self.activation(node_feat)

        node_feat = torch.mean(node_feat, dim=2)
        combined = torch.concat([node_feat, config_feat], dim=1)

        combined = self.activation(self.fc1(combined))
        combined = self.activation(self.fc2(combined))
        combined = self.activation(self.fc3(combined))
        combined = self.activation(self.fc4(combined))

        combined = self.fc5(combined)

        return combined

if __name__ == '__main__':
    model = OneDModel(num_features=141)

    input_tensor = torch.randn(2, 27, 141)
    config_feat = torch.randn(2, 24)

    output = model(input_tensor, config_feat)

    print(output)
