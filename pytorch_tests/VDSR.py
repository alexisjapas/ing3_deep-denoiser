from torch import nn


class VDSR(nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        self.input_layer = nn.Conv2d(in_channels=1,
                                     out_channels=64,
                                     kernel_size=3,
                                     padding='same')
        conv2d_relu = []
        for i in range(n_layers):
            conv2d_relu.append(nn.Conv2d(in_channels=64,
                                         out_channels=64,
                                         kernel_size=3,
                                         padding='same')
                               )
            conv2d_relu.append(nn.ReLU())
        self.conv2d_relu = nn.Sequential(*conv2d_relu)
        self.output_layer = nn.Conv2d(in_channels=64,
                                      out_channels=1,
                                      kernel_size=3,
                                      padding='same')
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.input_layer(x)
        x = self.conv2d_relu(x)
        x = identity + self.output_layer(x)
        return self.relu(x)
