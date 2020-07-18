#! /usr/bin/env python3

import torch


class CNNModel(torch.nn.Module):

    def __init__(self,
                 batch_norm=True,
                 activation='ReLU',
                 activation_params={},
                 ):
        super().__init__()

        # pure unadultered genius
        self.activation = torch.nn.__dict__[activation](**activation_params)

        self.bn_activated = batch_norm

        # first convolution section
        self.conv_1 = torch.nn.Conv2d(in_channels=1,
                                      out_channels=1,
                                      kernel_size=3,
                                      stride=1,
                                      padding=0,
                                      )
        self.maxpool_1 = torch.nn.MaxPool2d(kernel_size=2,
                                            stride=1,
                                            padding=0,
                                            )

        # second convolution section
        self.conv_2 = torch.nn.Conv2d(in_channels=1,
                                      out_channels=1,
                                      kernel_size=3,
                                      stride=1,
                                      padding=0,
                                      )
        self.maxpool_2 = torch.nn.MaxPool2d(kernel_size=2,
                                            stride=1,
                                            padding=0,
                                            )

        if self.bn_activated:
            self.norm_conv = torch.nn.BatchNorm2d(num_features=1)
        else:
            self.norm_conv = torch.nn.Identity()

        # linear section
        self.linear = torch.nn.Linear(in_features=16, out_features=2)

        if self.bn_activated:
            self.norm_linear = torch.nn.BatchNorm1d(num_features=1)
        else:
            self.norm_linear = torch.nn.Identity()

        # softmax
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.activation(x)
        x = self.maxpool_1(x)

        x = self.conv_2(x)
        x = self.activation(x)
        x = self.norm_conv(x)
        x = self.maxpool_2(x)
        x = x.reshape(-1, 1, 16)

        x = self.linear(x)
        x = self.activation(x)
        x = self.norm_linear(x)

        x = self.softmax(x)

        return x


if __name__ == '__main__':
    mod = CNNModel()

    t = torch.rand(size=(5, 1, 10, 10))

    print(mod(t))
