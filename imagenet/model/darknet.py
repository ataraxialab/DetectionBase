import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch
from torch.autograd import Variable

__all__ = ['DarkNet', 'darknet19']


model_urls = {
    'darknet19': 'https://somewhere in the future',
}


class DarkNet(nn.Module):

    def __init__(self, image_size=224, num_classes=1000):
        super(DarkNet, self).__init__()
        self.features = nn.Sequential(
            BasicConv2d(3, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicConv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicConv2d(64, 128, kernel_size=3, padding=1),
            BasicConv2d(128, 64, kernel_size=1),
            BasicConv2d(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicConv2d(128, 256, kernel_size=3, padding=1),
            BasicConv2d(256, 128, kernel_size=1),
            BasicConv2d(128, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicConv2d(256, 512, kernel_size=3, padding=1),
            BasicConv2d(512, 256, kernel_size=1),
            BasicConv2d(256, 512, kernel_size=3, padding=1),
            BasicConv2d(512, 256, kernel_size=1),
            BasicConv2d(256, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicConv2d(512, 1024, kernel_size=3, padding=1),
            BasicConv2d(1024, 512, kernel_size=1),
            BasicConv2d(512, 1024, kernel_size=3, padding=1),
            BasicConv2d(1024, 512, kernel_size=1),
            BasicConv2d(512, 1024, kernel_size=3, padding=1),
        )
        self.classifier = nn.Sequential(
            BasicConv2d(1024, 1000, kernel_size=1),
            nn.AvgPool2d(kernel_size=7),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 1024, 7, 7)
        x = self.classifier(x)
        x = x.view(x.size(0),-1)
        return x


def darknet19(pretrained=False, **kwargs):
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DarkNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['darknet19']))
    return model


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


if __name__ == '__main__':
    net = darknet19()
    print(net)
    input = Variable(torch.randn(5, 3, 224, 224))
    out = net(input)
    print(out)
