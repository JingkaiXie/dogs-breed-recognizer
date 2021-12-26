import torch
import torch.nn.functional as F
import torchvision.models as models


# class CNNClassifier(torch.nn.Module):
#     class Block(torch.nn.Module):
#         def __init__(self, n_input, n_output, stride=1):
#             super().__init__()
#             self.net = torch.nn.Sequential(
#                 torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride),
#                 torch.nn.BatchNorm2d(n_output),
#                 torch.nn.ReLU(),
#                 torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1),
#                 torch.nn.BatchNorm2d(n_output),
#                 torch.nn.ReLU(),
#                 torch.nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
#
#             )
#             self.downsample = None
#             if stride != 1 or n_output != n_input:
#                 self.downsample = torch.nn.Sequential(
#                     torch.nn.Conv2d(n_input, n_output, 1, stride=stride),
#                     torch.nn.BatchNorm2d(n_output)
#                 )
#
#         def forward(self, x):
#             identity = x
#             if self.downsample is not None:
#                 identity = self.downsample(x)
#             return self.net(x) + identity
#
#     def __init__(self, layers=None, n_input_channels=3):
#         super().__init__()
#         if layers is None:
#             layers = [32, 64, 128, 256]
#
#         L = [
#             torch.nn.Conv2d(n_input_channels, 32, kernel_size=7, padding=3, stride=2),
#             torch.nn.BatchNorm2d(32),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
#         ]
#         c = 32
#         for l in layers:
#             L.append(self.Block(c, l, stride=2))
#             c = l
#         L.append(torch.nn.Dropout2d())
#         self.net = torch.nn.Sequential(*L)
#         self.classifier = torch.nn.Linear(c, 120)
#         # self.relu1 = torch.nn.ReLU()
#         # self.classifier2 = torch.nn.Linear(64, 120)
#         # torch.nn.init.zeros_(self.classifier2.weight)
#         # self.mean = torch.tensor([0.3235, 0.3310, 0.3445])
#         # self.std = torch.tensor([0.1954, 0.1763, 0.1878])
#
#     def forward(self, x):
#         # x[:, 0, :, :] = (x[:, 0, :, :] - self.mean[0]) / self.std[0]
#         # x[:, 1, :, :] = (x[:, 1, :, :] - self.mean[1]) / self.std[1]
#         # x[:, 2, :, :] = (x[:, 2, :, :] - self.mean[2]) / self.std[2]
#         x = self.net(x)
#         x = x.mean(dim=[2, 3])
#         x = self.classifier(x.view(x.size(0), -1))
#         # x = self.relu1(x)
#         # x = self.classifier2(x)
#         x = F.softmax(x)
#         return x

class CNNClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.resnet18(pretrained=True)

    def forward(self, x):
        return self.net(x)


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, CNNClassifier):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'cnn.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = CNNClassifier()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th'), map_location='cpu'))
    return r
