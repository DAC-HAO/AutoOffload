
import torch
import torch.nn as nn

class SimpleNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(2048, 1024)
        self.proj1 = nn.Linear(1024, 1024)
        self.ln1 = nn.LayerNorm(1024)
        self.proj2 = nn.Linear(1024, 2048)
        self.ln2 = nn.LayerNorm(2048)
        self.classifier = nn.Linear(2048, 2048)

    def forward(self, x):
        x = self.embed(x)
        x = self.proj1(x)
        x = self.ln1(x)
        x = self.proj2(x)
        x = self.ln2(x)
        x = self.classifier(x)
        return x

class NetWithRepeatedlyComputedLayers(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.layers = [self.fc1, self.fc2, self.fc1, self.fc2, self.fc3]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

from typing import List
model = SimpleNet()
ps :List[nn.parameter.Parameter] = []
for p in model.parameters():
    print(type(p))
    print(p in ps)
    ps.append(p)

print("*******************************")
model = NetWithRepeatedlyComputedLayers()
ps :List[nn.parameter.Parameter] = []
for p in model.parameters():
    print(type(p))
    print(p in ps)
    ps.append(p)

