import bisect

# import torch
# import torch.nn as nn

# class SimpleNet(nn.Module):
#
#     def __init__(self) -> None:
#         super().__init__()
#         self.embed = nn.Embedding(2048, 1024)
#         self.proj1 = nn.Linear(1024, 1024)
#         self.ln1 = nn.LayerNorm(1024)
#         self.proj2 = nn.Linear(1024, 2048)
#         self.ln2 = nn.LayerNorm(2048)
#         self.classifier = nn.Linear(2048, 2048)
#
#     def forward(self, x):
#         x = self.embed(x)
#         x = self.proj1(x)
#         x = self.ln1(x)
#         x = self.proj2(x)
#         x = self.ln2(x)
#         x = self.classifier(x)
#         return x
#
# class NetWithRepeatedlyComputedLayers(nn.Module):
#
#     def __init__(self) -> None:
#         super().__init__()
#         self.fc1 = nn.Linear(1024, 1024)
#         self.fc2 = nn.Linear(1024, 1024)
#         self.fc3 = nn.Linear(1024, 512)
#         self.layers = [self.fc1, self.fc2, self.fc1, self.fc2, self.fc3]
#
#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x
#
# model = SimpleNet()
# ps = set()
# for p in model.parameters():
#     # print(ps.get(p, None) is None)
#     # ps[p] = 1
#     ps.add(p)
# p = model.proj2.weight
# # print(ps.get(p, None) is None)
# print(p in ps)

# print("*******************************")
# model = NetWithRepeatedlyComputedLayers()
# ps = {}
# for p in model.parameters():
#     print(ps.get(p, None) is None)
#     ps[p] = 1
# p = model.fc3.weight
# print(ps.get(p, None) is None)

import numpy as np
import random
import torch

# aaa = ['a','b','c','d','e']
# bbb = [[0,0],[1,1],[2,2],[3,3],[4,4]]
# ccc = list(zip(aaa, bbb))
# random.shuffle(ccc)
# aaa[:], bbb[:] = zip(*ccc)
# print(aaa)
# print(bbb)

a = [1,2,3]
print(torch.as_tensor(a))
a = torch.as_tensor(a)
bbb = a.cumsum(0).tolist()
print(bbb)
print(bisect.bisect_right(bbb, 2))