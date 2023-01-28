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

from dataclasses import dataclass
@dataclass
class Region:
    r_id: int = 0
    is_offload: bool = False
    param_size: int = 0
    region_to_prefetch = None
    is_syn: bool = False
    region_shared_param = None

r1 = Region(r_id=1)
r2 = Region(r_id=2)
r1.region_to_prefetch = r2

orig = r1.region_to_prefetch
r1.region_to_prefetch = None

print(orig)
print(r1.region_to_prefetch)


aaa = {'a':1,'b':3}
k, v = list(aaa.items())[0]
print(k)