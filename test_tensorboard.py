import torch
import torch.nn
import torch.optim
import torch.profiler
import torch.utils.data
import torchvision.datasets
import torchvision.models
import torchvision.transforms as T

from model_utils import *

def train(data=None):
    # inputs, labels = data[0].to(device=device), data[1].to(device=device)
    # outputs = model(inputs)
    # loss = criterion(outputs, labels)

    loss = torch.mean(model(**data_args))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# transform = T.Compose(
#     [T.Resize(224),
#      T.ToTensor(),
#      T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)

# device = torch.device("cuda:0")
# model = torchvision.models.resnet18(pretrained=False).cuda(device)
# criterion = torch.nn.CrossEntropyLoss().cuda(device)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# model.train()

# build model
device = torch.device("cuda:0")
get_components_func = non_distributed_component_funcs.get_callable('simplenet')
model_builder, data_gen = get_components_func()
data_args = data_gen(device=device)
model = model_builder().cuda(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model.train()


with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
) as prof:
    # for step, batch_data in enumerate(train_loader):
    #     if step >= (1 + 1 + 3) * 2:
    #         break
    #     train(batch_data)
    #     prof.step()

    for step in range(20):
        if step >= (1 + 1 + 3) * 2:
            break
        train()
        prof.step()
