import torch
import torch.nn as nn

from osnet import *

model = osnet_x1_0(751)
model_dict = model.state_dict()
# print(model_dict)
print("loaded statedict")


load_dict = torch.load("/home/louie/External/deep-person-reid/trained_checkpoints/model.pth.tar-100")
# load_dict = load_dict['state_dict']
loaded_dict = {}
for item, param in load_dict['state_dict'].items():
    item = item.replace("module.", "")
    if item not in model_dict:
        print(item)
    loaded_dict[item] = param

model.load_state_dict(loaded_dict)

model.classifier = nn.Sequential(
    nn.Linear(512, 128, bias=True),
    nn.ReLU(inplace=True),
    nn.Linear(128, 2),
    nn.Sigmoid()
)
for n, param in model.named_parameters():
    print(n)

model.eval()
x = torch.randn(1, 3, 64, 128)
print(model)
x = model(x)
print(x.size())
## Data loader and 