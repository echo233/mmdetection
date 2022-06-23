import torch
content = torch.load('/liu/code/d2l-zh/pycharm-local/FADI-Main/models/voc_split1_base.pth')
item = content.keys()
item = list(item)
for i in item:
    print(i)
print(1)