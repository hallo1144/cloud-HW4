import torch
import torch.nn as nn
import sys
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset

random.seed(0)
torch.manual_seed(0)

class MyDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        Y = self.y[index]
        return X, Y

class Classifier1(nn.Module):
    def __init__(self):
        super(Classifier1, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 2),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.fc(x)

model = Classifier1().cpu()
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.15, momentum=0)

with open(sys.argv[1], 'r') as f:
    raw = f.readlines()
    random.shuffle(raw)

data = []
label = []
for line in raw:
    nums = line.replace('\n', '').split(',')
    data.append([float(x) for x in nums[:-1]])
    label.append(int(nums[-1]))

sp_1 = int(len(data) * 0.6)
sp_2 = int(len(data) * 0.8)
test_data_count = len(data) - sp_2
val_data_count = sp_2 - sp_1

train_data = torch.Tensor(data[:sp_1]).cpu()
train_label = torch.LongTensor(label[:sp_1]).cpu()
train_set = MyDataset(train_data, train_label)
# train_set = MyDataset(data[:train_data_count], label[:train_data_count])
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_data = torch.Tensor(data[sp_1:sp_2]).cpu()
val_label = torch.LongTensor(label[sp_1:sp_2]).cpu()
test_data = torch.Tensor(data[sp_2:]).cpu()
test_label = torch.LongTensor(label[sp_2:]).cpu()

for i in range(100):
    model.train()
    for _, batch_data in enumerate(train_loader):
        optimizer.zero_grad()
        pred = model(batch_data[0])
        batch_loss = loss(pred, batch_data[1]).cpu()
        batch_loss.backward()
        optimizer.step()

    if (i+1) % 10 == 0:
        model.eval()
        pred = model(val_data)
        acc = np.sum(np.argmax(pred.cpu().data.numpy(), axis=1) == val_label.numpy()) / val_data_count
        print("validation on", i, "th epoch:", acc)

model.eval()
pred = model(test_data)
acc = np.sum(np.argmax(pred.cpu().data.numpy(), axis=1) == test_label.numpy()) / test_data_count
# print("test acc: ", acc)