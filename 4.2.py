#使用Pytoch和dataloader加载数据集
import torch

class GetLoader(torch.utils.data.Dataset):
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
    def __len__(self):
        return len(self.data)
torch.utils.data.DataLoader(dataset,batch_size,shuffle,drop_last，num_workers)
from torch.utils.data import DataLoader

datas = DataLoader(torch_data, batch_size=6, shuffle=True, drop_last=False, num_workers=2)

#使用nn.module类定义神经网络模型
class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, 1, 1)
        self.relu1 = torch.nn.ReLU()
        self.max_pooling1 = torch.nn.MaxPool2d(2, 1)

        self.conv2 = torch.nn.Conv2d(3, 32, 3, 1, 1)
        self.relu2 = torch.nn.ReLU()
        self.max_pooling2 = torch.nn.MaxPool2d(2, 1)

        self.dense1 = torch.nn.Linear(32 * 3 * 3, 128)
        self.dense2 = torch.nn.Linear(128, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.max_pooling1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max_pooling2(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

model = MyNet()
print(model)

