from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
# from PIL import Image
import glob
import numpy as np
import warnings
import torch
import torch.nn as nn

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

warnings.filterwarnings("ignore")

'''data_transform = transforms.Compose(
      [transforms.Grayscale(),
      transforms.Resize(28),
      transforms.ToTensor(),
      transforms.Normalize([0.5], [0.5])
     ])'''


class CNNnet(nn.Module):
    def __init__(self):
        super(CNNnet, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=16,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 3, 2, 1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 2, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 2, 2, 0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.mlp1 = torch.nn.Linear(1152, 100)
        self.mlp2 = torch.nn.Linear(100, 5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mlp1(x.view(x.size(0), -1))
        x = self.mlp2(x)
        return x


net = CNNnet()
# print(net)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
opt = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


class my_dataset(Dataset):
    def __init__(self, store_path, split, data_transform=None):
        self.store_path = store_path
        self.split = split
        self.transforms = data_transform
        self.img_list = []
        self.label_list = []
        for file in glob.glob(self.store_path + '/' + split + '/000' + '/*.npy'):
            cur_path = file.replace('\\', '/')
            cur_label = 0
            self.img_list.append(cur_path)
            self.label_list.append(cur_label)
        for file in glob.glob(self.store_path + '/' + split + '/001' + '/*.npy'):
            cur_path = file.replace('\\', '/')
            cur_label = 1
            self.img_list.append(cur_path)
            self.label_list.append(cur_label)
        for file in glob.glob(self.store_path + '/' + split + '/002' + '/*.npy'):
            cur_path = file.replace('\\', '/')
            cur_label = 2
            self.img_list.append(cur_path)
            self.label_list.append(cur_label)
        for file in glob.glob(self.store_path + '/' + split + '/003' + '/*.npy'):
            cur_path = file.replace('\\', '/')
            cur_label = 3
            self.img_list.append(cur_path)
            self.label_list.append(cur_label)
        for file in glob.glob(self.store_path + '/' + split + '/004' + '/*.npy'):
            cur_path = file.replace('\\', '/')
            cur_label = 4
            self.img_list.append(cur_path)
            self.label_list.append(cur_label)

    def __getitem__(self, item):
        img = np.load(self.img_list[item])
        if img.shape[2] < 150:
            img = np.expand_dims(img, 0).repeat(5, axis=3)
        img = img.squeeze()
        # print(img.shape)
        img = img[0::2, 0:150, :, 0]
        img = torch.from_numpy(img)
        img = img.reshape(1, 150, 34)
        img = img.to(torch.float32)

        # img = img [-1,-1,:,:,:]
        # img = img.resize((224, 224), Image.ANTIALIAS)
        if self.transforms is not None:
            img = self.transforms(img)
        label = self.label_list[item]
        return img, label

    def __len__(self):
        return len(self.img_list)


import numpy as np

'''import tool
from tool.visualise import visualise
from tool.graph import Graph'''
import matplotlib.pyplot as plt
from torch.autograd import Variable  # 获取变量

if __name__ == '__main__':
    store_path = r"data"
    split = 'train'
    splitt = 'test'
    train_dataset = my_dataset(store_path, split)
    test_dataset = my_dataset(store_path, splitt)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=1)

    '''for i, item in enumerate(train_dataset):
        data, label = item
        print('data:', data.shape)
        print('label:', label)
        break'''

    # for batch_idx, (inputs, targets) in enumerate(dataset_loader):
    #     print(inputs.shape)
    #     print(targets.shape)
    #     break

    '''dataiter = iter(dataset_loader)
    images, labels = dataiter.next()'''

    loss_count = []
    macc = 0
    for epoch in range(100):
        for i, (x, y) in enumerate(train_loader):
            batch_x = Variable(x)  # torch.Size([128, 1, 30, 34])
            batch_y = Variable(y)  # torch.Size([128])
            # 获取最后输出
            out = net(batch_x)  # torch.Size([128,10])
            # 获取损失
            loss = criterion(out, batch_y)
            # 使用优化器优化损失
            opt.zero_grad()  # 清空上一步残余更新参数值
            loss.backward()  # 误差反向传播，计算参数更新值
            opt.step()  # 将参数更新值施加到net的parmeters上
        loss_count.append(loss.detach().numpy())
        print('{}:\t'.format(epoch + 1), 'Loss:\t', loss.item())
        for a, b in test_loader:
            test_x = Variable(a)
            test_y = Variable(b)
            out = net(test_x)
            # print('test_out:\t',torch.max(out,1)[1])
            # print('test_y:\t',test_y)
            accuracy = torch.max(out, 1)[1].numpy() == test_y.numpy()
            print('{}:\t'.format(epoch + 1), 'accuracy:\t', accuracy.mean())
            uu = accuracy.mean()
        if uu > macc:
            macc = uu
            endnet = net
            PATH = r"./my_net.pth"
            torch.save(net.state_dict(), PATH)
    print('best accuracy:\t', macc)
    print(endnet)
    plt.figure('PyTorch_CNN_Loss')
    # print(loss_count)
    plt.plot(loss_count, label='Loss')
    plt.legend()
    plt.show()
