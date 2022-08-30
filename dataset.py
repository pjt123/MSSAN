from paddle.io import Dataset
import matplotlib.pyplot as plt



# 定义GetLoader类，继承Dataset方法，并重写__getitem__()和__len__()方法
class CyjDataset(Dataset):
	# 初始化函数，得到数据
    def __init__(self, data, label, transform=None):
        self.data = data
        self.label = label.squeeze()
        self.transform = transform

        # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼

    def __len__(self):
        return len(self.data)

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]

        if self.transform:
            data = self.transform(data)

        return data, label