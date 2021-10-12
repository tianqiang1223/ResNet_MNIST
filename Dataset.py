
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

def Load_data(data_dir, image_data, label_data):
    with open(data_dir + '/' + label_data, 'rb') as lbpath:  # rb表示的是读取二进制数据
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with open(data_dir + '/' + image_data, 'rb') as lbpath:  # rb表示的是读取二进制数据
        x_train = np.frombuffer(lbpath.read(), np.uint8, offset=16).reshape(len(y_train), 1, 28, 28)

    return x_train, y_train

class create_data:
    def __init__(self, data_dir, image_data, label_data):
        (images_data, labels_data) = Load_data(data_dir, image_data, label_data)
        self.data_dir = data_dir
        self.image_data = images_data
        self.label_data = labels_data

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, item):
        imgs, labels = self.image_data[item], self.label_data[item]
        return imgs, labels


def load_MNIST_data(data_dir, batch_size, images_data, label_data, training='train'):
    dataset = create_data(data_dir, images_data, label_data)
    if training == 'train':
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    elif training == 'test':
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader

