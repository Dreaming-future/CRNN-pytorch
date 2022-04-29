import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import scipy.io as sio
class FixHeightResize(object):
    """
    对图片做固定高度的缩放
    """
    def __init__(self, height=32, minwidth=100):
        self.height = height
        self.minwidth = minwidth

    # img 为 PIL.Image 对象
    def __call__(self, img):
        w, h = img.size
        width = max(int(w * self.height / h), self.minwidth)
        return img.resize((width, self.height), Image.ANTIALIAS)
    
class IIIT5k(Dataset):
    """
    用于加载IIIT-5K数据集，继承于torch.utils.data.Dataset

    Args:
        root (string): 数据集所在的目录
        training (bool, optional): 为True时加载训练集，为False时加载测试集，默认为True
        fix_width (bool, optional): 为True时将图片缩放到固定宽度，为False时宽度不固定，默认为False
    """
    def  __init__(self, root, training=True, fix_width=False):
        super(IIIT5k,self).__init__()
        data_str = 'traindata' if training else 'testdata'
        self.img, self.label = zip(*[(x[0][0],x[1][0]) for  x in 
                                    sio.loadmat(os.path.join(root, data_str + '.mat'))[data_str][0]])
        
        # 图片缩放 + 转化为灰度图 + 转化为张量
        transform = [transforms.Resize((32, 100), Image.ANTIALIAS)
                     if fix_width else FixHeightResize(32)]
        transform.extend([transforms.Grayscale(), transforms.ToTensor()])
        transform = transforms.Compose(transform)

        # 加载图片
        self.img = [transform(Image.open(root+'/'+img)) for img in self.img]
    
    
    # 以下两个方法必须要重载
    def __len__(self, ):
        return len(self.img)

    def __getitem__(self, idx):
        return self.img[idx], self.label[idx]

# 为了加快以后加载数据的过程，可以将我们的IIIT-5k实例存入.pkl文件。
# 这样以后加载数据时，省内存，加载更是一秒加载完。
import pickle
from torch.utils.data import DataLoader


def load_data(root, training=True, fix_width=False):
    """
    用于加载IIIT-5K数据集，继承于torch.utils.data.Dataset

    Args:
        root (string): 数据集所在的目录
        training (bool, optional): 为True时加载训练集，为False时加载测试集，默认为True
        fix_width (bool, optional): 为True时将图片缩放到固定宽度，为False时宽度不固定，默认为False

    Return:
        加载的训练集或者测试集
    """
    if training:
        batch_size = 128 if fix_width else 1
        filename = os.path.join(root, 'train' + '_fix_width' if fix_width else '') + '.pkl'
        if os.path.exists(filename):
            dataset = pickle.load(open(filename,'rb'))
        else:
            print('==== Loading data.. ====')
            dataset = IIIT5k(root, training=True, fix_width=fix_width)
            pickle.dump(dataset, open(filename, 'wb'), True)
        dataloader = DataLoader(dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=4)
    else:
        batch_size = 128 if fix_width else 1
        filename = os.path.join(root, 'test'+('_fix_width' if fix_width else '')+'.pkl')
        if os.path.exists(filename):
            dataset = pickle.load(open(filename, 'rb'))
        else:
            print('==== Loading data.. ====')
            dataset = IIIT5k(root, training=False, fix_width=fix_width)
            pickle.dump(dataset, open(filename, 'wb'), True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader
    
    
    