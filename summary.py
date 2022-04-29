import torch
from torchinfo import summary
from crnn import CRNN

if __name__ == "__main__":
    # 需要使用device来指定网络在GPU还是CPU运行
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    crnn = CRNN(3,10).to(device)
    summary(crnn,(1,3,32,128))