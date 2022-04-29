import argparse
from crnn import CRNN
import os
import torch

from utils_fit import test, train
def main(epoch_num, lr=0.1, training=True, fix_width=True):
    """
    Main

    Args:
        training (bool, optional): If True, train the model, otherwise test it (default: True)
        fix_width (bool, optional): Scale images to fixed size (default: True)
    """

    model_path = ('fix_width_' if fix_width else '') + 'crnn.pth'
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    root = 'data/IIIT5K/'
    if training:
        net = CRNN(1, len(letters) + 1)
        start_epoch = 0
        # if there is pre-trained model, load it
        if os.path.exists(model_path):
            print('Pre-trained model detected.\nLoading model...')
            net.load_state_dict(torch.load(model_path))
        if torch.cuda.is_available():
            print('GPU detected.')
        net = train(root, start_epoch, epoch_num, letters,
                    net=net, lr=lr, fix_width=fix_width)
        # save the trained model for training again
        torch.save(net.state_dict(), model_path)
        # test
        test(root, net, letters, fix_width=fix_width)
    else:
        net = CRNN(1, len(letters) + 1)
        if os.path.exists(model_path):
            net.load_state_dict(torch.load(model_path))
        test(root, net, letters, fix_width=fix_width)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch_num', type=int, default=50, help='number of epochs to train for (default=20)')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate for optim (default=0.1)')
    parser.add_argument('--test', action='store_true', help='Whether to test directly (default is training)')
    parser.add_argument('--fix_width', action='store_true', help='Whether to resize images to the fixed width (default is False)')
    opt = parser.parse_args()
    print(opt)
    main(opt.epoch_num, lr=opt.lr, training=(not opt.test), fix_width=opt.fix_width)