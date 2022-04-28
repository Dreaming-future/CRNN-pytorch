# -*- coding: utf-8 -*-
"""
author:LTH
data:
"""
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="ocr rec test")
    parser.add_argument('--project_name', type=str, default="CRNN 车牌文字识别")
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--resume', type=bool, default=True)
    parser.add_argument('--pretrained_weight', type=str, default='./weight/best.pth')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--milestones', type=list, default=[10, 30, 50, 80])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save_path', type=str, default="./weights/")

    parser.add_argument("--T_length", type=int, default=128)  # 模型的输出序列长度，这个数字由datalist中的图片宽度决定，不同的设定 T_length也是不同的
    parser.add_argument('--alphabet', type=str, default="./config_data/plate.txt")
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=128)

    args = parser.parse_args()
    return args


class GetConfig(object):
    def __init__(self):
        self.project_name = "CRNN 车牌文字识别"
        self.use_cuda = True
        self.seed = 123
        self.resume = True
        self.pretrained_weight = "None"
        self.lr = 0.0001
        self.milestone = [10, 30, 50, 80]
        self.epochs = 200
        self.save_path = "./weight/best.pth"
        self.T_length = 128
        self.alphabet = "./config_data/plate.txt"
        self.train_batch_size = 64
        self.test_batch_size = 64

