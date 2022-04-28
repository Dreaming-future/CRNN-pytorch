# -*- coding: utf-8 -*-
"""
author:LTH
data:
"""
import os
import random

import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import GetConfig
from datalist import RecTextLineDataset, rec_collate
from metric import RecMetric
from model import CRNN, crnn
from recognition.utils import CTCLabelConverter

best_acc = 0


class CarPlateRec(object):
    def __init__(self):
        self.args = GetConfig()

        print(f"-----------{self.args.project_name}-------------")

        use_cuda = self.args.use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        kwargs = {"num_workers": 0, "pin_memory": True} if use_cuda else {"num_workers": 0, "pin_memory": False}

        '''
        构造DataLoader
        '''
        self.data = self.get_data(self.args.base_dir)
        random.shuffle(self.data)
        self.num_val = int(len(self.data) * self.args.val_per)
        self.num_train = len(self.data) - self.num_val

        self.train_dataloader = DataLoader(RecTextLineDataset(self.data[:self.num_train],type="train"),
                                           batch_size=self.args.train_batch_size,
                                           shuffle=False,
                                           collate_fn=rec_collate,
                                           **kwargs)
        self.test_dataloader = DataLoader(RecTextLineDataset(self.data[self.num_train:],type="test"),
                                          batch_size=self.args.test_batch_size,
                                          shuffle=False,
                                          collate_fn=rec_collate,
                                          **kwargs)
        '''
        定义模型
        '''
        self.model = CRNN().to(self.device)

        '''
        CUDA加速
        '''
        if use_cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))
            cudnn.enabled = True
            cudnn.benchmark = True


        if self.args.resume:
            try:
                print("load the weight from pretrained-weight file")
                model_dict = self.model.state_dict()
                checkpoint = torch.load(self.args.pretrained_weight)['model_state_dict']
                model_dict.update(checkpoint)
                self.model.load_state_dict(model_dict, strict=True)
                print("Restoring the weight from pretrained-weight file \nFinished loading the weight")
            except Exception as e:
                print("can not load weight \n train the model from scratch")
                raise e

        '''
        构造loss目标函数
        选择优化器
        学习率变化选择
        '''
        self.loss = torch.nn.CTCLoss(reduction="mean").to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.args.lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.args.milestone,
                                                              gamma=0.5)  # 这里是模型训练的关键之处，调节的好训练的块

        self.convert = CTCLabelConverter(character=self.args.alphabet)
        self.metric = RecMetric(self.convert)

        try:
            for epoch in range(self.args.epochs):
                self.train(epoch)
                self.test(epoch)
        except:
            torch.cuda.empty_cache()
            print("Finish model training")

        finally:
            torch.save({
                "model_state_dict": self.model.state_dict()
            },
                self.args.save_path
            )
            print("model saved")
            with open("train_log.txt", 'a') as f:
                f.write("############################################################\n")
                import time
                f.write(str(time.asctime( time.localtime(time.time()) )))
                f.write('\n')

    @staticmethod
    def get_data(base_dir):
        data = []
        for root, dirs, files in os.walk(base_dir):
            for f in files:
                data.append(os.path.join(root, f))
        return data

    def train(self, epoch):

        self.model.train()

        correct = 0
        total = 0
        average_loss = 0

        pbar = tqdm(self.train_dataloader)

        for image, target, length, label_words in pbar:
            image = image.to(self.device)
            self.optimizer.zero_grad()
            input_length, target_length = self.sparse_tuple_for_ctc(self.args.T_length, length)
            output = self.model(image)
            # TNC
            output_prob = output.permute(1, 0, 2)

            output_prob = output_prob.log_softmax(2)
            loss = self.loss(output_prob,
                             target,
                             input_lengths=input_length,
                             target_lengths=target_length)
            loss.backward()
            self.optimizer.step()
            average_loss += loss.item()
            # NTC
            result = self.metric(output_prob.permute(1, 0, 2), label_words)

            total += len(output)
            correct += result['n_correct']
            acc = correct / total

            pbar.set_description(f"Train epoch:{epoch} "
                                 f"\tloss:{round(average_loss / total, 6)} "
                                 f"\tacc: {acc} "
                                 f"\tlr:{self.optimizer.param_groups[0]['lr']} ")
        self.scheduler.step()

    @torch.no_grad()
    def test(self, epoch):
        self.model.eval()
        global best_acc
        acc = 0
        correct = 0
        total = 0
        pbar = tqdm(self.test_dataloader)
        for image, target, check_target, label_words in pbar:
            output = self.model(image.to(self.device))
            output_prob = output.permute(1, 0, 2)
            output_prob = output_prob.log_softmax(2)

            result = self.metric(output_prob.permute(1, 0, 2), label_words)

            correct += result['n_correct']
            total += len(output)
            acc = correct / total

            pbar.set_description(f"Test epoch:{epoch}  "
                                 f"\tacc: {acc}"
                                 f'\tbest acc: {best_acc}')

        if best_acc < acc:
            best_acc = acc
            torch.save({
                "model_state_dict": self.model.state_dict()
            },
                self.args.save_path
            )
            print("model saved")

        with open("train_log.txt", 'a') as f:
            f.write(str(epoch) + "-->" + str(best_acc))
            f.write('\n')

    @staticmethod
    def sparse_tuple_for_ctc(t_length, lengths):
        input_lengths = []
        target_lengths = []

        for ch in lengths:
            input_lengths.append(t_length)
            target_lengths.append(ch)
        return tuple(input_lengths), tuple(target_lengths)


if __name__ == "__main__":
    train = CarPlateRec()
