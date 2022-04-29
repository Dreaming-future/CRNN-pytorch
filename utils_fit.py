import torch
import torch.optim as optim
from dataloader import load_data
from utils import LabelTransformer, get_lr
from crnn import CRNN
from tqdm import tqdm




def train(root, start_epoch, epoch_num, letters, 
          net=None, lr=0.1, fix_width=True):
    """
    Train CRNN model

    Args:
        root (str): Root directory of dataset
        start_epoch (int): Epoch number to start
        epoch_num (int): Epoch number to train
        letters (str): Letters contained in the data
        net (CRNN, optional): CRNN model (default: None)
        lr (float, optional): Coefficient that scale delta before it is applied
            to the parameters (default: 1.0)
        fix_width (bool, optional): Scale images to fixed size (default: True)

    Returns:
        CRNN: Trained CRNN model
    """
    # Load data
    trainloader = load_data(root, training=True, fix_width=fix_width)
    if not net:
        # create a new model if net is None
        net = CRNN(1, len(letters) + 1)
    # Loss function
    criterion = torch.nn.CTCLoss()
    # Adadelta
    optimizer = optim.Adadelta(net.parameters(), lr=lr, weight_decay=1e-3)
    # use gpu or not
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    net = net.to(device)
    criterion = criterion.to(device)
    
    # get encoder and decoder
    labeltransformer = LabelTransformer(letters)

    print('====   Training..   ====')
    epoch_step  = len(trainloader)
    # .train() has any effect on Dropout and BatchNorm.
    net.train()
    for epoch in range(start_epoch, start_epoch + epoch_num):
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch+1}|{epoch_num}',postfix=dict,mininterval=0.3)
        loss_sum = 0
        for iteration,(img,label) in enumerate(trainloader):
            label,label_length = labeltransformer.encode(label)
            img = img.to(device)
            
            optimizer.zero_grad()
            outputs = net(img)
            output_length = torch.IntTensor([outputs.size(0)]*outputs.size(1))
            
            # calc loss
            loss = criterion(outputs, label, output_length, label_length)            
            
            # update
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            pbar.set_postfix(**{'loss'  : loss_sum / (iteration + 1), 
                                    'lr'    : get_lr(optimizer)})
            pbar.update(1)
        pbar.close()
    print('Finished Training')
    return net

def test(root, net, letters, fix_width=True):
    """
    Test CRNN model

    Args:
        root (str): Root directory of dataset
        letters (str): Letters contained in the data
        net (CRNN, optional): trained CRNN model
        fix_width (bool, optional): Scale images to fixed size (default: True)
    """

    # load data
    trainloader = load_data(root, training=True, fix_width=fix_width)
    testloader = load_data(root, training=False, fix_width=fix_width)
    # use gpu or not
    # use gpu or not
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    net = net.to(device)
    criterion = criterion.to(device)
    
    # get encoder and decoder
    labeltransformer = LabelTransformer(letters)

    print('====    Testing..   ====')

    # .train() has any effect on Dropout and BatchNorm.

    # .eval() has any effect on Dropout and BatchNorm.
    net.eval()
    acc = []
    for loader in (testloader, trainloader):
        loader_step = len(loader)
        pbar = tqdm(total=loader_step,desc=f'test Data',postfix=dict,mininterval=0.3)
        correct = 0
        total = 0
        for iteration, (img, origin_label) in enumerate(loader):
            img = img.to(device)

            outputs = net(img)  # length × batch × num_letters
            outputs = outputs.max(2)[1].transpose(0, 1)  # batch × length
            outputs = labeltransformer.decode(outputs.data)
            correct += sum([out == real for out,
                            real in zip(outputs, origin_label)])
            total += len(origin_label)
            pbar.set_postfix(**{'acc'  : correct / total})
            pbar.update(1)
        # calc accuracy
        acc.append(correct / total * 100)
        pbar.close()
    print('testing accuracy: ', acc[0], '%')
    print('training accuracy: ', acc[1], '%')