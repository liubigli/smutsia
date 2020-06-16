import argparse
import time
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm

from definitions import SEMANTICKITTI_PATH, SEMANTICKITTI_CONFIG
from smutsia.deep_learning.models.u_net import UNet
from smutsia.utils.torchsummary import summary
from smutsia.deep_learning.common.parser import SKParser
from smutsia.deep_learning.common.avgmeter import AverageMeter
from smutsia.deep_learning.common.metrics import dice_coeff, generalized_iou
from matplotlib import pyplot as plt

# todo: parametrize this variable
PARAM_FILE = '/home/leonardo/Dev/github/smutsia/config/ground_detection/cnn.yaml'

MODELS = {'unet': UNet}


def eval_net(net, valid_loader, device, criterion):
    """Evaluation without the densecrf with the dice coefficient"""
    losses = AverageMeter()
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(valid_loader)  # the number of batch
    tot_acc = 0
    tot_iou = 0
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for i, data in enumerate(valid_loader):
            x_val, y_val = data[0], data[1]
            x_val = x_val.to(device=device, dtype=torch.float32)
            y_val = y_val.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(x_val)
                loss = criterion(mask_pred, y_val)
                losses.update(loss.mean().item(), x_val.size(0))
            if net.n_classes > 1:
                tot_acc += F.cross_entropy(mask_pred, y_val).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                tot_acc += dice_coeff(pred, y_val).item()
                tot_iou += generalized_iou(pred, y_val).item()
            pbar.update()

    net.train()
    return tot_acc / n_val, tot_iou / n_val,  losses.avg


def train_epoch(train_loader, net, criterion, optimizer, epoch, scheduler, epochs, gpu, device):
    """
    Parameters
    ----------
    train_loader: torch.utils.data.Dataset
    net: torch.nn.Model
    criterion:
    optimizer: torch.optim.Optimizer
    epoch:
    evaluator:
    scheduler:
    report:
    show_scans: bool
    gpu: bool
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    iou = AverageMeter()
    update_ratio_meter = AverageMeter()

    # empy the cache to train
    if gpu:
        torch.cuda.empty_cache()

    net.train()

    end = time.time()
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{epochs}') as pbar:
        for i, data in enumerate(train_loader):
            ## measure data loading time
            data_time.update(time.time() - end)
            x_input = data[0]
            y_true = data[1]
            x_input = x_input.float()
            y_true = y_true.float()
            x_input = x_input.to(device=device, dtype=torch.float32)
            y_type = torch.float32 if net.n_classes == 1 else torch.long
            y_true = y_true.to(device=device, dtype=y_type)

            # compute output
            y_pred = net(x_input)
            loss = criterion(y_pred, y_true)

            # pbar.set_postfix(**{'loss (batch)': loss.item()})
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.mean()
            with torch.no_grad():
                pred = torch.sigmoid(y_pred)
                pred = (pred > 0.5).float()
                accuracy = dice_coeff(pred, y_true).item()
                iou_score = generalized_iou(pred, y_true).item()

            losses.update(loss.item(), x_input.size(0))
            acc.update(accuracy, x_input.size(0))
            iou.update(iou_score, x_input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            update_ratios = []
            for g in optimizer.param_groups:
                lr = g["lr"]
                for value in g["params"]:
                    if value.grad is not None:
                        w = np.linalg.norm(value.data.cpu().numpy().reshape((-1)))
                        update = np.linalg.norm(-max(lr, 1e-10) *
                                                value.grad.cpu().numpy().reshape((-1)))
                        update_ratios.append(update / max(w, 1e-10))
                update_ratios = np.array(update_ratios)
                update_mean = update_ratios.mean()
                update_std = update_ratios.std()
                update_ratio_meter.update(update_mean)
            print("\n")
            pbar.set_postfix(**{
                'lr': lr,
                'loss': losses.val,
                'loss.avg': losses.avg,
                'acc': acc.val,
                'acc.avg': acc.avg,
                'iou': iou.val,
                'iou.avg': iou.avg
            })
            pbar.update()
            scheduler.step()

        return acc.avg, iou.avg, losses.avg


def train(net, parser, device, epochs, lr=0.001):

    summary(net, input_size=(net.n_channels, 64, 2048), device=device)
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001,max_lr=0.01, gamma=0.99994, mode='triangular')
    criterion = nn.BCEWithLogitsLoss()
    train_acc = []
    train_iou = []
    train_loss = []
    val_acc = []
    val_iou = []
    val_loss = []
    best_val_acc = 0.0
    for epoch in range(epochs):
        acc, iou, loss = train_epoch(parser.get_train_set(), net=net, criterion=criterion, optimizer=optimizer,
                                     epoch=epoch, scheduler=scheduler, epochs=epochs, gpu=True, device=device)

        train_acc.append(acc)
        train_iou.append(iou)
        train_loss.append(loss)
        v_acc, v_iou, v_loss = eval_net(net, parser.get_valid_set(), device, criterion=criterion)
        val_acc.append(v_acc)
        val_iou.append(v_iou)
        val_loss.append(v_loss)

        if v_acc > best_val_acc:
            print("Best new model found! Saving...")
            # todo save best model
            torch.save(net.state_dict(), f'best_model.pth')

    x = np.arange(epochs)
    train_acc = np.array(train_acc)
    val_acc = np.array(val_acc)
    train_iou = np.array(train_iou)
    val_iou = np.array(val_iou)
    train_loss = np.array(train_loss)
    val_loss = np.array(val_loss)

    plt.style.use('ggplot')
    plt.figure()
    plt.plot(x, train_acc)
    plt.plot(x, val_acc)
    plt.legend(['train', 'acc'])
    plt.title("Accuracy plot")
    plt.show()

    plt.figure()
    plt.plot(x, train_iou)
    plt.plot(x, val_iou)
    plt.legend(['train', 'acc'])
    plt.title("IoU plot")
    plt.show()

    plt.figure()
    plt.plot(x, train_loss)
    plt.plot(x, val_loss)
    plt.legend(['train', 'acc'])
    plt.title("Losses")
    plt.show()

def parse_config_file(config_file):
    from smutsia.utils import load_yaml
    config = load_yaml(config_file)
    model_args = config['model']
    model_name = model_args.pop('name').lower()
    net = MODELS[model_name](**model_args)
    add_normals = config['add_normals']
    sensor = config['sensor']

    return net, add_normals, sensor



def main(config_file, data_path, sk_config_file, gpu=0):
    device = torch.device("cuda:"+str(gpu) if torch.cuda.is_available() else "cpu")
    model, add_normals, sensor = parse_config_file(config_file)
    model.to(device=device)
    train_seq = [i for i in range(8)]
    valid_seq = [i for i in range(8)]
    test_seq = None

    parser = SKParser(root=data_path,
                      train_sequences=train_seq,
                      valid_sequences=valid_seq,
                      test_sequences=test_seq,
                      add_normals=add_normals,
                      sk_config_path=sk_config_file,
                      sensor=sensor,
                      batch_size=8,
                      workers=8,
                      end=0.5,
                      step=10,
                      vstart=0.5,
                      vstep=10,
                      gt=True,
                      shuffle_train=True,
                      detect_ground=True)

    train(model, parser, device=device, epochs=10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='path to config file')
    parser.add_argument('--datapath', type=str, default=SEMANTICKITTI_PATH, help='path to dataset')
    parser.add_argument('--sk_config', type=str, default=SEMANTICKITTI_CONFIG, help='path to semantic kitti config')
    parser.add_argument('--gpu', type=int, default=0, help='Id of gpu to use')

    args = parser.parse_args()

    config_file = args.config
    datapath = args.datapath
    sk_config = args.sk_config
    gpu = args.gpu

    main(config_file, datapath, sk_config, gpu)
