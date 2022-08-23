# import related torch library functions
# we use the train.py to train the model on the pascal voc dataset
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torch.utils.data as Data

# from torchsummary import summary
from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt
import pandas as pd
import copy
import os
import time


import cfg
from not_filter_dataset import MyDataset
from not_filter_dataset import img_transforms
from not_filter_dataset import val_img_transforms
from not_filter_dataset import label2image
from not_filter_dataset import inv_normalize_image
from metrics import runningScore
from loss import FocalLoss

from seg_models.modeling import deeplabv3plus_resnet101
# from seg_models._deeplab import convert_to_separable_conv
# set the device we work

base_lr = 0.01

def adjust_lr(optimizer, base_lr, itr, max_itr):
    now_lr = max(base_lr * (1 - itr/(max_itr+1)) ** 0.9, 1e-6)
    optimizer.param_groups[0]['lr'] = now_lr * 0.1
    optimizer.param_groups[1]['lr'] = now_lr
    return now_lr

os.makedirs('/storage/Segmentation/PascalVOC/deepLogs_nofilter_180_FocalLoss_cross',exist_ok=True)
writer = SummaryWriter('/storage/Segmentation/PascalVOC/deepLogs_nofilter_180_FocalLoss_cross')
def train_model(model, criterion, optimizer, train_dataloader, val_dataloader, num_epochs=60):
    since = time.time()
    stop_counter = 0
    max_itr = num_epochs * len(train_dataloader)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    #accmulation_steps = 8

    train_loss_all = []
    val_loss_all = []

    for epoch in range(num_epochs):
        running_score_train = runningScore(cfg.num_classes)
        running_score_val = runningScore(cfg.num_classes)
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        print("Epoch:{} lr is:{}".format(epoch + 1, optimizer.state_dict()['param_groups'][0]['lr']))
        print("Current stop counter is {}".format(stop_counter))
        train_loss = 0
        train_num = 0
        val_loss = 0
        val_num = 0
        # train_dice = 0
        # val_dice = 0
        model.train()
        for step, (b_x, b_y) in enumerate(train_dataloader):
            itr = epoch * len(train_dataloader) + step
            now_lr = adjust_lr(optimizer, base_lr, itr, max_itr)
            optimizer.zero_grad()
            b_x = b_x.float().to(device)
            b_y = b_y.long().to(device)

            output = model(b_x)
            output_ = F.log_softmax(output, dim=1)
            loss = criterion(output, b_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(b_y)
            train_num += len(b_y)

            pre_label = output_.max(dim=1)[1].data.cpu().numpy()
            true_label = b_y.data.cpu().numpy()
            running_score_train.update(true_label, pre_label)
            #writer.add_scalar('train_loss', loss.item(), epoch)
        metrics = running_score_train.get_scores()
        train_mIOU = metrics[0]['mIou']
        #for k, v in metrics[0].items():
        #    print(k, v)
        train_loss_all.append(train_loss / train_num)
        print('{}Train total Loss:{:.4f}, Train mean Iou:{:.4f}'.format(epoch + 1, train_loss_all[-1], train_mIOU))
        writer.add_scalar('train_loss', train_loss_all[-1], epoch)
        writer.add_scalar('train_miou', train_mIOU, epoch)
        model.eval()
        for step, (b_x, b_y) in enumerate(val_dataloader):
            b_x = b_x.float().to(device)
            b_y = b_y.long().to(device)

            output = model(b_x)
            output_ = F.log_softmax(output, dim=1)
            loss = criterion(output, b_y)
            val_loss += loss.item() * len(b_y)
            val_num += len(b_y)
            #writer.add_scalar('val_loss', loss.item(), step)

            pre_label = output_.max(dim=1)[1].data.cpu().numpy()
            true_label = b_y.data.cpu().numpy()
            running_score_val.update(true_label, pre_label)
        metrics = running_score_val.get_scores()
        valid_mIOU = metrics[0]['mIou']
        val_loss_all.append(val_loss / val_num)
        writer.add_scalar('valid_loss', val_loss_all[-1], epoch)
        writer.add_scalar('valid_miou', valid_mIOU, epoch)
        print('{}Val total Loss:{:.4f}, Valid mean Iou:{:.4f}'.format(epoch + 1, val_loss_all[-1], valid_mIOU))
        #for k, v in metrics[0].items():
        #    print(k, v)

        if val_loss_all[-1] < best_loss:
            best_loss = val_loss_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
            stop_counter = 0
        else:
            stop_counter = stop_counter + 1
        if  stop_counter == 70:
            model.load_state_dict(best_model_wts)
            return model
        time_use = time.time() - since
        print("Train and val complete in{:.0f}m {:.0f}s".format(time_use // 60, time_use % 60))
        # scheduler.step()
    model.load_state_dict(best_model_wts)
    return model

if __name__ =='__main__':
    torch.cuda.set_device(6)
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    print(device)
    high = 513
    width = 513
    train_batchsize = 20
    val_batchsize = 4
    train = MyDataset(cfg.traindata, cfg.trainlabel, cfg.train_list, high, width, img_transforms, cfg.colormap)
    val = MyDataset(cfg.valdata, cfg.vallabel, cfg.val_list, high, width, val_img_transforms, cfg.colormap)

    train_loader = Data.DataLoader(train, batch_size=train_batchsize, shuffle=True, num_workers=8, pin_memory=True)
    valid_loader = Data.DataLoader(val, batch_size=val_batchsize, shuffle=False, num_workers=8, pin_memory=True)

    for step, (b_x, b_y) in enumerate(train_loader):
        if step > 0:
            break
    print('b_x.shape in one batch:', b_x.shape)
    print('b_y.shape in one batch:', b_y.shape)
    print('the length of train loader is:', len(train))
    print('the length of val loader is:', len(val))

    b_x_numpy = b_x.data.numpy()
    b_x_numpy = b_x_numpy.transpose(0, 2, 3, 1)
    b_y_numpy = b_y.data.numpy()
    plt.figure(figsize=(10, 6))

    #for ii in range(4):
    #    plt.subplot(2, 4, ii + 1)
    #    plt.imshow(inv_normalize_image(b_x_numpy[ii]))
    #    plt.axis("off")
    #    plt.subplot(2, 4, ii + 5)
    #    plt.imshow(label2image(b_y_numpy[ii], cfg.colormap))
    #    plt.axis("off")
    #plt.subplots_adjust(wspace=0.01, hspace=0.01)
    #plt.show()
    print('Construction deeplabvsplus_resnet101......')
    model = deeplabv3plus_resnet101(cfg.num_classes,16)
    #convert_to_separable_conv(model.classifier)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.0003
    model = model.to(device)
    print('finished')
    # model = DeepLabv3_plus(3,cfg.num_classes,os=16,_print=True)
    # criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean').to(device)
    criterion = FocalLoss(alpha=1, gamma=0, ignore_index=255).to(device)
    optimizer = optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr':0.1*base_lr},
        {'params': model.classifier.parameters(), 'lr': base_lr},],
        lr = base_lr, momentum=0.9, weight_decay = 5e-4)
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    final = train_model(model, criterion, optimizer, train_loader, valid_loader, 180)
    torch.save(final, "/storage/Segmentation/PascalVOC/Models/deeplabv3plus_a180_FocalLoss_Cross.pkl")





