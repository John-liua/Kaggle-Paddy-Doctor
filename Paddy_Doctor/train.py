# -*- coding: utf-8 -*

# -------------------------------------------------------------------------------
# Author: LiuNing
# Contact: 2742229056@qq.com
# Software: PyCharm
# File: train.py
# Time: 6/27/19 3:00 PM
# Description: 
# -------------------------------------------------------------------------------
import torch.optim as optim
from torch.optim import lr_scheduler
from shutil import copyfile
from datetime import datetime
from core.config import *
from core.model import *
from core.step_lr import StepLRScheduler
from dataload.dataloader import *
from tqdm import tqdm

init_environment()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
multi_gpus = False
model_name = 'agriculture_model1'


def main():
    save_dir = os.path.join(SAVE_DIR, model_name + '_' +
                            datetime.now().strftime('%Y%m%d_%H%M%S'))
    if os.path.exists(save_dir):
        raise NameError('model dir exists!')
    os.makedirs(save_dir)
    copyfile('./train.py', save_dir + '/train.py')
    copyfile('./core/model.py', save_dir + '/model.py')
    copyfile('./core/config.py', save_dir + '/config.py')
    logging = init_log(save_dir)
    _print = logging.info

    train_iter, test_iter = load_data(
        root='./database',
        train_paths=['./datalist/train.txt'],
        test_paths=['./datalist/val.txt'],
        signal=' ',
        resize_size=RESIZE_SIZE,
        input_size=INPUT_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=2
    )

    net = agriculture_model1(classes=10)
    ignore_params = list(map(id, net.fc7.parameters()))
    ignore_params += list(map(id, net.cls.parameters()))
    base_params = filter(lambda p: id(p) not in ignore_params, net.parameters())
    extra_params = filter(lambda p: id(p) in ignore_params, net.parameters())

    optimizer = optim.SGD(
        [{'params': base_params, 'lr': 0.001},
         {'params': extra_params, 'lr': 0.01}],
        weight_decay=1e-4, momentum=0.9, nesterov=True
    )

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    if RESUME:
        ckpt = torch.load(RESUME)
        net.load_state_dict(ckpt['net_state_dict'])

    net = net.cpu()
    if multi_gpus:
        net = nn.DataParallel(net)

    losses = AverageMeter()
    train_acc = AverageMeter()
    train_acc2 = AverageMeter()
    train_acc3 = AverageMeter()
    test_acc = AverageMeter()
    test_acc2 = AverageMeter()
    test_acc3 = AverageMeter()

    max_test_acc = 0.0

    for epoch in range(TOTAL_EPOCH):

        # train
        flag = False
        net.train()
        exp_lr_scheduler.step(epoch)
        losses.reset()
        train_acc.reset()
        train_acc2.reset()
        train_acc3.reset()
        test_acc.reset()

        for data in tqdm(train_iter, desc='Train Epoch: {}'.format(epoch + 1)):
            inputs, labels = data

            if inputs.size(0) == 1:
                continue

            inputs = inputs.cuda()
            labels = labels.long().cuda()
            b_size = labels.size(0)

            optimizer.zero_grad()

            logits = net(inputs, labels)
            if multi_gpus:
                loss = net.module.get_loss(logits, labels)
            else:
                loss = net.get_loss(logits, labels)

            loss.backward()
            optimizer.step()

            acc = accuracy(logits[0].data, labels, topk=(1,))
            losses.update(loss.item(), b_size)
            train_acc.update(acc[0], b_size)

        _print('Train Epoch: {}\t'
               'Loss: {loss.avg:.4f}\t'
               'TrainAcc: Prec@1 {train_acc.avg:.3f}%\tPrec@2 {train_acc2.avg:.3f}%\tPrec@3 {train_acc3.avg:.3f}%'.format(
            epoch + 1, loss=losses, train_acc=train_acc, train_acc2=train_acc2, train_acc3=train_acc3
        ))

        # val
        if (epoch + 1) % TEST_FREQ == 0:
            net.eval()
            test_acc.reset()
            test_acc2.reset()
            test_acc3.reset()

            for data in tqdm(test_iter, desc='Train Epoch: {}'.format(epoch + 1)):
                with torch.no_grad():
                    inputs, labels = data
                    inputs = inputs.cuda()
                    labels = labels.long().cuda()
                    b_size = labels.size(0)

                    if multi_gpus == True:
                        logits = net.module.features(inputs)
                    else:
                        logits = net.features(inputs)

                    acc = accuracy(logits[0].data, labels, topk=(1,))
                    test_acc.update(acc[0], b_size)
            _print('Train Epoch: {}\t'
                   'TestAcc: Prec@1 {test_acc.avg:.3f}%\tPrec@2 {test_acc2.avg:.3f}%\tPrec@2 {test_acc3.avg:.3f}%'.format(
                epoch + 1, test_acc=test_acc, test_acc2=test_acc2, test_acc3=test_acc3
            ))

            if max_test_acc <= test_acc.avg:
                max_test_acc = test_acc.avg
                flag = True

        # save
        if flag == True:
            msg = 'Saving checkpoint: {}'.format(epoch + 1)
            _print(msg)
            if multi_gpus:
                net_state_dict = net.module.state_dict()
            else:
                net_state_dict = net.state_dict()

            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            torch.save(
                {'epoch': epoch,
                 'net_state_dict': net_state_dict},
                os.path.join(save_dir, 'model.ckpt')
            )

    _print('finish')


if __name__ == '__main__':
    main()