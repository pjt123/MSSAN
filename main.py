
from __future__ import print_function
import argparse
import paddle
# import torch
import paddle.nn.functional as F
# from paddle.autograd import Variable
from paddle.static import Variable
import os
import math
import data_loader
# import resnet as models
# import res18 as models
import network as models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import numpy as np


use_gpu = True
paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')
seed = 100
lr = [0.001, 0.01]
momentum = 0.9
log_interval = 10
l2_decay = 5e-4
iteration = 10000
# root_path = "/home/member_1/PJT/Data/CYJ/sgt_image/"
# source1_name = "D2"
# source2_name = 'D4'
# target_name = "D1"
def get_args():
    parser = argparse.ArgumentParser(description='DG')
    parser.add_argument('--anneal_iters', type=int,
                        default=500, help='Penalty anneal iters used in VREx')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--data_file', type=str, default='', help='root_dir')
    parser.add_argument('--dataset', type=str, default='CYJ')
    parser.add_argument('--data_dir', type=str, default='/home/member_1/PJT/Data/CYJ/sgt/', help='data dir')
    parser.add_argument('--source1_name', type=str, default='D1')
    parser.add_argument('--source2_name', type=str, default='D2')
    parser.add_argument('--target_name', type=str, default='D4')
    parser.add_argument('--gpu_id', type=str, nargs='?',
                        default='1', help="device id to run")
    parser.add_argument('--net', type=str, default='resnet18',
                        help="featurizer: resnet18, reanet34, resnet50, resnet101")
    parser.add_argument('--N_WORKERS', type=int, default=1)
    args = parser.parse_args()
    args.data_dir = args.data_file+args.data_dir
    os.environ['CUDA_VISIBLE_DEVICS'] = args.gpu_id
    return args


paddle.seed(seed)

args = get_args()
kwargs = {'num_workers': args.N_WORKERS} if use_gpu else {}

source1_loader = data_loader.load_training(args.data_dir, args.source1_name, args.batch_size, kwargs)    # 3228
source2_loader = data_loader.load_training(args.data_dir, args.source2_name, args.batch_size, kwargs)    # 3500
target_train_loader = data_loader.load_training(args.data_dir, args.target_name, args.batch_size, kwargs)  # 2849
target_test_loader = data_loader.load_testing(args.data_dir, args.target_name, args.batch_size, kwargs)    # 2849


def train(model):

    source1_iter = iter(source1_loader)
    source2_iter = iter(source2_loader)
    target_iter = iter(target_train_loader)
    correct = 0
    accuracy_list = []
    labels = np.array(['0', '1', '2', '3', '4', '5', '6'])
    # paddle.optimizer.lr =
    optimizer = paddle.optimizer.Momentum(
        learning_rate=lr[1],
        parameters=[
            {'params': model.sharedNet.parameters(), 'learning_rate': lr[0]},
            {'params': model.cls_fc_son1.parameters(), 'learning_rate': lr[1]},
            {'params': model.cls_fc_son2.parameters(), 'learning_rate': lr[1]},
            {'params': model.sonnet1.parameters(), 'learning_rate': lr[1]},
            {'params': model.sonnet2.parameters(), 'learning_rate': lr[1]},
        ],
        momentum=momentum, weight_decay=l2_decay)

    for i in range(1, args.iteration + 1):
        model.train()

        optimizer._param_groups[0]['lr'] = lr[0] / math.pow((1 + 10 * (i - 1) / (args.iteration)), 0.75)
        optimizer._param_groups[1]['lr'] = lr[1] / math.pow((1 + 10 * (i - 1) / (args.iteration)), 0.75)
        optimizer._param_groups[2]['lr'] = lr[1] / math.pow((1 + 10 * (i - 1) / (args.iteration)), 0.75)
        optimizer._param_groups[3]['lr'] = lr[1] / math.pow((1 + 10 * (i - 1) / (args.iteration)), 0.75)
        optimizer._param_groups[4]['lr'] = lr[1] / math.pow((1 + 10 * (i - 1) / (args.iteration)), 0.75)


# ===================================================对齐source1与target=====================================================
        try:
            source_data, source_label = source1_iter.next()
        except Exception as err:
            source1_iter = iter(source1_loader)
            source_data, source_label = source1_iter.next()
        try:
            target_data, __ = target_iter.next()
        except Exception as err:
            target_iter = iter(target_train_loader)
            target_data, __ = next(target_iter)
        # if cuda:
        #     source_data, source_label = source_data.cuda(), source_label.cuda()
        #     target_data = target_data.cuda()
        # source_data, source_label = Variable(source_data), Variable(source_label)
        # target_data = Variable(target_data)

        # print("source label:", source_label)
        cls_loss, lmmd_loss, l1_loss = model(source_data, target_data, source_label, mark=1)
        gamma = 2 / (1 + math.exp(-10 * (i) / (args.iteration))) - 1

        loss = cls_loss + gamma * (lmmd_loss + l1_loss)
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        if i % log_interval == 0:
            print(
                'Train source1 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tlmmd_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
                    i, 100. * i / args.iteration, loss.item(), cls_loss.item(), lmmd_loss.item(), l1_loss.item()))

# ===================================================对齐source2与target=====================================================
        try:
            source_data, source_label = source2_iter.next()
        except Exception as err:
            source2_iter = iter(source2_loader)
            source_data, source_label = source2_iter.next()
        try:
            target_data, __ = target_iter.next()
        except Exception as err:
            target_iter = iter(target_train_loader)
            target_data, __ = target_iter.next()
        # if cuda:
        #     source_data, source_label = source_data.cuda(), source_label.cuda()
        #     target_data = target_data.cuda()
        # source_data, source_label = Variable(source_data), Variable(source_label)
        # target_data = Variable(target_data)

        cls_loss, lmmd_loss, l1_loss = model(source_data, target_data, source_label, mark=2)
        gamma = 2 / (1 + math.exp(-10 * (i) / (args.iteration))) - 1

        loss = cls_loss + gamma * (lmmd_loss + l1_loss)
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        if i % log_interval == 0:
            print(
                'Train source2 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tlmmd_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
                    i, 100. * i / args.iteration, loss.item(), cls_loss.item(), lmmd_loss.item(), l1_loss.item()))

        if i % (log_interval * 20) == 0:
            t_correct, y_true, y_pred, accuracy = test(model)
            if t_correct > correct:
                correct = t_correct
            accuracy_list.append('{:.2f}'.format(accuracy.item()))
            plot_confusion_matrix(y_true, y_pred, classes=labels, iteration=i, normalize=False, title="Confusion Matrix")
            print(args.source1_name, args.source2_name, "to", args.target_name, "%s max correct:" % args.target_name, correct.item(), "\n", accuracy_list)


def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    correct1 = 0
    correct2 = 0
    y_true = []
    y_pred = []
    feat = []


    with paddle.no_grad():
        for data, target in target_test_loader:
            # if cuda:
            #     data, target = data.cuda(), target.cuda()
            y_true += target.tolist()

            # data, target = Variable(data), Variable(target)
            pred1, pred2 = model(data, mark=0)    # (16, 7)

            pred1 = paddle.nn.functional.softmax(pred1, axis=1)
            pred2 = paddle.nn.functional.softmax(pred2, axis=1)

            # 计算信息熵作为权重
            entropy1 = -1 * pred1 * paddle.log(pred1)
            entropy1 = paddle.sum(entropy1, axis=1, keepdim=True)
            entropy2 = -1 * pred1 * paddle.log(pred2)
            entropy2 = entropy2.sum(axis=1, keepdim=True)
            w_1 = entropy1 / (entropy1 + entropy2)
            w_2 = entropy2 / (entropy1 + entropy2)
            pred = ((1 - w_1) * pred1) + ((1 - w_2) * pred2)

            # 计算对抗损失作为权重
            # w_1 = adv_loss1 / (adv_loss1 + adv_loss2)
            # w_2 = adv_loss2 / (adv_loss1 + adv_loss2)
            # pred = w_1 * pred1 + w_2 * pred2




            test_loss += F.nll_loss(F.log_softmax(pred, axis=1), paddle.to_tensor(target, dtype='int64')).item()
            # print('target:', target)

            pred = paddle.argmax(pred, axis=1)

            y_pred += pred.tolist()

            # print('pred:', pred)
            #
            correct += paddle.equal(pred, paddle.to_tensor(paddle.clone(target), dtype='int64')).sum()
            # print(correct)
            pred = paddle.argmax(paddle.clone(pred1), axis=1)
            correct1 += paddle.equal(pred, paddle.to_tensor(paddle.clone(target), dtype='int64')).sum()
            pred = paddle.argmax(paddle.clone(pred2), axis=1)
            correct2 += paddle.equal(pred, paddle.to_tensor(paddle.clone(target), dtype='int64')).sum()

        test_loss /= len(target_test_loader.dataset)
        accuracy = 100. * correct / len(target_test_loader.dataset)
        # print(accuracy)


        print(args.target_name, '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(target_test_loader.dataset),
            accuracy.item()))
        print('\nsource1 accnum {}, source2 accnum {}'.format(correct1, correct2))
    return correct, y_true, y_pred, accuracy

# ===========================================================可视化==========================================================
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          iteration=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    loadpath = 'out/wlmmd18/L1/Confusion_Matrix/{}{}to{}/'.format(args.source1_name, args.source2_name, args.target_name)
    if not os.path.exists(loadpath):
        os.makedirs(loadpath)
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        pass
        # print('Confusion matrix, without normalization')
    # print(cm)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    ax.set_ylim(len(classes)-0.5, -0.5)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(loadpath + 'iteration_{}.png'.format(iteration))
    # plt.show()
    return ax

if __name__ == '__main__':
    model = models.MFSAN(num_classes=7)
    print(model)
    train(model)
