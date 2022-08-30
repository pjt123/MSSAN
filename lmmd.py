# import torch
# import torch.nn as nn
# from torch.autograd import Variable
import numpy as np
import paddle
import paddle.nn as nn




def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.shape[0])+int(target.shape[0])
    total = paddle.concat([source, target], axis=0)
    # print(total.shape)
    total0 = paddle.expand(paddle.unsqueeze(total, axis=0), [int(total.shape[0]), int(total.shape[0]), int(total.shape[1])])
    total1 = paddle.expand(paddle.unsqueeze(total, axis=1), [int(total.shape[0]), int(total.shape[0]), int(total.shape[1])])
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = paddle.sum(L2_distance) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [paddle.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)      #/len(kernel_val)

def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.shape[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = paddle.mean(XX + YY - XY -YX)
    return loss

def lmmd(source, target, source_label, target_logits,kernel_mul=2.0, kernel_num=5, fix_sigma=None,
         gamma=1.0, max_iter=1000):

    batch_size = source.shape[0]
    weight_ss, weight_tt, weight_st = cal_weight(source_label, target_logits)
    weight_ss = paddle.to_tensor(weight_ss).cuda()  # B, B
    weight_tt = paddle.to_tensor(weight_tt).cuda()
    weight_st = paddle.to_tensor(weight_st).cuda()

    kernels = guassian_kernel(source, target,
                                   kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = paddle.to_tensor([0])
    if paddle.sum(paddle.isnan(sum(kernels))):
        return loss
    SS = kernels[:batch_size, :batch_size]
    TT = kernels[batch_size:, batch_size:]
    ST = kernels[:batch_size, batch_size:]

    loss += paddle.sum(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST)
    # Dynamic weighting
    curr_iter = 0
    curr_iter = min(curr_iter + 1, max_iter)
    p = curr_iter / max_iter
    lamb = 2. / (1. + np.exp(-gamma * p)) - 1
    loss = loss * lamb
    return loss

def cal_weight(source_label, target_logits):
    num_class = 7
    batch_size = source_label.shape[0]
    source_label = source_label.cpu().numpy()
    # print(source_label)
    source_label_onehot = np.eye(num_class)[source_label]  # one hot

    source_label_sum = np.sum(source_label_onehot, axis=0).reshape(1, num_class)
    source_label_sum[source_label_sum == 0] = 100
    source_label_onehot = source_label_onehot / source_label_sum  # label ratio

    # Pseudo label
    target_label = target_logits.cpu().max(1)[1].numpy()

    target_logits = target_logits.cpu().numpy()
    target_logits_sum = np.sum(target_logits, axis=0).reshape(1, num_class)
    target_logits_sum[target_logits_sum == 0] = 100
    target_logits = target_logits / target_logits_sum

    weight_ss = np.zeros((batch_size, batch_size))
    weight_tt = np.zeros((batch_size, batch_size))
    weight_st = np.zeros((batch_size, batch_size))

    set_s = set(source_label)
    set_t = set(target_label)
    count = 0
    for i in range(num_class):  # (B, C)
        if i in set_s and i in set_t:
            s_tvec = source_label_onehot[:, i].reshape(batch_size, -1)  # (B, 1)
            t_tvec = target_logits[:, i].reshape(batch_size, -1)  # (B, 1)

            ss = np.dot(s_tvec, s_tvec.T)  # (B, B)
            weight_ss = weight_ss + ss
            tt = np.dot(t_tvec, t_tvec.T)
            weight_tt = weight_tt + tt
            st = np.dot(s_tvec, t_tvec.T)
            weight_st = weight_st + st
            count += 1

    length = count
    if length != 0:
        weight_ss = weight_ss / length
        weight_tt = weight_tt / length
        weight_st = weight_st / length
    else:
        weight_ss = np.array([0])
        weight_tt = np.array([0])
        weight_st = np.array([0])
    return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')