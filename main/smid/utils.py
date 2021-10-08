import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torchmetrics

sns.set_style("whitegrid")


def accuracy(y_pred, y_gt, cm):
    y_pred_tag = torch.argmax(y_pred, dim=-1)
    for i in range(len(cm)):
        for j in range(len(cm)):
            cm[i][j] += (y_pred_tag[y_gt == i] == j).sum().int().item()
    #cm[0][1] += (y_pred_tag[y_gt == 0] == 1).sum().int().item()
    #cm[0][0] += (y_pred_tag[y_gt == 0] == 0).sum().int().item()
    #cm[1][1] += (y_pred_tag[y_gt == 1] == 1).sum().int().item()
    #cm[1][0] += (y_pred_tag[y_gt == 1] == 0).sum().int().item()
    correct_results_sum = (y_pred_tag == y_gt).sum().float()
    acc = correct_results_sum / y_gt.shape[0]
    acc = acc * 100
    return acc


def precision_recall(pred, target, pos_label=1):
    assert not target.max() > 1
    if len(pred.shape) > 1:
        pred = torch.argmax(pred, dim=-1)
    pr_curve = torchmetrics.PrecisionRecallCurve(pos_label=pos_label)
    average_precision = torchmetrics.AveragePrecision(pos_label=pos_label)
    precision, recall, thresholds = pr_curve(pred, target)
    avg_precision = average_precision(pred, target)
    return precision, recall, thresholds, avg_precision


def f1(pred, target):
    assert not target.max() > 1
    if len(pred.shape) > 1:
        pred = torch.argmax(pred, dim=-1)
    f1 = torchmetrics.F1(num_classes=2)
    f1_score = f1(pred, target)
    return f1_score


def display_plt(save, file_name):
    if save:
        plt.savefig(file_name, dpi=600)
    else:
        plt.show()
    plt.close()


def cm_by_bins(y_pred, y_gt, y_means, bin_width=0.1, figsize=(12, 6), save=False, save_path=None):
    """
    y_pred: 1 is bad, 0 is good
    """
    y_pred_target = np.argmax(y_pred, axis=-1)
    acc_t_gt = y_pred_target[np.array(y_means) > 3.5] == np.array(y_gt)[np.array(y_means) > 3.5]
    acc_t_gt = np.mean(acc_t_gt)

    acc_t_lt = y_pred_target[np.array(y_means) < 2.5] == np.array(y_gt)[np.array(y_means) < 2.5]
    acc_t_lt = np.mean(acc_t_lt)

    bar_color = sns.color_palette('deep')[2]
    bar_color_edge = sns.color_palette('deep')[-3]
    bar_color_alpha = 1.
    bins = np.arange(1, 5. + bin_width, bin_width)
    # print(bins)
    bins_dict = dict.fromkeys([f"{b:.1f}" for b in list(bins)])
    for key in list(bins_dict.keys()):
        bins_dict[key] = {'cnt': 0, 'conf': 0.0, 'acc': 0, 'err': 0}
    y_pred_t = [y_pred[idx][e] for idx, e in enumerate(y_pred_target)]
    err = np.abs(1 - np.array(y_pred_t))
    conf = np.abs(np.array(y_pred_t))

    relative_bin_width = 0.8
    plt.figure(figsize=figsize)
    n, _, patches = plt.hist(y_means, bins=bins, density=True, facecolor=bar_color,
                             edgecolor=bar_color_edge, alpha=bar_color_alpha, rwidth=relative_bin_width)

    plt.title('Histogram of data distribution')
    plt.grid(False)
    display_plt(save, os.path.join(save_path, 'data_distribution.png'))


    plt.figure(figsize=figsize)
    n, _, patches = plt.hist(y_pred_t, bins=len(bins), density=True, facecolor=bar_color,
                             edgecolor=bar_color_edge, alpha=bar_color_alpha, rwidth=relative_bin_width)

    plt.title('Histogram of pred distribution')
    plt.grid(False)
    display_plt(save, os.path.join(save_path, 'pred_distribution.png'))

    for i in range(len(y_pred)):
        key = round(y_means[i], 1)
        if not (round((key - 1) * 100)) % round((bin_width * 100)) == 0:
            # print('test', key)
            diff = ((key - 1) * 100) % (bin_width * 100)
            key = round(key + bin_width - (diff / 100), 1)

        bins_dict[str(key)]['cnt'] += 1
        bins_dict[str(key)]['acc'] += np.argmax(y_pred[i]) == y_gt[i]
        bins_dict[str(key)]['err'] += err[i]
        bins_dict[str(key)]['conf'] += conf[i]

    for key in list(bins_dict.keys()):
        bins_dict[key]['inv_acc'] = bins_dict[key]['acc']
        if bins_dict[key]['cnt'] == 0:
            bins_dict[key]['acc'] = 0
            bins_dict[key]['inv_acc'] = 1
            bins_dict[key]['cnt'] = 1

    plt_bins = [f"{b:.2f}" for b in bins]
    data_ = [bins_dict[f"{b:.1f}"]['conf'] / bins_dict[f"{b:.1f}"]['cnt'] for b in bins]
    plt.figure(figsize=figsize)
    plt.title('Histogram of confidence')
    plt.bar(bins, data_, width=bin_width * relative_bin_width,
            alpha=bar_color_alpha, color=bar_color, edgecolor=bar_color_edge, linewidth=1.)
    plt.xticks(bins, plt_bins, rotation='vertical')
    plt.grid(False)
    display_plt(save, os.path.join(save_path, 'conf_hist.png'))

    data_ = [bins_dict[f"{b:.1f}"]['acc'] / bins_dict[f"{b:.1f}"]['cnt'] for b in bins]
    plt.figure(figsize=figsize)
    plt.title('Histogram of accuracy')
    plt.bar(bins, data_, width=bin_width * relative_bin_width,
            alpha=bar_color_alpha, color=bar_color, edgecolor=bar_color_edge, linewidth=1.)
    plt.xticks(bins, plt_bins, rotation='vertical')
    plt.grid(False)
    display_plt(save, os.path.join(save_path, 'acc_hist.png'))

    """
    acc_t_gt = 0
    acc_t_lt = 0
    cnt_gt = 0
    cnt_lt = 0
    for idx, b in enumerate(bins):
        if b < 2.5:
            cnt_lt += 1
            acc_t_lt += data_[idx]
        elif b > 3.5:
            cnt_gt += 1
            acc_t_gt += data_[idx]

    acc_t_gt /= cnt_gt
    acc_t_lt /= cnt_lt
    """

    data_ = [(1 - bins_dict[f"{b:.1f}"]['inv_acc'] / bins_dict[f"{b:.1f}"]['cnt']) for b in bins]
    plt.figure(figsize=figsize)
    plt.title('Histogram of error')
    plt.ylim(0, 0.4)
    plt.yticks(np.arange(0, 0.41, step=0.1))

    plt.bar(bins, data_, width=bin_width * relative_bin_width,
            alpha=bar_color_alpha, color=bar_color, edgecolor=bar_color_edge, linewidth=1.)
    plt.xticks(bins, plt_bins, rotation='vertical')
    plt.grid(False)
    display_plt(save, os.path.join(save_path, 'err_hist.png'))

    return acc_t_gt, acc_t_lt


"""
def cm_by_thresholds(y_pred, y_gt):
    thresholds = [(1.5, 4.5), (2.0, 4.0), (2.5, 3.5), (3.0, 2.9999)]
    for low_t, high_t in thresholds:
        acc, cnt = 0., 0
        cm_dict[f'{low_t} {high_t}'] = {0: {'cnt': 0, 'correct': 0},
                                        1: {'cnt': 0, 'correct': 0},
                                        'removed': 0,
                                        'acc': 0}
        for elem in data:
            pred_label_idx = int(elem[3])
            means = [float(elem[5]), float(elem[4])]
            mean = means[moral_or_valence]
            if low_t <= mean < high_t:
                continue
            if mean < low_t:
                if pred_label_idx == 1:
                    acc += 1
                    cm_dict[f'{low_t} {high_t}'][1]['correct'] += 1
                cm_dict[f'{low_t} {high_t}'][1]['cnt'] += 1
            elif mean > high_t:
                if pred_label_idx == 0:
                    acc += 1
                    cm_dict[f'{low_t} {high_t}'][0]['correct'] += 1
                cm_dict[f'{low_t} {high_t}'][0]['cnt'] += 1
            cnt += 1

        print(f'\nResults {low_t} {high_t}')
        print(f'Removed data: {len(data) - cnt}/{len(data)}')
        cm_dict[f'{low_t} {high_t}']['removed'] = len(data) - cnt
        cm_dict[f'{low_t} {high_t}']['acc'] = acc * 100
        acc /= cnt

        print(f'Accuracy by moral: {acc * 100:.2f}')

    print('low high    | non toxic    | toxic')
    for d_key in list(cm_dict.keys()):
        print(
            f"{d_key}     | {cm_dict[d_key][0]['correct'] / cm_dict[d_key][0]['cnt'] * 100:.0f}%          | {cm_dict[d_key][1]['correct'] / cm_dict[d_key][1]['cnt'] * 100:.0f}%")
"""


def test_cm_by_bins():
    y_pred = []
    y_gt = []

    y_pred.append(np.array([1.1] * 3) + [.1, .0, .4] * 3)
    y_gt.append(np.array([1.1]))

    y_pred.append(np.array([2.1] * 3) + [.1, .0, .2] * 3)
    y_gt.append(np.array([2.1]))


if __name__ == '__main__':
    cm_by_bins()
