import torch
from tqdm import tqdm
from main.smid.utils import accuracy, cm_by_bins
import os
import pickle
import time
import numpy as np
from main.smid.dataset import setup_dataset, switch_split
from rtpt import RTPT


def cross_validation(tag, setup_model,
                     language_model='Clip_ViT-B/32',
                     n_splits=2,
                     num_classes=2,
                     smooth_labels=False,
                     test_size=None,
                     train_size=0.9,
                     t_low=2.5, t_high=3.5,
                     lr=0.01, epochs=500,
                     data_type='moral'):
    torch.random.manual_seed(42)
    pos_label = 0
    if data_type == 'harm':
        pos_label = 1
    model = setup_model(language_model, num_classes=num_classes, pos_label=pos_label)

    print('Loading dataset')
    dataset = setup_dataset(model.preprocess, test_size=test_size, train_size=train_size, n_splits=n_splits,
                            t_low=t_low, t_high=t_high, num_classes=num_classes,
                            smooth_labels=smooth_labels,
                            verbose=False, data_type=data_type)

    experiment_id = time.time()
    res = []
    torch.cuda.is_available()
    save_dir = f'/workspace/results/offendingCLIP/SMID/{data_type}/'
    smooth_label_tag = ''# TODO fix to: '_SmoothLabels_' if smooth_labels else ''
    num_classes_tag = f'_NumClasses{num_classes}_' if num_classes == 3 else ''

    #save_path = f'{save_dir}{language_model.replace("/32", "")}/{tag}/cv_{int(n_splits)}/{test_size:.2f}_{t_low:.2f}_{t_high:.2f}{smooth_label_tag}{num_classes_tag}/{experiment_id}'
    save_path = f'{save_dir}{language_model.replace("/32", "-32").replace("/16", "-16")}/{tag}/cv_{int(n_splits)}/{train_size:.2f}_{t_low:.2f}_{t_high:.2f}{smooth_label_tag}{num_classes_tag}/{experiment_id}'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("Starting CV")
    for split in range(n_splits):
        print('--' * 42)
        print(f'\n{split}# split')
        save_path_split = os.path.join(save_path, f'{split}')
        os.makedirs(os.path.join(save_path_split, 'init'), exist_ok=False)
        os.makedirs(os.path.join(save_path_split, 'tuned'), exist_ok=False)

        model = setup_model(language_model, num_classes=num_classes)
        train_dataloader, test_dataloader = switch_split(split, dataset)

        y_pred_bt, y_gt_bt, means_bt, epoch_acc_bt, \
        epoch_cmt_bt, acc_t_gt_bt, acc_t_lt_bt = test(test_dataloader,
                                                      model,
                                                      smooth_labels=smooth_labels,
                                                      bin_width=0.25,
                                                      save_path=os.path.join(save_path_split, 'init'),
                                                      save=True)

        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        y_pred, y_gt, means, epoch_acc, epoch_cmt = train(train_dataloader, test_dataloader,
                                                          model, optimizer, smooth_labels=smooth_labels,
                                                          epochs=epochs,
                                                          save_path=os.path.join(save_path_split, 'init'),
                                                          bin_width=0.25, verbose=False, print_interval=epochs // 10)

        y_pred, y_gt, means, epoch_acc, \
        epoch_cmt, acc_t_gt, acc_t_lt = test(test_dataloader, model,
                                             smooth_labels=smooth_labels,
                                             bin_width=0.25,
                                             save_path=os.path.join(save_path_split, 'tuned'),
                                             save=True)

        res.append((split, epoch_acc, acc_t_gt, acc_t_lt, epoch_acc_bt, acc_t_gt_bt, acc_t_lt_bt))

        if hasattr(model, 'prompts'):
            prompts = model.prompts.data.cpu().detach().numpy()
            pickle.dump(prompts, open(os.path.join(save_path_split, 'prompts.p'), 'wb'))
        pickle.dump({
            'y_pred': y_pred,
            'y_gt': y_gt,
            'y_pred_zero': y_pred_bt,
            'y_gt_zero': y_gt_bt,
            'epoch_cmt_zero': epoch_cmt_bt,
            'means': means,
            'epoch_acc': epoch_acc,
            'epoch_cmt': epoch_cmt,
            'acc_t_gt': acc_t_gt,
            'acc_t_lt': acc_t_lt,
            'test_samples': train_dataloader.dataset.test_idx,
            'train_samples': train_dataloader.dataset.train_idx,
        }, open(os.path.join(save_path_split, 'results.p'), 'wb'))

    np.savetxt(os.path.join(save_path, 'prediction.csv'), res, delimiter=',',
               header='split,total_acc,acc_t_gt,acc_t_lt,init_total_acc,init_acc_t_gt,init_acc_t_lt',
               fmt=('%d,%2f,%2f,%2f,%2f,%2f,%2f'))


def eval_loop(model, dataloader, device, criterion):
    epoch_loss = 0
    epoch_acc = 0
    #epoch_cmt = [[0, 0], [0, 0]]
    if hasattr(model, 'num_classes'):
        epoch_cmt = [[0]*model.num_classes for _ in range(model.num_classes)]
    else:
        epoch_cmt = [[0, 0], [0, 0]]
    model.eval()
    y_pred = list()
    y_gt = list()
    means = list()
    with torch.no_grad():
        for X_batch, y_batch, means_batch, conf_batch in tqdm(dataloader):
            X_batch, y_batch, conf_batch = X_batch.to(device), y_batch.to(device), conf_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch, conf_batch)
            y_pred_batch = logits.softmax(dim=-1)
            y_pred += y_pred_batch.detach().cpu().tolist()
            y_gt += y_batch.detach().cpu().tolist()
            means += means_batch.detach().cpu().tolist()

            acc = accuracy(y_pred_batch, y_batch, epoch_cmt)
            # print(loss)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            #
        print(
            f'Test: | Loss: {epoch_loss / len(dataloader):.5f} | Acc: {epoch_acc / len(dataloader):.3f}')
        print(epoch_cmt)
    return y_pred, y_gt, means, epoch_acc / len(dataloader), epoch_cmt


def train_loop(model, dataloader, epochs, optimizer, criterion, device, print_interval=1):
    model.train()
    rtpt = RTPT(name_initials='PS', experiment_name='OffImg', max_iterations=epochs)
    rtpt.start()
    for e in tqdm(range(1, epochs + 1)):
        epoch_loss = 0
        epoch_acc = 0
        if hasattr(model, 'num_classes'):
            epoch_cmt = [[0] * model.num_classes for _ in range(model.num_classes)]
        else:
            epoch_cmt = [[0, 0], [0, 0]]
        for X_batch, y_batch, _, conf_batch in dataloader:
            X_batch, y_batch, conf_batch = X_batch.to(device), y_batch.to(device), conf_batch.to(device)
            optimizer.zero_grad()

            logits = model(X_batch)
            loss = criterion(logits, y_batch, conf_batch)

            loss.backward()
            y_pred = logits.softmax(dim=-1)
            acc = accuracy(y_pred, y_batch, epoch_cmt)
            optimizer.step()
            # print(loss)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        rtpt.step(subtitle=f"acc={epoch_acc / len(dataloader):.3f}, loss={epoch_loss / len(dataloader):.5f}")
        if e % print_interval == 0:
            print(
                f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(dataloader):.5f} | Acc: {epoch_acc / len(dataloader):.3f}')
            print(epoch_cmt)


class LabelSmoothingCrossEntropyLoss(torch.nn.Module):
    """NLL loss with label smoothing.
    """

    def __init__(self, weight, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        raise ValueError('Does not work')
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.label_weights = weight

    def forward(self, x, target, conf):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = (conf * nll_loss + (1.0 - conf) * smooth_loss) * self.label_weights[target]
        return loss.mean()


def train(train_dataloader, test_dataloader, model, optimizer, smooth_labels=False, epochs=5, bin_width=0.1,
          verbose=False, print_interval=1, save_path=''):
    device = f'cuda'

    criterion = get_criterion(train_dataloader, smooth_labels, device, save_path)

    print("Eval before training")
    y_pred, y_gt, means, epoch_acc, epoch_cmt = eval_loop(model, test_dataloader, device, criterion)
    if verbose:
        acc_t_gt, acc_t_lt = cm_by_bins(y_pred, y_gt, means, bin_width=bin_width)
        print(f'Accuracy bad: {acc_t_lt * 100:.2f}, Accuracy good: {acc_t_gt * 100:.2f}')

    print("\n\nTraining")
    train_loop(model, train_dataloader, epochs, optimizer, criterion, device, print_interval=print_interval)

    print("\n\nEval after training")
    y_pred, y_gt, means, epoch_acc, epoch_cmt = eval_loop(model, test_dataloader, device, criterion)
    if verbose:
        acc_t_gt, acc_t_lt = cm_by_bins(y_pred, y_gt, means, bin_width=bin_width)
        print(f'Accuracy bad: {acc_t_lt * 100:.2f}, Accuracy good: {acc_t_gt * 100:.2f}')

    return y_pred, y_gt, means, epoch_acc, epoch_cmt


def test(test_dataloader, model, smooth_labels=False, bin_width=0.1, save=False, save_path=None):
    device = f'cuda'
    criterion = get_criterion(test_dataloader, smooth_labels, device, save_path)

    print("\n\nEval")
    y_pred, y_gt, means, epoch_acc, epoch_cmt = eval_loop(model, test_dataloader, device, criterion)

    acc_t_gt, acc_t_lt = cm_by_bins(y_pred, y_gt, means, bin_width=bin_width, save=save, save_path=save_path)
    print(f'Accuracy bad: {acc_t_lt * 100:.2f}, Accuracy good: {acc_t_gt * 100:.2f}')
    return y_pred, y_gt, means, epoch_acc, epoch_cmt, acc_t_gt, acc_t_lt


def get_criterion(dataloader, smooth_labels, device, save_path):
    torch_tensor = torch.HalfTensor
    if 'tuning_imagenet' in save_path:
        torch_tensor = torch.FloatTensor
        # TODO remove this hack
    if not smooth_labels:
        criterion_ = torch.nn.CrossEntropyLoss(
            weight=torch_tensor(dataloader.dataset.label_weights).to(device))
        criterion = lambda x, y, z: criterion_(x, y)
    else:
        criterion = LabelSmoothingCrossEntropyLoss(
            weight=torch_tensor(dataloader.dataset.label_weights).to(device))
    return criterion