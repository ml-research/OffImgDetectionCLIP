from PIL import Image
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import glob
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from torch.utils.data import SubsetRandomSampler
import seaborn as sns


def show_data_hist(y_means, name='test'):
    bin_width = 0.1
    bar_color = sns.color_palette('deep')[2]
    bar_color_edge = sns.color_palette('deep')[-3]
    bar_color_alpha = 1.
    bins = np.arange(1, 5. + bin_width, bin_width)
    # print(bins)
    bins_dict = dict.fromkeys([f"{b:.1f}" for b in list(bins)])
    for key in list(bins_dict.keys()):
        bins_dict[key] = {'cnt': 0, 'conf': 0.0, 'acc': 0, 'err': 0}

    relative_bin_width = 0.8
    plt.figure(figsize=(6, 6))
    n, _, patches = plt.hist(y_means, bins=bins, density=False, facecolor=bar_color,
                             edgecolor=bar_color_edge, alpha=bar_color_alpha, rwidth=relative_bin_width)
    plt.savefig('/workspace/results/tmp/' + name + '.png')


class SmidDataset(Dataset):
    def __init__(self, preprocess, transform=None,
                 verbose=False, test_size=None, train_size=None, n_splits=1,
                 t_low=2.5, t_high=3.5, smooth_labels=True, data_type='moral'):
        self.preprocess = preprocess
        self.transform = transform

        data_set_path = '/workspace/datasets/SMID_images_400px/'
        df = pd.read_csv(os.path.join(data_set_path, 'SMID_norms.csv'), sep=',', header=0)
        valence_means = df['valence_mean'].values
        moral_means = df['moral_mean'].values
        harm_means = df['harm_mean'].values
        data_types = {'valence': valence_means, 'moral': moral_means, 'harm': harm_means}
        self.data_type = data_type
        img_paths = []
        img_labels = []
        img_conf = []
        img_labels_means_rd = []
        img_labels_means = []

        label_weights = [0, 0]
        data_indices = []
        cnt = 0
        for idx, image_name in enumerate(tqdm(df['img_name'].values)):
            image_path = os.path.join(data_set_path, 'img', image_name)
            image_path = glob.glob(image_path + '.*')[0]
            # valence_means[idx]
            # moral_means[idx]
            if data_types[self.data_type][idx] < t_low:
                # 1-t_low
                if smooth_labels:
                    img_conf.append(max(min((1 / t_low) + 0.25, 1), 0.7))
                else:
                    img_conf.append(1.0)
                img_labels.append(1)

                img_labels_means_rd.append(int(round(data_types[self.data_type][idx])))
                img_labels_means.append(data_types[self.data_type][idx])
                label_weights[0] += 1
                data_indices.append(cnt)
            elif data_types[self.data_type][idx] > t_high:
                if smooth_labels:
                    img_conf.append(1 - max(min((1 / (6 - t_high)) + 0.25, 1), 0.7))
                else:
                    img_conf.append(1.0)
                img_labels.append(0)
                img_labels_means_rd.append(int(round(data_types[self.data_type][idx])))
                img_labels_means.append(data_types[self.data_type][idx])
                label_weights[1] += 1
                data_indices.append(cnt)
            if verbose:
                input_text = input('Press enter to show img')
                if input_text == '':
                    img = Image.open(image_path)
                    plt.imshow(img)
                    plt.title(f'Moral mean {moral_means[idx]:.3f}\nValence mean {valence_means[idx]:.3f}')
                    plt.axis('off')
                    plt.show()
                    plt.close()
            img_paths.append(image_path)
            cnt += 1

        label_weights[0] /= len(data_indices)
        label_weights[1] /= len(data_indices)
        self.label_weights = label_weights
        img_paths = np.array(img_paths)
        img_labels = np.array(img_labels)
        print('label_weights', label_weights, '#samples', np.sum(img_labels == 0), np.sum(img_labels == 1))
        img_labels_means_rd = np.array(img_labels_means_rd)
        self.img_labels_means = np.array(img_labels_means)
        data_indices = np.array(data_indices)

        # Create cv splits
        self.train_idx, self.test_idx = [], []
        self.train_size = train_size
        self.test_size = test_size
        self._split_data(n_splits, img_labels, img_labels_means_rd, test_size=test_size, train_size=train_size)
        print('     load images')
        self.imgs = [self.preprocess(Image.open(img_path)) for img_path in tqdm(img_paths)]
        self.img_labels_means_rd = img_labels_means_rd
        self.img_labels = img_labels
        self.data_indices = data_indices
        self.img_conf = img_conf

        self.train_sampler = SubsetRandomSampler(self.train_idx)
        self.test_sampler = SubsetRandomSampler(self.test_idx)

    def _split_data(self, n_splits, img_labels, img_labels_means_rd, test_size=None, train_size=None):
        print('     Create cv splits')

        # get whole dataset
        if test_size is None and train_size is None:
            print('Loading full dataset')
            self.train_idx = np.arange(len(img_labels))
            self.test_idx = np.arange(len(img_labels))
            self.indices = None  # [range(len(img_labels)) for _ in range(n_splits)]
            return
        # First create testset independent of # train samples (#)
        splits = StratifiedShuffleSplit(test_size=0.1, n_splits=n_splits, random_state=42)
        # self.train_idx, self.test_idx = next(iter(splits.split(img_labels, img_labels_means_rd)))
        self.indices = [i for i in iter(splits.split(img_labels, img_labels_means_rd))]
        indices_train_total, self.test_idx = self.indices[0]
        print('         Dataset Type:', self.data_type)
        print('         Total #samples', len(indices_train_total) + len(self.test_idx))
        print('         Total #testsamples', len(self.test_idx))
        if train_size == 1:
            self.train_idx = indices_train_total
        else:
            splits = StratifiedShuffleSplit(train_size=train_size, n_splits=1, random_state=42)
            indices_splits = [i for i in iter(splits.split(img_labels[indices_train_total],
                                                           img_labels_means_rd[indices_train_total]))]
            selected_train, _ = indices_splits[0]
            self.train_idx = indices_train_total[selected_train]

        # show_data_hist(self.img_labels_means[self.train_idx], name='train')
        # show_data_hist(self.img_labels_means[self.test_idx], name='test')
        print('         Total #trainsamples', len(self.train_idx))
        # exit()
        # indices = range(len(img_paths))

    def switch_split(self, split):
        indices_train_total, self.test_idx = self.indices[split]
        if self.train_size == 1:
            self.train_idx = indices_train_total
        else:
            splits = StratifiedShuffleSplit(train_size=self.train_size, n_splits=1, random_state=42)
            indices_splits = [i for i in iter(splits.split(self.img_labels[indices_train_total],
                                                           self.img_labels_means_rd[indices_train_total]))]
            selected_train, _ = indices_splits[0]
            self.train_idx = indices_train_total[selected_train]

        self.train_sampler = SubsetRandomSampler(self.train_idx)
        self.test_sampler = SubsetRandomSampler(self.test_idx)

    def _split_data_old(self, n_splits, img_labels, img_labels_means_rd, test_size=None, train_size=None):
        print('     Create cv splits')

        # get whole dataset
        if test_size is None and train_size is None:
            print('Loading full dataset')
            self.train_idx = np.arange(len(img_labels))
            self.test_idx = np.arange(len(img_labels))
            self.indices = None  # [range(len(img_labels)) for _ in range(n_splits)]
            return
        # First create testset independent of # train samples (#)
        splits = StratifiedShuffleSplit(test_size=0.5, n_splits=1, random_state=42)
        # self.train_idx, self.test_idx = next(iter(splits.split(img_labels, img_labels_means_rd)))
        indices_split1 = [i for i in iter(splits.split(img_labels, img_labels_means_rd))]
        indices_train_total, self.test_idx = indices_split1[0]

        print('         Total #samples', len(indices_train_total) + len(self.test_idx))
        print('         Total #testsamples', len(self.test_idx))
        if train_size == 1:
            self.train_idx = indices_train_total
            self.indices = [indices_split1[0] for _ in range(n_splits)]
        else:
            splits = StratifiedShuffleSplit(train_size=train_size, n_splits=n_splits, random_state=42)
            indices_splits = [i for i in iter(splits.split(img_labels[indices_train_total],
                                                           img_labels_means_rd[indices_train_total]))]
            self.indices = []
            for train_indices, remaining_indices in indices_splits:
                self.indices.append((indices_train_total[train_indices],
                                     indices_train_total[remaining_indices]))
            self.train_idx, _ = self.indices[0]

        # show_data_hist(self.img_labels_means[self.train_idx], name='train')
        # show_data_hist(self.img_labels_means[self.test_idx], name='test')
        print('         Total #trainsamples', len(self.train_idx))
        # exit()
        # indices = range(len(img_paths))

    def switch_split_old(self, split):
        self.train_idx, _ = self.indices[split]
        self.train_sampler = SubsetRandomSampler(self.train_idx)
        self.test_sampler = SubsetRandomSampler(self.test_idx)

    """    
    def _split_data_old(self, n_splits, img_labels, img_labels_means_rd, test_size=None, train_size=None):
        print('     Create cv splits')
        splits = StratifiedShuffleSplit(test_size=test_size, n_splits=n_splits, random_state=42)
        # self.train_idx, self.test_idx = next(iter(splits.split(img_labels, img_labels_means_rd)))
        self.indices = [i for i in iter(splits.split(img_labels, img_labels_means_rd))]
        self.train_idx, self.test_idx = self.indices[0]
        # indices = range(len(img_paths))
    """

    """
    def switch_split_old(self, split):
        self.train_idx, self.test_idx = self.indices[split]
        self.train_sampler = SubsetRandomSampler(self.train_idx)
        self.test_sampler = SubsetRandomSampler(self.test_idx)
    """

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_idx = self.data_indices[idx]
        # image = self.preprocess(Image.open(self.img_paths[idx]))
        image = self.imgs[img_idx]  # self.preprocess(Image.open(self.img_paths[idx]))
        label = self.img_labels[idx]
        means = self.img_labels_means[idx]
        conf = self.img_conf[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, means, conf

def switch_split(split: int, dataset: SmidDataset):
    dataset.switch_split(split)
    return get_dataloaders(dataset)


def setup_dataset(preprocess, transform=None, test_size=None, train_size=0.9, t_low=2.5, t_high=3.5, num_classes=2,
                  n_splits=1, data_type='moral',
                  smooth_labels=False, verbose=False):
    if num_classes == 2:
        DatasetClass = SmidDataset
    else:
        raise ValueError(f'SMID dataset with class count {num_classes} not supported')
    dataset = DatasetClass(preprocess=preprocess, transform=transform, data_type=data_type,
                           verbose=verbose, test_size=test_size, train_size=train_size, n_splits=n_splits,
                           t_low=t_low, t_high=t_high, smooth_labels=smooth_labels)

    return dataset


def get_dataloaders(dataset):
    train_dataloader = DataLoader(dataset, batch_size=64, sampler=dataset.train_sampler)
    test_dataloader = DataLoader(dataset, batch_size=64, sampler=dataset.test_sampler, drop_last=False)
    return train_dataloader, test_dataloader
