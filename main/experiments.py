import argparse
import torch
from rtpt import RTPT
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from torchvision.transforms import Normalize
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
import glob
import pickle
from main.models.baseline import initialize_model_imagenet
from main.models.clip import load_finetuned_model_clip
import clip

parser = argparse.ArgumentParser(description='Crazy Stuff')
parser.add_argument('--data_dir', default='./data',
                    help='Select data path')
parser.add_argument('--data_name', default='rt-polarity', type=str, choices=['rt-polarity', 'toxicity',
                                                                             'toxicity_full', 'ethics', 'restaurant'],
                    help='Select name of data set')
parser.add_argument('--num_prototypes', default=10, type=int,
                    help='Total number of prototypes')
parser.add_argument('--num_classes', default=2, type=int,
                    help='How many classes are to be classified?')
parser.add_argument('--class_weights', default=[0.5, 0.5],
                    help='Class weight for cross entropy loss')
parser.add_argument('-g', '--gpu', type=int, default=[0], nargs='+',
                    help='GPU device number(s)')
parser.add_argument('--one_shot', type=bool, default=False,
                    help='Whether to use one-shot learning or not (i.e. only a few training examples)')
parser.add_argument('--proto_size', type=int, default=1,
                    help='Define how many words should be used to define a prototype')
parser.add_argument('--language_model', type=str, default='Bert',
                    choices=['Resnet', 'Clip_ViT-B/32', 'Clip_ViT-B/16', 'Clip_RN50x4', 'Clip_RN50'],
                    help='Define which language model to use')
parser.add_argument('--avoid_spec_token', type=bool, default=False,
                    help='Whether to manually set PAD, SEP and CLS token to high value after Bert embedding computation')
parser.add_argument('--compute_emb', type=bool, default=False,
                    help='Whether to recompute (True) the embedding or just load it (False)')
parser.add_argument('--metric', type=str, default='L2',
                    help='metric')
parser.add_argument('--input_type', type=str, required=True, choices=['text', 'img'],
                    help='choose between text and image')
parser.add_argument('--explain', type=bool, default=False,
                    help='Who needs help anyway?')
parser.add_argument('--only_offending', type=bool, default=False,
                    help='Who needs help anyway?')
parser.add_argument('--model_type', type=str, default='probe', choices=['probe', 'sim', 'resnet50', 'finetuned'],
                    help='Who needs help anyway?')
parser.add_argument('--prompt_path', type=str,
                    help='Who needs help anyway?')
labels = ['non toxic', 'toxic']
labels_care = ['toxic', 'non toxic']

inv_normalize_clip = Normalize(
    mean=[-0.48145466 / 0.26862954, -0.4578275 / 0.26130258, -0.40821073 / 0.27577711],
    std=[1 / 0.26862954, 1 / 0.26130258, 1 / 0.27577711]
)

inv_normalize_resnet = Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
)

default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                 [(0, '#ffffff'),
                                                  (0.25, '#000000'),
                                                  (1, '#000000')], N=256)


class ClipSimModel(torch.nn.Module):
    def __init__(self, args, prompts=None):
        super(ClipSimModel, self).__init__()
        self.MMM, self.preprocess = clip.load(args.language_model.split('_')[1], f'cuda:{args.gpu[0]}', jit=False)
        self.MMM.to(f'cuda:{args.gpu[0]}')
        self.MMM.eval()

        labels_clip_prompt = ['positive', 'negative']
        # labels = ['unpleasant', 'pleasant']
        # labels = ['blameworthy', 'praiseworthy']
        text = clip.tokenize([f"This image is about something {labels_clip_prompt[0]}",
                              f"This image is about something {labels_clip_prompt[1]}"
                              ]).to(f'cuda:{args.gpu[0]}')
        if prompts is not None:
            self.text_features = torch.HalfTensor(prompts).to(f'cuda:{args.gpu[0]}')
            print('Using tuned prompts', self.text_features.shape)
        else:
            self.text_features = self.MMM.encode_text(text)

    def forward(self, x):
        image_features = self.MMM.encode_image(x)
        text_features_norm = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
        # Pick the top 5 most similar labels for the image
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features_norm @ text_features_norm.T)
        # values, indices = similarity[0].topk(5)
        return similarity.squeeze()


class ClipSingleSimModel(torch.nn.Module):
    def __init__(self, args, labels):
        super(ClipSingleSimModel, self).__init__()
        self.MMM, self.preprocess = clip.load(args.language_model.split('_')[1], f'cuda:{args.gpu[0]}', jit=False)
        self.MMM.to(f'cuda:{args.gpu[0]}')
        self.MMM.eval()
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        # labels = ['unpleasant', 'pleasant']
        # labels = ['blameworthy', 'praiseworthy']
        tokens = [f"This image is about something {label}" for label in labels]
        self.text = clip.tokenize(tokens).to(f'cuda:{args.gpu[0]}')
        self.text_features = self.MMM.encode_text(self.text)

    def forward(self, x):
        image_features = self.MMM.encode_image(x)
        # text_features = self.MMM.encode_text(self.text)
        text_features = self.text_features
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        # Pick the top 5 most similar labels for the image
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features_norm @ text_features_norm.T)
        # sim = self.cos(image_features, text_features)
        return similarity.squeeze()


def eval_model_(args, x, model, file_name, save_path=None, verbose=True, show=False,data_type='moral'):
    x = x.to(f'cuda:{args.gpu[0]}')

    logits = model(x)
    probs = logits.softmax(dim=-1)

    prediction_score, pred_label_idx = torch.topk(probs.float(), 1)

    pred_label_idx = pred_label_idx.squeeze_()
    if data_type == 'harm':
        predicted_label = labels_care[pred_label_idx.cpu().detach().numpy()]
    else:
        predicted_label = labels[pred_label_idx.cpu().detach().numpy()]
    #predicted_label = labels[pred_label_idx.cpu().detach().numpy()]

    if verbose:
        print(f'Predicted: {predicted_label} ({prediction_score.squeeze().item() * 100:.2f})')

    suffix = f'{prediction_score.squeeze().item() * 100:.0f}'
    save_path_sep = os.path.join(save_path, predicted_label, suffix)

    save_filename = False
    if not args.only_offending or (predicted_label == 'toxic' and prediction_score >= .90):
        save_filename = True
    return prediction_score.item(), predicted_label, pred_label_idx.cpu().detach().numpy().item(), save_filename


def run_model_smid(args):
    # 'Clip_ViT-B'
    if args.model_type == 'probe':
        files_expl = glob.glob(
            f'./experiments/train_results/toxicity/*_baseline_{args.language_model.split("/")[0]}/best_model.pth.tar')
        if len(files_expl) == 0: raise ValueError('trained model not found')
        args.model_path = files_expl[0]
    save_path = f'/workspace/results/offendingCLIP/SMID/{args.language_model.split("/")[0]}/'
    # model_type = 'probe'
    model, save_path = load_model(args, save_path)

    data_set_path = '/workspace/datasets/SMID_images_400px/'
    df = pd.read_csv(os.path.join(data_set_path, 'SMID_norms.csv'), sep=',', header=0)
    valence_means = df['valence_mean'].values
    moral_means = df['moral_mean'].values

    res = list()
    for idx, image_name in enumerate(tqdm(df['img_name'].values)):
        image_path = os.path.join(data_set_path, 'img', image_name)
        image_path = glob.glob(image_path + '.*')[0]

        x = model.preprocess(Image.open(image_path)).unsqueeze(0)
        # TODO save offensive

def find_images(image_paths):
    types = ('/*.JPEG', '/*.png', '/*.jpg', '/*/*.JPEG', '/*/*.png', '/*/*.jpg')  # the tuple of file types
    files_grabbed = []
    for files in types:
        files_grabbed.extend(glob.glob(image_paths + files))
    if len(files_grabbed) == 0:
        raise ValueError('no data found')
    return files_grabbed


def load_model(args, save_path):
    model_type = args.model_type
    save_path = os.path.join(save_path, model_type)
    prompts = None
    if args.prompt_path is not None:
        save_path += '_prompt_tuned' + str(os.path.basename(os.path.dirname(os.path.dirname(args.prompt_path))))
        prompts = pickle.load(open(args.prompt_path, 'rb'))
    os.makedirs(save_path, exist_ok=True)
    if model_type == 'sim':
        model = ClipSimModel(args, prompts=prompts)
    elif model_type == 'finetuned':
        model, input_size = load_finetuned_model_clip(2, device=f'cuda:{args.gpu[0]}',
                                                      path='/workspace/results/offendingCLIP/SMID/CLIPResnet/fine_tuning/model.pt')
        model.to(f'cuda:{args.gpu[0]}')
        model.eval()
    elif model_type == 'resnet50':
        model, input_size = initialize_model_imagenet(2, True, 'cuda')
        model.load_state_dict(torch.load('/workspace/results/offendingCLIP/SMID/Resnet/fine_tuning/model.pt'))
        model.to(f'cuda:{args.gpu[0]}')
        model.eval()
    else:
        raise ValueError('not implemented')

    return model, save_path


def run_model_imagefolder(args, data_set_path, save_dir):
    torch.set_num_threads(6)
    # 'Clip_ViT-B'
    if args.model_type == 'probe':
        files_expl = glob.glob(
            f'./experiments/train_results/toxicity/*_baseline_{args.language_model.split("/")[0]}/best_model.pth.tar')
        if len(files_expl) == 0: raise ValueError('trained model not found')
        args.model_path = files_expl[0]

    if 'SMID/moral' in args.prompt_path:
        data_type = 'moral'
    elif 'SMID/valence' in args.prompt_path:
        data_type = 'valence'
    elif 'SMID/harm' in args.prompt_path:
        data_type = 'harm'
    else:
        data_type = 'unkown_datatype'


    save_path = os.path.join('/workspace/results/offendingCLIP',
                             save_dir, data_type,
                             f'{args.language_model.split("/")[0]}/')
    # model_type = 'probe'
    model, save_path = load_model(args, save_path)

    image_paths = os.path.join(data_set_path)
    image_paths = find_images(image_paths)
    res = list()
    filenames_tosave = list()

    # Create RTPT object and start the RTPT tracking
    rtpt = RTPT(name_initials='PS', experiment_name='check_offending_images', max_iterations=len(image_paths))
    rtpt.start()

    for idx, image_name in enumerate(tqdm(image_paths)):
        image_path = image_name
        x = model.preprocess(Image.open(image_path)).unsqueeze(0)
        filename = os.path.basename(image_path)
        prediction_score, predicted_label, pred_label_idx, save_filename = eval_model_(args=args,
                                                                                       x=x,
                                                                                       model=model,
                                                                                       file_name=filename.split('.')[0],
                                                                                       save_path=os.path.join(save_path,
                                                                                                              'plots'),
                                                                                       verbose=False,
                                                                                       data_type=data_type)
        if save_filename:
            filenames_tosave.append((predicted_label, pred_label_idx, prediction_score, filename))
        # res.append((image_name, prediction_score, predicted_label, pred_label_idx
        res.append((image_name, f'{prediction_score:.4f}', predicted_label, f'{pred_label_idx}'))
        rtpt.step(f'{len(image_paths)-idx-1}')
        ##if idx > len(image_paths) // 10:
        #    break
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'offending_images.csv'), 'w') as f:
        for label, idx, score, item in filenames_tosave:
            f.write(f"{label},{idx},{score:.2f},{item}\n")

    # np.savetxt(os.path.join(save_path, 'prediction.csv'), res, delimiter=',',
    #           header='img_name,prediction_score,predicted_label,pred_label_idx,valence_mean,moral_mean',
    #           fmt=('%s,%s,%s,%s,%s,%s'))


def run_model_image(args, save_dir, images, filenames):
    torch.set_num_threads(6)
    # 'Clip_ViT-B'
    if args.model_type == 'probe_protos':
        files_expl = glob.glob(
            f'./experiments/train_results/toxicity/*_baseline_{args.language_model.split("/")[0]}/best_model.pth.tar')
        if len(files_expl) == 0: raise ValueError('trained model not found')
        args.model_path = files_expl[0]
    save_path = os.path.join('/workspace/results/offendingCLIP',
                             save_dir,
                             f'{args.language_model.split("/")[0]}/')
    # model_type = 'probe'
    model, save_path = load_model(args, save_path)

    res = list()
    for img_idx, image in enumerate(images):
        x = model.preprocess(image).unsqueeze(0)
        prediction_score, predicted_label, pred_label_idx, _ = eval_model_(args=args,
                                                                           x=x,
                                                                           model=model,
                                                                           file_name=str(filenames[img_idx]).split('.')[
                                                                               0],
                                                                           save_path=os.path.join(save_path, 'plots'),
                                                                           verbose=False,
                                                                           show=True)

        # res.append((image_name, prediction_score, predicted_label, pred_label_idx
        res.append((str(filenames[img_idx]), f'{prediction_score:.4f}', predicted_label, f'{pred_label_idx}'))
    return res
    # np.savetxt(os.path.join(save_path, 'prediction.csv'), res, delimiter=',',
    #           header='img_name,prediction_score,predicted_label,pred_label_idx,valence_mean,moral_mean',
    #           fmt=('%s,%s,%s,%s,%s,%s'))


def main():
    # torch.manual_seed(0)
    # np.random.seed(0)
    torch.set_num_threads(6)
    args = parser.parse_args()

    # Create RTPT object and start the RTPT tracking
    rtpt = RTPT(name_initials='Kersting', experiment_name='CrazyStuff', max_iterations=1)
    rtpt.start()

    # eval_model()
    # run_model_smir()

    dir_name = '/workspace/datasets/imagenet1k/val'
    save_dir = 'imagenet1k_val'

    # dir_name = '/workspace/datasets/yfcc100m'
    # save_dir = 'yfcc100m'

    run_model_imagefolder(args, dir_name, save_dir)


if __name__ == '__main__':
    main()
