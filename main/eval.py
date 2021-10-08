import argparse
import torch
from rtpt import RTPT
from models import BaseNet
import clip
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import Saliency
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
import numpy as np
from torchvision.transforms import Normalize
import matplotlib.pyplot as plt
import os

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
parser.add_argument('--class_weights', default=[0.5,0.5],
                    help='Class weight for cross entropy loss')
parser.add_argument('-g','--gpu', type=int, default=[0], nargs='+',
                    help='GPU device number(s)')
parser.add_argument('--one_shot', type=bool, default=False,
                    help='Whether to use one-shot learning or not (i.e. only a few training examples)')
parser.add_argument('--discard', type=bool, default=False,
                    help='Whether edge cases in the middle between completely toxic(1) and not toxic(0) shall be omitted')
parser.add_argument('--proto_size', type=int, default=1,
                    help='Define how many words should be used to define a prototype')
parser.add_argument('--language_model', type=str, default='Bert', choices=['Bert','SentBert','GPT2', 'Clip_ViT-B/32','Clip_RN50x4', 'Clip_RN50'],
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


labels = ['non toxic', 'toxic']

inv_normalize = Normalize(
    mean=[-0.48145466/0.26862954, -0.4578275/0.26130258, -0.40821073/0.27577711],
    std=[1/0.26862954, 1/0.26130258, 1/0.27577711]
)

default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                 [(0, '#ffffff'),
                                                  (0.25, '#000000'),
                                                  (1, '#000000')], N=256)


class ClipProbeModel(torch.nn.Module):
    def __init__(self):
        super(ClipProbeModel, self).__init__()
        self.model_probe = BaseNet(args)
        self.model_probe.load_state_dict(torch.load(args.model_path))
        self.model_probe.to(f'cuda:{args.gpu[0]}')
        self.model_probe.eval()
        self.MMM, self.preprocess = clip.load(args.language_model.split('_')[1], f'cuda:{args.gpu[0]}')
        self.MMM.to(f'cuda:{args.gpu[0]}')
        self.MMM.eval()

    def forward(self, x):
        if args.input_type == 'text':
            emb = self._forward_txt(x)
        else:
            emb = self._forward_img(x)
        predicted_label = self.model_probe.forward(emb, [])
        return predicted_label

    def _forward_txt(self, x):
        return self.MMM.encode_text(x).float()

    def _forward_img(self, x):
        #x = x.unsqueeze(0)
        return self.MMM.encode_image(x).float()


def explain_pred(model, x, y, file_name):
    #gradientshap(model, x, y, file_name)
    #saliency(model, x, y, file_name)
    #occlusion(model, x, y, file_name)
    noise_tunnel(model, x, y, file_name)
    #gradientshap(model, x, y, file_name)



def eval_model(args):
    args.model_path = './experiments/train_results/toxicity/05-19-14:07_baseline_Clip_ViT-B/best_model.pth.tar'
    #args.model_path = './experiments/train_results/toxicity/05-19-17:14_baseline_Clip_RN50/best_model.pth.tar'
    #args.model_path = './experiments/train_results/toxicity/05-19-17:30_baseline_Clip_RN50x4/best_model.pth.tar'
    model = ClipProbeModel()

    if args.input_type == 'text':
        print('Eval a text')
        txt = ['You are an asshole.']
        file_name = txt[0].replace(' ', '_').replace('.', '')
        x = clip.tokenize(txt)
    elif args.input_type == 'img':
        print('Eval an image')
        file_name = 'b14_p254_12'
        #file_name = 'b14_p253_4'
        #file_name = 'b13_p233_1'
        #file_name = 'b14_p253_11'
        #file_name = 'b2_p28_8'
        #file_name = 'b10_p136_15'
        x = model.preprocess(Image.open(f"/workspace/datasets/SMID_images_400px/img/{file_name}.jpg")).unsqueeze(0)
    else:
        raise ValueError('input type unknown')
    print("Running on gpu {}".format(args.gpu))

    x = x.to(f'cuda:{args.gpu[0]}')

    logits = model(x)
    probs = logits.softmax(dim=-1)

    prediction_score, pred_label_idx = torch.topk(probs.float(), 1)

    pred_label_idx.squeeze_()
    predicted_label = labels[pred_label_idx]
    print(f'Predicted: {predicted_label} ({prediction_score.squeeze().item() * 100:.2f})')

    if args.explain and args.input_type == 'img':
        explain_pred(model, x, pred_label_idx, file_name)


def occlusion(model, x, y, file_name):
    inv_transformed_img = inv_normalize(x)
    ablator = Occlusion(model)
    attribution = ablator.attribute(x, target=y, sliding_window_shapes=(8, 8), strides=(2,2))
    fig, axis = viz.visualize_image_attr_multiple(np.transpose(attribution.cpu().detach().numpy(), (1, 2, 0)),
                                          np.transpose(inv_transformed_img.cpu().detach().numpy(), (1, 2, 0)),
                                          ["original_image", "heat_map"],
                                          ["all", "absolute_value"],
                                          cmap=default_cmap,
                                          show_colorbar=True)

    save_path = './clip_stuff/explain/toxicity/'
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, f'occlusion_{file_name}.png'))


def saliency(model, x, y, file_name):
    inv_transformed_img = inv_normalize(x)
    saliency = Saliency(model)

    attribution = saliency.attribute(x, target=y)
    fig, axis = viz.visualize_image_attr_multiple(np.transpose(attribution.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                          np.transpose(inv_transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                          ["original_image", "heat_map"],
                                          ["all", "absolute_value"],
                                          cmap=default_cmap,
                                          show_colorbar=True)

    save_path = './clip_stuff/explain/toxicity/'
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, f'saliency_{file_name}.png'))



def gradientshap(model, x, y, file_name):
    inv_transformed_img = inv_normalize(x)
    gradient_shap = GradientShap(model)

    # Defining baseline distribution of images
    rand_img_dist = torch.cat([x * 0, x * 1])

    attributions_gs = gradient_shap.attribute(x,
                                              n_samples=50,
                                              stdevs=0.0001,
                                              baselines=rand_img_dist,
                                              target=y)
    fig, axis = viz.visualize_image_attr_multiple(np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                          np.transpose(inv_transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                          ["original_image", "heat_map"],
                                          ["all", "absolute_value"],
                                          cmap=default_cmap,
                                          show_colorbar=True)

    save_path = './clip_stuff/explain/toxicity/'
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, f'gradientshap_{file_name}.png'))


def noise_tunnel(model, x, y, file_name):
    inv_transformed_img = inv_normalize(x)
    integrated_gradients = IntegratedGradients(model)
    noise_tunnel_ = NoiseTunnel(integrated_gradients)

    attributions_ig_nt = noise_tunnel_.attribute(x, nt_samples=5, nt_type='smoothgrad_sq', target=y)
    fig, axis = viz.visualize_image_attr_multiple(np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                          np.transpose(inv_transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                          ["original_image", "heat_map"],
                                          ["all", "positive"],
                                          cmap=default_cmap,
                                          show_colorbar=True,
                                          use_pyplot=False)

    save_path = './clip_stuff/explain/toxicity/'
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, f'noise_tunnel_{file_name}.png'))


if __name__ == '__main__':
    # torch.manual_seed(0)
    # np.random.seed(0)
    torch.set_num_threads(6)
    args = parser.parse_args()

    # Create RTPT object and start the RTPT tracking
    rtpt = RTPT(name_initials='Kersting', experiment_name='CrazyStuff', max_iterations=1)
    rtpt.start()

    eval_model(args)
