from main.tune_clip import run_cross_validation, setup_model_clip_probe, setup_model_clip_finetune, \
    setup_model_clip_prompt, \
    setup_model_imagenet_probe, setup_model_imagenet_probe_full, \
    setup_model_imagenet_probe_21k, setup_model_imagenet_probe_21k_full

import torch
import argparse

parser = argparse.ArgumentParser(description='Crazy Stuff')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Select learning rate')
parser.add_argument('-e', '--num_epochs', default=300, type=int,
                    help='How many epochs?')
parser.add_argument('-ts', '--train_size', default=1.0, type=float,
                    help='')
parser.add_argument('--t_low', default=2.5, type=float,
                    help='')
parser.add_argument('--t_high', default=3.5, type=float,
                    help='')
parser.add_argument('-nc', '--num_classes', default=2, type=int, choices=[2,3],
                    help='')
parser.add_argument('-t', '--tag', required=True, choices=['prompt_tuning_clip', 'probe_tuning_clip',
                                                           'probe_tuning_clip_full',
                                                           'probe_tuning_imagenet', 'probe_tuning_imagenet_full',
                                                           'probe_tuning_imagenet_21k'],
                    help='')
parser.add_argument('-d', '--data_type', default='moral', choices=['harm', 'moral', 'valence'],
                    help='')
parser.add_argument('-lm', '--language_model', required=True, choices=['Clip_RN50', 'Clip_ViT-B/32', 'Clip_ViT-B/16',
                                                                       'resnet50'],
                    help='')
parser.add_argument('-sl', '--smooth_labels', type=bool, default=False,
                    help='Whether to smooth labels based on human means')

if __name__ == '__main__':
    print('Device count:', torch.cuda.device_count())
    print('CUDA available:', torch.cuda.is_available())
    torch.set_num_threads(6)

    args = parser.parse_args()

    tag = args.tag
    language_model = args.language_model
    if tag == 'probe_tuning_imagenet':
        setup_model = setup_model_imagenet_probe
        assert language_model == 'resnet50'
    elif tag == 'probe_tuning_imagenet_21k':
        setup_model = setup_model_imagenet_probe_21k
        assert language_model == 'resnet50'
    elif tag == 'probe_tuning_imagenet_21k_full':
        setup_model = setup_model_imagenet_probe_21k_full
        assert language_model == 'resnet50'
    elif tag == 'probe_tuning_imagenet_full':
        setup_model = setup_model_imagenet_probe_full
        assert language_model == 'resnet50'
    elif tag == 'probe_tuning_clip':
        setup_model = setup_model_clip_probe
        assert language_model == 'Clip_RN50' or 'Clip_ViT-B' in language_model
    elif tag == 'probe_tuning_clip_full':
        setup_model = setup_model_clip_finetune
        assert language_model == 'Clip_RN50' or 'Clip_ViT-B' in language_model
    elif tag == 'prompt_tuning_clip':
        setup_model = setup_model_clip_prompt
        assert language_model == 'Clip_RN50' or 'Clip_ViT-B' in language_model
    elif tag == 'proto_tuning_clip':
        setup_model = setup_model_clip_proto
        assert language_model == 'Clip_RN50' or 'Clip_ViT-B' in language_model
    else:
        raise ValueError('config not found')
    print(tag, language_model, '| Label smoothing:', args.smooth_labels, '| Num classes:', args.num_classes,
          '| Dataset type:', args.data_type)

    run_cross_validation(tag, setup_model, language_model=language_model,
                         n_splits=10, smooth_labels=args.smooth_labels, train_size=args.train_size,
                         t_low=args.t_low, t_high=args.t_high, num_classes=args.num_classes,
                         lr=args.lr, epochs=args.num_epochs, data_type=args.data_type)
