from main.experiments import torch, parser, RTPT
from main.experiments import run_model_imagefolder

torch.set_num_threads(6)
args = parser.parse_args()


dir_name = '/workspace/datasets/imagenet1k/train'
save_dir = 'imagenet1k_train'

run_model_imagefolder(args, dir_name, save_dir)
