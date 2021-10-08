from main.experiments import torch, parser, RTPT
from main.experiments import run_model_imagefolder

torch.set_num_threads(6)
args = parser.parse_args()


dir_name = '/workspace/datasets/imagenet1k/val'
save_dir = 'imagenet1k_val'

run_model_imagefolder(args, dir_name, save_dir)
