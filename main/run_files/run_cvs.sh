# other experiments
#CUDA_VISIBLE_DEVICES=1 python main/smid/cross_val_clipprompt.py -ts 1.0 -t prompt_tuning_clip -lm Clip_ViT-B/16 --t_low 2.0 --t_high 2.5
#CUDA_VISIBLE_DEVICES=0 python main/smid/cross_val_clipprompt.py -ts 1.0 -t prompt_tuning_clip -lm Clip_ViT-B/16 --t_low 1.5 --t_high 3.5

#CUDA_VISIBLE_DEVICES=0 python main/smid/cross_val_clipprompt.py -ts 0.01 -t prompt_tuning_clip -lm Clip_ViT-B/16
#CUDA_VISIBLE_DEVICES=5 python main/smid/cross_val_clipprompt.py -ts 0.01 -t prompt_tuning_clip -lm Clip_ViT-B/16
#CUDA_VISIBLE_DEVICES=6 python main/smid/cross_val_clipprompt.py -ts 0.01 -t prompt_tuning_clip -lm Clip_ViT-B/16

# few shot comparison
#CUDA_VISIBLE_DEVICES=0 python main/smid/cross_val_clipprompt.py -ts 0.01 -t prompt_tuning_clip -lm Clip_RN50
#CUDA_VISIBLE_DEVICES=1 python main/smid/cross_val_clipprompt.py -ts 0.01 -t probe_tuning_clip -lm Clip_RN50
#CUDA_VISIBLE_DEVICES=2 python main/smid/cross_val_clipprompt.py -ts 0.01 -t prompt_tuning_clip -lm Clip_ViT-B/32
#CUDA_VISIBLE_DEVICES=3 python main/smid/cross_val_clipprompt.py -ts 0.01 -t probe_tuning_imagenet -lm resnet50
CUDA_VISIBLE_DEVICES=0 python main/smid/cross_val_clipprompt.py -ts 0.01 -t probe_tuning_clip_full -lm Clip_RN50
#CUDA_VISIBLE_DEVICES=0 python main/smid/cross_val_clipprompt.py -ts 0.01 -t probe_tuning_imagenet_full -lm resnet50
#CUDA_VISIBLE_DEVICES=4 python main/smid/cross_val_clipprompt.py -ts 0.01 -t prompt_tuning_clip -lm Clip_ViT-B/16

# prompt tuning and probing
#CUDA_VISIBLE_DEVICES=0 python main/smid/cross_val_clipprompt.py -ts 0.02 -t prompt_tuning_clip -lm Clip_RN50
#CUDA_VISIBLE_DEVICES=1 python main/smid/cross_val_clipprompt.py -ts 0.02 -t probe_tuning_clip -lm Clip_RN50
#CUDA_VISIBLE_DEVICES=2 python main/smid/cross_val_clipprompt.py -ts 0.02 -t prompt_tuning_clip -lm Clip_ViT-B/32
#CUDA_VISIBLE_DEVICES=3 python main/smid/cross_val_clipprompt.py -ts 0.02 -t probe_tuning_imagenet -lm resnet50
CUDA_VISIBLE_DEVICES=1 python main/smid/cross_val_clipprompt.py -ts 0.02 -t probe_tuning_clip_full -lm Clip_RN50

#CUDA_VISIBLE_DEVICES=1 python main/smid/cross_val_clipprompt.py -ts 0.02 -t probe_tuning_imagenet_full -lm resnet50
#CUDA_VISIBLE_DEVICES=7 python main/smid/cross_val_clipprompt.py -ts 0.02 -t prompt_tuning_clip -lm Clip_ViT-B/16

#CUDA_VISIBLE_DEVICES=0 python main/smid/cross_val_clipprompt.py -ts 0.04 -t prompt_tuning_clip -lm Clip_RN50
#CUDA_VISIBLE_DEVICES=1 python main/smid/cross_val_clipprompt.py -ts 0.04 -t probe_tuning_clip -lm Clip_RN50
#CUDA_VISIBLE_DEVICES=2 python main/smid/cross_val_clipprompt.py -ts 0.04 -t prompt_tuning_clip -lm Clip_ViT-B/32
#CUDA_VISIBLE_DEVICES=3 python main/smid/cross_val_clipprompt.py -ts 0.04 -t probe_tuning_imagenet -lm resnet50
CUDA_VISIBLE_DEVICES=2 python main/smid/cross_val_clipprompt.py -ts 0.04 -t probe_tuning_clip_full -lm Clip_RN50
#CUDA_VISIBLE_DEVICES=3 python main/smid/cross_val_clipprompt.py -ts 0.04 -t probe_tuning_imagenet_full -lm resnet50
#CUDA_VISIBLE_DEVICES=1 python main/smid/cross_val_clipprompt.py -ts 0.04 -t prompt_tuning_clip -lm Clip_ViT-B/16

#CUDA_VISIBLE_DEVICES=0 python main/smid/cross_val_clipprompt.py -ts 0.06 -t prompt_tuning_clip -lm Clip_RN50
#CUDA_VISIBLE_DEVICES=1 python main/smid/cross_val_clipprompt.py -ts 0.06 -t probe_tuning_clip -lm Clip_RN50
#CUDA_VISIBLE_DEVICES=2 python main/smid/cross_val_clipprompt.py -ts 0.06 -t prompt_tuning_clip -lm Clip_ViT-B/32
#CUDA_VISIBLE_DEVICES=3 python main/smid/cross_val_clipprompt.py -ts 0.06 -t probe_tuning_imagenet -lm resnet50
CUDA_VISIBLE_DEVICES=3 python main/smid/cross_val_clipprompt.py -ts 0.06 -t probe_tuning_clip_full -lm Clip_RN50
#CUDA_VISIBLE_DEVICES=5 python main/smid/cross_val_clipprompt.py -ts 0.06 -t probe_tuning_imagenet_full -lm resnet50
#CUDA_VISIBLE_DEVICES=2 python main/smid/cross_val_clipprompt.py -ts 0.06 -t prompt_tuning_clip -lm Clip_ViT-B/16

#CUDA_VISIBLE_DEVICES=0 python main/smid/cross_val_clipprompt.py -ts 0.08 -t prompt_tuning_clip -lm Clip_RN50
#CUDA_VISIBLE_DEVICES=1 python main/smid/cross_val_clipprompt.py -ts 0.08 -t probe_tuning_clip -lm Clip_RN50
#CUDA_VISIBLE_DEVICES=2 python main/smid/cross_val_clipprompt.py -ts 0.08 -t prompt_tuning_clip -lm Clip_ViT-B/32
#CUDA_VISIBLE_DEVICES=3 python main/smid/cross_val_clipprompt.py -ts 0.08 -t probe_tuning_imagenet -lm resnet50
CUDA_VISIBLE_DEVICES=4 python main/smid/cross_val_clipprompt.py -ts 0.08 -t probe_tuning_clip_full -lm Clip_RN50
#CUDA_VISIBLE_DEVICES=3 python main/smid/cross_val_clipprompt.py -ts 0.08 -t probe_tuning_imagenet_full -lm resnet50
#CUDA_VISIBLE_DEVICES=3 python main/smid/cross_val_clipprompt.py -ts 0.08 -t prompt_tuning_clip -lm Clip_ViT-B/16

#CUDA_VISIBLE_DEVICES=0 python main/smid/cross_val_clipprompt.py -ts 0.1 -t prompt_tuning_clip -lm Clip_RN50
#CUDA_VISIBLE_DEVICES=1 python main/smid/cross_val_clipprompt.py -ts 0.1 -t probe_tuning_clip -lm Clip_RN50
#CUDA_VISIBLE_DEVICES=2 python main/smid/cross_val_clipprompt.py -ts 0.1 -t prompt_tuning_clip -lm Clip_ViT-B/32
#CUDA_VISIBLE_DEVICES=3 python main/smid/cross_val_clipprompt.py -ts 0.1 -t probe_tuning_imagenet -lm resnet50
CUDA_VISIBLE_DEVICES=5 python main/smid/cross_val_clipprompt.py -ts 0.1 -t probe_tuning_clip_full -lm Clip_RN50
#CUDA_VISIBLE_DEVICES=6 python main/smid/cross_val_clipprompt.py -ts 0.1 -t probe_tuning_imagenet_full -lm resnet50
#CUDA_VISIBLE_DEVICES=4 python main/smid/cross_val_clipprompt.py -ts 0.1 -t prompt_tuning_clip -lm Clip_ViT-B/16

#CUDA_VISIBLE_DEVICES=0 python main/smid/cross_val_clipprompt.py -ts 0.2 -t prompt_tuning_clip -lm Clip_RN50
#CUDA_VISIBLE_DEVICES=1 python main/smid/cross_val_clipprompt.py -ts 0.2 -t probe_tuning_clip -lm Clip_RN50
#CUDA_VISIBLE_DEVICES=2 python main/smid/cross_val_clipprompt.py -ts 0.2 -t prompt_tuning_clip -lm Clip_ViT-B/32
#CUDA_VISIBLE_DEVICES=3 python main/smid/cross_val_clipprompt.py -ts 0.2 -t probe_tuning_imagenet -lm resnet50
CUDA_VISIBLE_DEVICES=6 python main/smid/cross_val_clipprompt.py -ts 0.2 -t probe_tuning_clip_full -lm Clip_RN50
#CUDA_VISIBLE_DEVICES=7 python main/smid/cross_val_clipprompt.py -ts 0.2 -t probe_tuning_imagenet_full -lm resnet50
#CUDA_VISIBLE_DEVICES=5 python main/smid/cross_val_clipprompt.py -ts 0.2 -t prompt_tuning_clip -lm Clip_ViT-B/16

#CUDA_VISIBLE_DEVICES=0 python main/smid/cross_val_clipprompt.py -ts 0.4 -t prompt_tuning_clip -lm Clip_RN50
#CUDA_VISIBLE_DEVICES=1 python main/smid/cross_val_clipprompt.py -ts 0.4 -t probe_tuning_clip -lm Clip_RN50
#CUDA_VISIBLE_DEVICES=2 python main/smid/cross_val_clipprompt.py -ts 0.4 -t prompt_tuning_clip -lm Clip_ViT-B/32
#CUDA_VISIBLE_DEVICES=3 python main/smid/cross_val_clipprompt.py -ts 0.4 -t probe_tuning_imagenet -lm resnet50
CUDA_VISIBLE_DEVICES=7 python main/smid/cross_val_clipprompt.py -ts 0.4 -t probe_tuning_clip_full -lm Clip_RN50
#CUDA_VISIBLE_DEVICES=0 python main/smid/cross_val_clipprompt.py -ts 0.4 -t probe_tuning_imagenet_full -lm resnet50
#CUDA_VISIBLE_DEVICES=6 python main/smid/cross_val_clipprompt.py -ts 0.4 -t prompt_tuning_clip -lm Clip_ViT-B/16

#CUDA_VISIBLE_DEVICES=0 python main/smid/cross_val_clipprompt.py -ts 0.6 -t prompt_tuning_clip -lm Clip_RN50
#CUDA_VISIBLE_DEVICES=1 python main/smid/cross_val_clipprompt.py -ts 0.6 -t probe_tuning_clip -lm Clip_RN50
#CUDA_VISIBLE_DEVICES=2 python main/smid/cross_val_clipprompt.py -ts 0.6 -t prompt_tuning_clip -lm Clip_ViT-B/32
#CUDA_VISIBLE_DEVICES=3 python main/smid/cross_val_clipprompt.py -ts 0.6 -t probe_tuning_imagenet -lm resnet50
CUDA_VISIBLE_DEVICES=0 python main/smid/cross_val_clipprompt.py -ts 0.6 -t probe_tuning_clip_full -lm Clip_RN50
#CUDA_VISIBLE_DEVICES=1 python main/smid/cross_val_clipprompt.py -ts 0.6 -t probe_tuning_imagenet_full -lm resnet50
#CUDA_VISIBLE_DEVICES=1 python main/smid/cross_val_clipprompt.py -ts 0.6 -t prompt_tuning_clip -lm Clip_ViT-B/16

#CUDA_VISIBLE_DEVICES=0 python main/smid/cross_val_clipprompt.py -ts 0.8 -t prompt_tuning_clip -lm Clip_RN50
#CUDA_VISIBLE_DEVICES=1 python main/smid/cross_val_clipprompt.py -ts 0.8 -t probe_tuning_clip -lm Clip_RN50
#CUDA_VISIBLE_DEVICES=2 python main/smid/cross_val_clipprompt.py -ts 0.8 -t prompt_tuning_clip -lm Clip_ViT-B/32
#CUDA_VISIBLE_DEVICES=3 python main/smid/cross_val_clipprompt.py -ts 0.8 -t probe_tuning_imagenet -lm resnet50
CUDA_VISIBLE_DEVICES=1 python main/smid/cross_val_clipprompt.py -ts 0.8 -t probe_tuning_clip_full -lm Clip_RN50
#CUDA_VISIBLE_DEVICES=3 python main/smid/cross_val_clipprompt.py -ts 0.8 -t probe_tuning_imagenet_full -lm resnet50
#CUDA_VISIBLE_DEVICES=2 python main/smid/cross_val_clipprompt.py -ts 0.8 -t prompt_tuning_clip -lm Clip_ViT-B/16

#CUDA_VISIBLE_DEVICES=0 python main/smid/cross_val_clipprompt.py -ts 1.0 -t prompt_tuning_clip -lm Clip_RN50
#CUDA_VISIBLE_DEVICES=1 python main/smid/cross_val_clipprompt.py -ts 1.0 -t probe_tuning_clip -lm Clip_RN50
#CUDA_VISIBLE_DEVICES=2 python main/smid/cross_val_clipprompt.py -ts 1.0 -t prompt_tuning_clip -lm Clip_ViT-B/32
#CUDA_VISIBLE_DEVICES=3 python main/smid/cross_val_clipprompt.py -ts 1.0 -t probe_tuning_imagenet -lm resnet50
CUDA_VISIBLE_DEVICES=2 python main/smid/cross_val_clipprompt.py -ts 1.0 -t probe_tuning_clip_full -lm Clip_RN50
#CUDA_VISIBLE_DEVICES=5 python main/smid/cross_val_clipprompt.py -ts 1.0 -t probe_tuning_imagenet_full -lm resnet50
#CUDA_VISIBLE_DEVICES=3 python main/smid/cross_val_clipprompt.py -ts 1.0 -t prompt_tuning_clip -lm Clip_ViT-B/16
