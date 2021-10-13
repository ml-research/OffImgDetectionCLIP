# Detecting offending images

This repository provides the sources code for the publication: TODO

The required datasets can be downloaded here:

SMID: https://osf.io/2rqad/

ImageNet: https://image-net.org/download.php

## Reproducing the results

### The evaluation of ImageNet-based pre-trained models as well as the prompt optimization 
The experiments can be executed by running the bash script ./main/run_files/run_cvs.sh


### Detecting offending images contained in ImageNet1k and 21k
The bash script ./main/run_files/run_eval_imagenet.sh can be used to reproduce the papers experiments.
The underlying scripts are contained in main/check_datasets and can be adapted to any Dataset which follows the pytorch torchvision.datasets.ImageFolder class.


### Notebooks to reproduce figures
Further the optimized prompts to detect offending images with CLIP models are provided in /results .

The list of as possible offensive detected images are provided in /results .

The notebooks contained in /main/notebooks can be used to reproduce the paper's figures.
