import torch

from main.smid.dataset import setup_dataset, get_dataloaders
from main.smid.tune_utils import test, train, cross_validation
from main.models.baseline import initialize_model_imagenet, resnet_transforms
from main.models.clip import ClipSimModel, initialize_model_clip
import timm



def setup_model_clip_prompt(language_model='Clip_ViT-B/32', num_classes=2, pos_label=0):
    model = ClipSimModel(language_model=language_model, gpu=0, num_classes=num_classes, pos_label=pos_label)
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name != 'prompts':
                param.requires_grad = False

    print('Training parameters:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    return model


def setup_model_clip_probe(language_model='Clip_RN50', num_classes=2, pos_label=0):
    device = f'cuda'
    model, input_size = initialize_model_clip(num_classes, language_model=language_model, device=device)
    model.to(device)
    return model


def setup_model_clip_finetune(language_model='Clip_RN50', num_classes=2, pos_label=0):
    device = f'cuda'
    model, input_size = initialize_model_clip(num_classes, language_model=language_model, device=device, fine_tune=True)
    model.to(device)
    return model


def setup_model_imagenet_probe_full(language_model='resnet50', num_classes=2,
                                    feature_extraction_forward=False, pos_label=0):
    return _setup_model_imagenet_probe(language_model, num_classes, feature_extraction_forward,
                                       feature_extract=False)


def setup_model_imagenet_probe(language_model='resnet50', num_classes=2,
                               feature_extraction_forward=False, pos_label=0):
    model = _setup_model_imagenet_probe(language_model, num_classes, feature_extraction_forward,
                                        feature_extract=True)
    return model


def setup_model_imagenet_probe_21k_full(language_model='resnet50', num_classes=2,
                                        feature_extraction_forward=False, pos_label=0):
    return _setup_model_imagenet_probe_21k(language_model, num_classes, feature_extraction_forward,
                                           feature_extract=False)


def setup_model_imagenet_probe_21k(language_model='resnet50', num_classes=2,
                                   feature_extraction_forward=False, pos_label=0):
    return _setup_model_imagenet_probe_21k(language_model, num_classes, feature_extraction_forward,
                                           feature_extract=True)


def _setup_model_imagenet_probe_21k(language_model='resnet50', num_classes=2,
                                    feature_extraction_forward=False, feature_extract=False):
    def _load_model_weights(model, model_path):
        state = torch.load(model_path, map_location='cpu')
        for key in model.state_dict():
            if 'num_batches_tracked' in key:
                continue
            p = model.state_dict()[key]
            if key in state['state_dict']:
                ip = state['state_dict'][key]
                if p.shape == ip.shape:
                    p.data.copy_(ip.data)  # Copy the data of parameters
                else:
                    print(
                        'could not load layer: {}, mismatch shape {} ,{}'.format(key, (p.shape), (ip.shape)))
            else:
                print('could not load layer: {}, not in checkpoint'.format(key))
        return model

    def _model_loss_placeholder():
        return 0

    model = timm.create_model(language_model, pretrained=False, num_classes=num_classes)
    model = _load_model_weights(model, '/workspace/datasets/imagenet21k/models/resnet50_miil_21k.pth')
    if feature_extract:
        for name, param in model.named_parameters():
            if name == 'fc.weight' or name == 'fc.bias':
                print('Training', name)
            else:
                param.requires_grad = False
                print('Not Training', name)

    transform = resnet_transforms['test']
    model.preprocess = lambda x: transform(x)
    model.encode = lambda x: model.forward(x)

    device = f'cuda'
    model.to(device)
    return model


def _setup_model_imagenet_probe(language_model='resnet50', num_classes=2,
                                feature_extraction_forward=False, feature_extract=True):
    device = f'cuda'
    model, input_size = initialize_model_imagenet(num_classes, language_model=language_model, device=device,
                                                  feature_extraction_forward=feature_extraction_forward,
                                                  feature_extract=feature_extract)
    model.to(device)
    return model


def run_cross_validation(tag, setup_model, language_model='Clip_ViT-B/32', n_splits=2, smooth_labels=False,
                         test_size=None, train_size=0.9, t_low=2.5, num_classes=2,
                         t_high=3.5, lr=0.01, epochs=500, data_type='moral'):
    if test_size is not None:
        input(f'test_size is deprecated but set to {test_size}. Please check that deprecated impl. is enabled.\nOr is'
              f'train_size instead.')
    cross_validation(tag, setup_model, language_model=language_model,
                     n_splits=n_splits, smooth_labels=smooth_labels,
                     test_size=test_size, train_size=train_size, t_low=t_low, num_classes=num_classes,
                     t_high=t_high, lr=lr, epochs=epochs, data_type=data_type)


def main():
    torch.random.manual_seed(1)
    language_model = 'Clip_ViT-B/32'
    model = setup_model_clip_prompt(language_model)
    dataset = setup_dataset(model.preprocess, train_size=.1, verbose=False)
    train_dataloader, test_dataloader = get_dataloaders(dataset)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    train(train_dataloader, test_dataloader, model, optimizer, epochs=50)


if __name__ == '__main__':
    main()
