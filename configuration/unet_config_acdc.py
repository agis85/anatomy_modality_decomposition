
from loaders import acdc

loader = acdc

params = {
    'normalise': 'batch',
    'seed': 1,
    'folder': 'experiment_unet_acdc',
    'epochs': 500,
    'batch_size': 4,
    'split': 0,
    'dataset_name': 'acdc',
    'test_dataset': 'acdc',
    'prefix': 'norm',  # Prefix used to load a dataset, e.g. norm_baseline for data_by_dog
    'augment': True,
    'model': 'unet.UNet',
    'executor': 'base_executor.Executor',
    'num_masks': loader.ACDCLoader().num_masks,
    'out_channels': loader.ACDCLoader().num_masks + 1,
    'residual': False,
    'deep_supervision': False,
    'filters': 64,
    'downsample': 4,
    'input_shape': (None, None, 1),
    'modality': 'MR',
    'image_downsample': 1,
    'lr': 0.0001,
    'l_mix': 1,
    'decay': 0.0001,
}


def get():
    return params
