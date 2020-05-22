import numpy as np
from configuration import discriminator_config, unet_config_acdc
from loaders import acdc

loader = acdc

params = {
    'seed': 1,
    'folder': 'experiment_sdnet_acdc',
    'data_len': 0,
    'epochs': 500,
    'batch_size': 4,
    'pool_size': 50,
    'split': 0,
    'description': '',
    'dataset_name': 'acdc',
    'test_dataset': 'acdc',
    'input_shape': loader.ACDCLoader().input_shape,
    'image_downsample': 1,
    'modality': 'MR',
    'prefix': 'norm',                         # Prefix used to load a dataset, e.g. norm_baseline for data_by_dog
    'augment': True,
    'model': 'sdnet.SDNet',
    'executor': 'sdnet_executor.SDNetExecutor',
    'l_mix': 1,
    'ul_mix': 1,
    'rounding': 'encoder',
    'num_mask_channels': 8,
    'num_z': 8,
    'w_adv_M': 10,
    'w_rec_X': 1,
    'w_rec_Z': 1,
    'w_kl': 0.01,
    'w_sup_M': 10,
    'w_dc': 0,
    'lr': 0.0001,
    'decay': 0.0001
}

d_mask_params = discriminator_config.params
d_mask_params['downsample_blocks'] = 4
d_mask_params['filters'] = 64
d_mask_params['lr'] = 0.0001
d_mask_params['name'] = 'D_Mask'
d_mask_params['decay'] = 0.0001
d_mask_params['output'] = '1D'

anatomy_encoder_params = unet_config_acdc.params
anatomy_encoder_params['normalise'] = 'batch'
anatomy_encoder_params['downsample'] = 4
anatomy_encoder_params['filters'] = 64
anatomy_encoder_params['out_channels'] = params['num_mask_channels']


def get():
    shp = params['input_shape']
    ratio = params['image_downsample']
    shp = (int(np.round(shp[0] / ratio)), int(np.round(shp[1] / ratio)), shp[2])

    params['input_shape'] = shp
    d_mask_params['input_shape'] = (shp[:-1]) + (loader.ACDCLoader().num_masks,)
    anatomy_encoder_params['input_shape'] = shp

    params.update({'anatomy_encoder_params': anatomy_encoder_params,
                   'd_mask_params': d_mask_params})
    return params


