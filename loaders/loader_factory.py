from loaders.acdc import ACDCLoader


def init_loader(dataset):
    """
    Factory method for initialising data loaders by name.
    """
    if dataset == 'acdc':
        return ACDCLoader()
    return None