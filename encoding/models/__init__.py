#from .model_zoo import get_model
#from .model_store import get_model_file
#from .base import *
from .deeplabv3plus import get_deeplabv3plus
from .fcn import get_fcn8s
# from .archs import *

def get_segmentation_model(name, **kwargs):
    models = {
        'deeplabv3plus': get_deeplabv3plus,
        'fcn8s': get_fcn8s
        # 'psp': get_psp,
        # 'encnet': get_encnet,
        # 'danet': get_danet,
    }
    return models[name.lower()](**kwargs)
