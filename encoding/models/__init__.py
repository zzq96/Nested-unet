#from .model_zoo import get_model
#from .model_store import get_model_file
#from .base import *
from .deeplabv3plus import DeepLabV3Plus
# from .archs import *

# def get_segmentation_model(name, **kwargs):
#     from .fcn import get_fcn
#     models = {
#         'fcn': get_fcn,
#         'psp': get_psp,
#         'encnet': get_encnet,
#         'danet': get_danet,
#     }
#     return models[name.lower()](**kwargs)
