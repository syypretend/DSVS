import os

from sacred import Ingredient

ingredient = Ingredient('test_multi_objective_binary_programming')


@ingredient.config
def base_cfg():
    # ==================== Path Config ====================
    hidden_data_dir = '/home/syy/FedMVSnet/data/hidden_data.npy'

    # ==================== Learning Config ====================
    min_size = 15
    jump = 1
    pen = 0.03


def parse_config(_config):
    """ Parse base configurations and add new configurations. """

    config = Map({k: v for k, v in _config['test_multi_objective_binary_programming'].items()})

    return config


class Map(dict):
    """ Support getting dict item by dot(.) operation """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, obj):
        new_dict = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, dict):
                    new_dict[k] = Map(v)
                else:
                    new_dict[k] = v
        else:
            raise TypeError(f"`obj` must be a dict, got {type(obj)}")
        super(Map, self).__init__(new_dict)
