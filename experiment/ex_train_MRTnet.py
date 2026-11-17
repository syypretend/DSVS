import os

from sacred import Ingredient

ingredient = Ingredient('train_MRTnet')


@ingredient.config
def base_cfg():
    # ==================== Path Config ====================
    base_dir = '/big_data/CholecT50-train'

    # ==================== CPU & GPU Config ====================
    gpus = []
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus))
    num_workers = 8

    # ==================== Train / Validate / Test Split Config ====================
    train_index = []
    valid_index = []
    # ==================== Learning Config ====================
    batch_size = 256
    grad_norm = True
    alpha = 0.16
    epochs = 200
    sequence_length = 1
    warmup_epochs = 10
    DEBUG = False
    backbone_lr = 1e-3
    subnet_lr = 1e-04
    crf_lr = 1e-03
    optim = 'sgd'
    scheduler_gamma = 0.98

    mix_prob = 0.5

    # ==================== Input Files Config ====================
    loss_weight_file = None
    adj_matrix = None
    checkpoint = None

    # ==================== Monitoring Config ====================
    print_freq = 20


def parse_config(_config):
    """ Parse base configurations and add new configurations. """

    config = Map({k: v for k, v in _config['train_MRTnet'].items()})

    # ==================== Path Config ====================
    base_dir = config['base_dir']
    data_dir = os.path.join(base_dir, 'data')
    dict_dir = os.path.join(base_dir, 'dict')
    triplet_dir = os.path.join(base_dir, 'triplet')
    map_path = os.path.join(dict_dir, 'maps.txt')
    video_names = sorted(os.listdir(data_dir))
    phase_dir = os.path.join(base_dir, "phase")

    config['data_dir'] = data_dir
    config['dict_dir'] = dict_dir
    config['triplet_dir'] = triplet_dir
    config['map_path'] = map_path
    config['video_names'] = video_names
    config["phase_dir"] = phase_dir

    # ==================== Train / Validate / Test Split Config ====================
    train_index = config['train_index']
    valid_index = config['valid_index']
    config['test_index'] = [i for i in range(45) if i not in train_index and i not in valid_index]
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
