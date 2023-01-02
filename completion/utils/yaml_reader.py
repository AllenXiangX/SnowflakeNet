import yaml
from easydict import EasyDict as edict

def create_edict(pack):
    d = edict()
    for key, value in pack.items():
        if isinstance(value, dict):
            d[key] = create_edict(value)
        else:
            d[key] = value
    return d

def read_yaml(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)

    return create_edict(config)