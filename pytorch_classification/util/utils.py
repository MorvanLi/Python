import yaml

def load_cfg_from_yaml(yaml_file: str):
    with open(yaml_file, 'r') as stream:
        stream_data = stream.read()
    cfg = yaml.load(stream_data, Loader=yaml.SafeLoader)
    return cfg


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
