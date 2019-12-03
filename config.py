from easydict import EasyDict as edict
import yaml


with open("train.yaml", "r") as yf:
    data = yaml.load(yf)
CONFIG = edict(data)
CONFIG.img_size = tuple(CONFIG.img_size)
