import argparse
from easydict import EasyDict as edict

parser = argparse.ArgumentParser()
parser.add_argument('--device', '-d', type=str, default="0")
parser.add_argument('--output_name', '-o', type=str, default="")

args = parser.parse_args()

OPTION = edict()
OPTION.device = args.device
OPTION.output_name = args.output_name
