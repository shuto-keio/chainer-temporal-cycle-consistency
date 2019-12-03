import argparse
from easydict import EasyDict as edict
from config import CONFIG
import os
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--device', '-d', type=str, default="0")
parser.add_argument('--computer', '-c', type=int, default=0)

args = parser.parse_args()

OPTION = edict()
OPTION.device = args.device

dt_now = datetime.datetime.today()
time = dt_now.strftime("%Y-%m-%d-%Hh%Mm%Ss")

if args.computer == 0:
    OPTION.dataset_dir = "/media/shuto/HDD-3TB/dataset/public/"
    OPTION.output_dir = "/home/shuto/synology2/output/tcc/" + \
        time + "(" + CONFIG.out_dir + ")/"
elif args.computer == 1:
    OPTION.dataset_dir = "/home/ubuntu/local/dataset/public/"
    OPTION.output_dir = "/home/ubuntu/data/output/tcc/" + \
        time + "(" + CONFIG.out_dir + ")/"
elif args.computer == 1:
    OPTION.dataset_dir = "/home/ubuntu/data/dataset/public/"
    OPTION.output_dir = "/home/ubuntu/data/output/tcc/" + \
        time + "(" + CONFIG.out_dir + ")/"
else:
    print("Error. Please indicate right directory.")


os.makedirs(OPTION.output_dir, exist_ok=True)
