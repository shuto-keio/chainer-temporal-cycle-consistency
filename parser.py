import ipdb
import argparse
from easydict import EasyDict as edict
from config import CONFIG
import os
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--device', '-d', type=str, default="0")
parser.add_argument('--computer', '-c', type=int, default=0)
parser.add_argument('--name', '-n', type=str, default="")

args = parser.parse_args()

OPTION = edict()
OPTION.device = args.device

dt_now = datetime.datetime.today()
time = dt_now.strftime("%Y-%m-%d-%Hh%Mm%Ss")
name = args.name
if name != "":
    name = "({})".format(name)


if args.computer == 0:
    OPTION.dataset_dir = "/media/shuto/HDD-3TB/dataset/"
    OPTION.output_dir = "/home/shuto/synology3/output/tcc/" + CONFIG.dataset + "/" + \
        time + name + "/"
elif args.computer == 1:
    OPTION.dataset_dir = "/home/ubuntu/local/dataset/"
    OPTION.output_dir = "/home/ubuntu/data/output/tcc/" + CONFIG.dataset + "/" + \
        time + name + "/"
elif args.computer == 2:
    OPTION.dataset_dir = "/home/ubuntu/data/dataset/"
    OPTION.output_dir = "/home/ubuntu/data/output/tcc/" + CONFIG.dataset + "/" + \
        time + name + "/"
else:
    print("Error. Please indicate right directory.")

os.makedirs(OPTION.output_dir, exist_ok=True)
