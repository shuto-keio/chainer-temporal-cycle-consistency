import glob
import os
import pandas as pd
import numpy as np
from config import CONFIG


def load_penn_action(dataset_dir, stride, dict_ok=True):
    # 'baseball_pitch',1~167
    # 'baseball_swing',
    # 'bench_press',
    # 'bowl',
    # 'clean_and_jerk',
    # 'golf_swing',
    # 'jump_rope',
    # 'jumping_jacks',
    # 'pullup',
    # 'pushup',
    # 'situp',
    # 'squat',
    # 'strum_guitar',
    # 'tennis_forehand',
    # 'tennis_serve'index=2141~2326, max_len=100
    if CONFIG.dataset == "tennis_serve":
        start = 2141
        end = 2326
    # max_len = 100

    img_paths = load_img_path(start, end, dataset_dir + "/penn_action/")
    img_paths = stride_dataset(img_paths, stride)

    if dict_ok is True:
        return img_paths
    else:
        return list(img_paths.values())


def load_img_path(start, end, dataset_dir):
    img_paths = {}
    for i in range(start, end + 1):
        tmp = glob.glob(os.path.join(
            dataset_dir, "frames", "%04d" % i, "*.jpg"))
        tmp.sort()
        img_paths["%04d" % i] = tmp

    return img_paths


def load_pouring(dataset_dir, stride, dict_ok=True):
    img_train_paths = {}
    for i in [2, 3, 5, 6, 8, 10, 11, 13, 14, 15, 17]:
        tmp = glob.glob(
            dataset_dir + "pouring/frames/train/pouring_" + "%03d" % i + "*.jpg")
        tmp.sort()
        img_train_paths["%03d" % i] = tmp

    img_test_paths = {}
    for i in [1, 4, 7, 9, 12, 16]:
        tmp = glob.glob(
            dataset_dir + "pouring/frames/test/pouring_" + "%03d" % i + "*.jpg")
        tmp.sort()
        img_test_paths["%03d" % i] = tmp

    img_train_paths = stride_dataset(img_train_paths, stride)
    img_test_paths = stride_dataset(img_test_paths, stride)

    if dict_ok is True:
        return img_train_paths, img_test_paths
    else:
        return list(img_train_paths.values()), list(img_test_paths.values())


def load_multiview_pouring(dataset_dir, stride, dict_ok=True):
    pouring = {'train': 70, 'val': 14, 'test': 32}

    img_train_paths = {}
    tmp = glob.glob(
        dataset_dir + "multiview-pouring/frames/train/*" + "real" + "*")
    tmp.sort()
#    for i in tmp:
    for i in range(pouring["train"]):
        tmp2 = glob.glob(tmp[i] + "/*.jpg")
        tmp2.sort()
        img_train_paths[os.path.basename(tmp[i])] = tmp2

    img_test_paths = {}
    tmp = glob.glob(
        dataset_dir + "multiview-pouring/frames/test/*" + "real" + "*")
#    for i in tmp:
    for i in range(pouring["test"]):
        tmp2 = glob.glob(tmp[i] + "/*.jpg")
        tmp2.sort()
        img_test_paths[os.path.basename(tmp[i])] = tmp2

    img_train_paths = stride_dataset(img_train_paths, stride)
    img_test_paths = stride_dataset(img_test_paths, stride)

    if dict_ok is True:
        return img_train_paths, img_test_paths
    else:
        return list(img_train_paths.values()), list(img_test_paths.values())


def load_tennismix(dataset_dir, sequence_len, dataset_name="tennis_serve", dict_ok=True):
    dataset_path = dataset_dir
    img_paths = dataset_path + "tennis_mov/crop_person_all2/"
    shot_csv_path = dataset_path + "tennis_mov/shot_frame/"
    # output_dir = "/media/shuto/HDD-3TB/dataset/tennis_mixed/"
    shot_csv_paths_f = sorted(glob.glob(shot_csv_path + "*_f.csv"))
    shot_csv_paths_l = sorted(glob.glob(shot_csv_path + "*_l.csv"))
    shot_csv_f = [pd.read_csv(shot_csv_paths_f[i], header=None).values[0]
                  for i in range(len(shot_csv_paths_f))]
    shot_csv_l = [pd.read_csv(shot_csv_paths_l[i], header=None).values[0]
                  for i in range(len(shot_csv_paths_l))]

    if dataset_name == "tennis_serve":
        serve_player = []  # 0:nearside, 1:farside
        for i in range(len(shot_csv_f)):
            if shot_csv_f[i][0] == -1:
                serve_player.append(1)
            elif shot_csv_l[i][0] == -1:
                serve_player.append(0)
            else:
                if shot_csv_f[i][0] < shot_csv_l[i][0]:
                    serve_player.append(0)
                else:
                    serve_player.append(1)
        img_dirs = sorted(glob.glob(img_paths + "*"))
        img_paths_f = [sorted(glob.glob(i + "/*_f.jpg")) for i in img_dirs]
        img_paths_l = [sorted(glob.glob(i + "/*_l.jpg")) for i in img_dirs]
        sequence_len = 20
        # randomized img sequence
        noise_index = np.random.randint(-5, 5, (len(shot_csv_f)))

        img_serve_paths = []
        for i in range(len(shot_csv_f)):
            if serve_player[i] == 0:
                shot_frame_tmp = shot_csv_f[i][0] - 1
                index_min = shot_frame_tmp - sequence_len // 2 + noise_index[i]
                index_max = shot_frame_tmp + sequence_len // 2 + noise_index[i]
                img_serve_paths.append(img_paths_f[i][index_min:index_max])
            elif serve_player[i] == 1:
                shot_frame_tmp = shot_csv_l[i][0] - 1
                index_min = shot_frame_tmp - sequence_len // 2 + noise_index[i]
                index_max = shot_frame_tmp + sequence_len // 2 + noise_index[i]
                img_serve_paths.append(img_paths_l[i][index_min:index_max])
    return img_serve_paths


def stride_dataset(dataset, stride):
    if stride != 1:
        dataset_new = {}
        for key_tmp, value_tmp in dataset.items():
            tmp = [value_tmp[j] for j in range(0, len(value_tmp), stride)]
            dataset_new[key_tmp] = tmp
    else:
        dataset_new = dataset

    return dataset_new
