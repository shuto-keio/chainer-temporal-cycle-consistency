import glob
import os


def load_pen_action(dataset_dir, dict_ok=True):
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

    start = 2141
    end = 2326
    # max_len = 100

    img_paths = load_img_path(start, end, dataset_dir+"penn_action/")

    if dict_ok is True:
        return img_paths

    else:
        return list(img_paths.values())


def load_img_path(start, end, dataset_dir):
    img_paths = {}
    for i in range(start, end+1):
        tmp = glob.glob(os.path.join(
            dataset_dir, "frames", "%04d" % i, "*.jpg"))
        tmp.sort()
        img_paths["%04d" % i] = tmp

    return img_paths


def load_pouring(dataset_dir, dict_ok=True):
    img_train_paths = {}
    for i in [2, 3, 5, 6, 8, 10, 11, 13, 14, 15, 17]:
        tmp = glob.glob(
            dataset_dir + "/pouring/frames/train/pouring_" + "%03d" % i + "*.jpg")
        tmp.sort()
        img_train_paths["%03d" % i] = tmp

    img_test_paths = {}
    for i in [1, 4, 7, 9, 12, 16]:
        tmp = glob.glob(
            dataset_dir + "/pouring/frames/test/pouring_" + "%03d" % i + "*.jpg")
        tmp.sort()
        img_test_paths["%03d" % i] = tmp

    if dict_ok is True:
        return img_train_paths, img_test_paths
    else:
        return list(img_train_paths.values()), list(img_test_paths.values())
