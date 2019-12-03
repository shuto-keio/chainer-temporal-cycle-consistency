import ipdb
from updater import tcc_updater
from model import tcc
import chainer
# import chainerx
from chainer.iterators import MultiprocessIterator
from chainer.training import Trainer
from chainer.training import extensions
from chainer.optimizer_hooks import WeightDecay

from datasets import load_pen_action, load_pouring
from load_dataset import load_dataset
from evaluator import evaluator
from config import CONFIG
from parser import OPTION
import shutil
shutil.copyfile("train.yaml", OPTION.output_dir + "train.yaml")


def main():
    # chainer.config.autotune = True
    if CONFIG.dataset == "tennis_serve":
        dataset = load_pen_action(
            dataset_dir=OPTION.dataset_dir, dict_ok=False)
        dataset_train = dataset[:CONFIG.tennis_serve.train_size]
        dataset_test = dataset[CONFIG.tennis_serve.train_size:]
    elif CONFIG.dataset == "pouring":
        dataset_train, dataset_test = load_pouring(
            dataset_dir=OPTION.dataset_dir, dict_ok=False)
    else:
        print("dataset error.")
        exit()

    dataset_train = load_dataset(dataset_train, augment=None,
                                 img_size=CONFIG.img_size)
    dataset_test = load_dataset(dataset_test, augment=None,
                                img_size=CONFIG.img_size)

    train_iter = MultiprocessIterator(
        dataset_train, batch_size=CONFIG.batchsize, n_processes=6, shared_mem=10**9)

    test_iter = MultiprocessIterator(
        dataset_test, batch_size=CONFIG.batchsize, n_processes=6, shared_mem=10**9, repeat=False, shuffle=None)

    model = tcc(use_bn=True, k=1)
    device = chainer.get_device(OPTION.device)
    device.use()
    model.to_device(device)

    optimizer = make_optimizer(model)

    if CONFIG.weight_decay_rate != 0:
        for param in model.params():
            param.update_rule.add_hook(WeightDecay(CONFIG.weight_decay_rate))

    updater = tcc_updater({"main": train_iter}, optimizer, device)

    trainer = Trainer(updater, (CONFIG.epoch, 'epoch'), out=OPTION.output_dir)

    display_interval = (20, 'iteration')
    plot_interval = (10, 'epoch')

    trainer.extend(extensions.snapshot_object(
        model, "{.updater.epoch}" + ".npz"), trigger=(100, "epoch"))
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PlotReport(
        ["main/loss"], "epoch", file_name='loss.png'), trigger=plot_interval)
    trainer.extend(extensions.ProgressBar(update_interval=5))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', "main/loss", 'elapsed_time']), trigger=display_interval)

    # trainer.extend(evaluator(test_iter, model, device, OPTION=OPTION), trigger=(100, "epoch"))
    trainer.run()


def make_optimizer(model):
    if CONFIG.opt == "Adam":
        optimizer = chainer.optimizers.Adam(alpha=CONFIG.lr)
    elif CONFIG.opt == "SGD":
        optimizer = chainer.optimizers.SGD(lr=CONFIG.opt)
    elif CONFIG.opt == "Momentum":
        optimizer = chainer.optimizers.MomentumSGD(lr=CONFIG.opt)

    # import sys
    # sys.path.append('/home/shuto/synology/git/chainer_profutil')
    # from chainer_profutil import create_marked_profile_optimizer
    # optimizer = create_marked_profile_optimizer(
    #     chainer.optimizers.Adam(alpha=CONFIG.lr), sync=True, sync_level=2)

    optimizer.setup(model)

    return optimizer


if __name__ == "__main__":
    main()
