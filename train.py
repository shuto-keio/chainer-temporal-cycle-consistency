import ipdb
from updater import tcc_updater
from model import tcc
import chainer
# import chainerx
from chainer.iterators import MultiprocessIterator
from chainer.training import Trainer
from chainer.training import extensions
from chainer.optimizer_hooks import WeightDecay

from datasets import load_penn_action, load_pouring, load_tennismix, load_multiview_pouring
from load_dataset import load_dataset
from evaluator import evaluator
from config import CONFIG
from parser import OPTION
import shutil
shutil.copyfile("train.yaml", OPTION.output_dir + "train.yaml")


def main():
    chainer.config.autotune = True
    chainer.config.cudnn_fast_batch_normalization = True
    print(CONFIG.dataset)

    if CONFIG.dataset == "tennis_serve":
        dataset = load_penn_action(
            dataset_dir=OPTION.dataset_dir, stride=CONFIG.penn_action.stride, dict_ok=False)
        dataset_train = dataset[:115]
        dataset_test = dataset[115:]
    elif CONFIG.dataset == "pouring":
        dataset_train, dataset_test = load_pouring(
            dataset_dir=OPTION.dataset_dir, stride=CONFIG.pouring.stride, dict_ok=False)
    elif CONFIG.dataset == "multiview_pouring":
        dataset_train, dataset_test = load_multiview_pouring(
            dataset_dir=OPTION.dataset_dir, stride=CONFIG.multiview_pouring.stride, dict_ok=False)
    elif CONFIG.dataset == "tennismix":
        dataset = load_tennismix(dataset_dir=OPTION.dataset_dir,
                                 sequence_len=20, dataset_name="tennis_serve", dict_ok=False)
        dataset_train = dataset[:11]
        dataset_test = dataset[27:30]
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
        dataset_test, batch_size=1, n_processes=6, shared_mem=10**9, repeat=False, shuffle=None)

    model = tcc(use_bn=True, k=CONFIG.k)
    device = chainer.get_device(OPTION.device)
    device.use()
    model.to_device(device)

    optimizer = make_optimizer(model)

    if CONFIG.weight_decay_rate != 0:
        for param in model.params():
            param.update_rule.add_hook(WeightDecay(CONFIG.weight_decay_rate))

    updater = tcc_updater({"main": train_iter}, optimizer, device)

    trainer = Trainer(updater, (CONFIG.epoch, 'epoch'), out=OPTION.output_dir)

    display_interval = (5, 'epoch')
    plot_interval = (100, 'epoch')
    trainer.extend(extensions.ProgressBar(update_interval=5))
    trainer.extend(extensions.LogReport(
        trigger=display_interval, filename='log.txt'))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', "main/loss", 'elapsed_time']), trigger=display_interval)

    trainer.extend(extensions.PlotReport(
        ["main/loss"], "epoch", file_name='loss.png'), trigger=plot_interval)

    trainer.extend(evaluator(test_iter, model, device,
                             epoch=plot_interval[0], out=OPTION.output_dir), trigger=plot_interval)
    trainer.extend(extensions.PlotReport(
        ["test/tau"], "epoch", file_name='tau.png'), trigger=plot_interval)

    trainer.extend(extensions.snapshot_object(
        model, "{.updater.epoch}" + ".npz"), trigger=plot_interval)

    trainer.run()


def make_optimizer(model):
    if CONFIG.opt == "Adam":
        optimizer = chainer.optimizers.Adam(alpha=CONFIG.lr)
    elif CONFIG.opt == "SGD":
        optimizer = chainer.optimizers.SGD(lr=CONFIG.opt)
    elif CONFIG.opt == "Momentum":
        optimizer = chainer.optimizers.MomentumSGD(lr=CONFIG.opt)
    optimizer.setup(model)

    return optimizer


if __name__ == "__main__":
    main()
