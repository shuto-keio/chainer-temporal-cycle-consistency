from updater import tcc_updater
from model import tcc
import chainer
import os
from chainer.iterators import MultiprocessIterator
from chainer.training import Trainer
from chainer.training import extensions
from chainer.optimizer_hooks import WeightDecay

from datasets import load_penn_action, load_pouring, load_multiview_pouring
from load_dataset import load_dataset
from evaluator import evaluator
from config import CONFIG
from parser import OPTION
import shutil

output_dir = os.path.join(CONFIG.output_path, OPTION.output_name)
os.makedirs(output_dir, exist_ok=True)
shutil.copyfile("train.yaml", os.path.join(output_dir, "train.yaml"))


def main():
    # chainer.config.autotune = True
    # chainer.config.cudnn_fast_batch_normalization = True

    print("dataset", CONFIG.dataset)
    print("output_dir:", output_dir)

    if CONFIG.dataset == "tennis_serve":
        dataset = load_penn_action(
            dataset_dir=CONFIG.dataset_path, stride=CONFIG.penn_action.stride, dict_ok=False)
        dataset_train = dataset[:115]
        dataset_test = dataset[115:]
    elif CONFIG.dataset == "pouring":
        dataset_train, dataset_test = load_pouring(
            dataset_dir=CONFIG.dataset_path, stride=CONFIG.pouring.stride, dict_ok=False)
    elif CONFIG.dataset == "multiview_pouring":
        dataset_train, dataset_test = load_multiview_pouring(
            dataset_dir=CONFIG.dataset_path, stride=CONFIG.multiview_pouring.stride, dict_ok=False)
    else:
        print("dataset error.")
        exit()

    dataset_train = load_dataset(dataset_train, augment=None,
                                 img_size=CONFIG.img_size, k=CONFIG.k)
    dataset_test = load_dataset(dataset_test, augment=None,
                                img_size=CONFIG.img_size, k=CONFIG.k)
    train_iter = MultiprocessIterator(
        dataset_train, batch_size=CONFIG.batchsize, n_processes=6)
    test_iter = MultiprocessIterator(
        dataset_test, batch_size=1, n_processes=6, repeat=False, shuffle=None)

    model = tcc(use_bn=True, k=CONFIG.k)
    device = chainer.get_device(OPTION.device)
    device.use()
    model.to_device(device)

    optimizer = make_optimizer(model)

    if CONFIG.weight_decay_rate != 0:
        for param in model.params():
            param.update_rule.add_hook(WeightDecay(CONFIG.weight_decay_rate))

    updater = tcc_updater({"main": train_iter}, optimizer, device)

    trainer = Trainer(updater, (CONFIG.iteration, 'iteration'), out=output_dir)

    display_interval = (100, 'iteration')
    plot_interval = (100, 'iteration')
    trainer.extend(extensions.ProgressBar(update_interval=5))
    trainer.extend(extensions.LogReport(
        trigger=display_interval, filename='log.txt'))
    trainer.extend(extensions.PrintReport(
        ["iteration", "main/loss", "test/loss", "test/tau", "elapsed_time"]), trigger=display_interval)

    trainer.extend(extensions.PlotReport(
        ["main/loss", "test/loss"], "iteration", file_name="loss.png"), trigger=plot_interval)

    trainer.extend(evaluator(test_iter, model, device,
                             epoch=plot_interval[0], out=output_dir), trigger=plot_interval)
    trainer.extend(extensions.PlotReport(
        ["test/tau"], "iteration", file_name="tau.png"), trigger=plot_interval)

    trainer.extend(extensions.snapshot_object(
        model, "{.updater.iteration}" + ".npz"), trigger=plot_interval)

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
