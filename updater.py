import numpy as np
import chainer
import chainer.functions as F
from model import cycle_back_regression2, cycle_back_classification2
import ipdb
from config import CONFIG


class tcc_updater(chainer.training.updaters.StandardUpdater):
    def __init__(self, iterater, optimizer, device):
        if CONFIG.alignment == "classification":
            self.alignment = cycle_back_classification2
        elif CONFIG.alignment == "regression":
            self.alignment = cycle_back_regression2
        super().__init__(iterater, optimizer, device=device)

    def update_core(self):
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        batch = train_iter.next()

        train_batch = converter_batch(batch, self.device)
        train_batch = optimizer.target(train_batch)
        # train_batch = train_batch.reshape(batch_size, CONFIG.max_img_seq, 128)

        loss = 0
        num = 0
        for i in range(len(batch)):
            for j in range(len(batch)):
                if i != j:
                    loss += self.alignment(train_batch[i], train_batch[j])
                    num += len(train_batch[i])
        loss /= num
        optimizer.target.cleargrads()  # Clear the parameter gradients
        loss.backward()  # Backprop
        optimizer.update()  # Update the parameters
        # loss.unchain_backward()  # Truncate the graph
        chainer.reporter.report({"loss": loss}, optimizer.target)


def converter(batch, device):
    batch = np.array(batch, dtype=np.float32)
    batch = batch.transpose((0, 3, 1, 2))
    return device.send(batch)


def converter_batch(batch, device):
    batch = np.array(batch, dtype=np.float32)  # batchsize,time,ch,w,h
    return device.send(batch)
