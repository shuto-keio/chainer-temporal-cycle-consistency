import numpy as np
import chainer
import chainer.functions as F
from model import cycle_back_classification, cycle_back_regression
import ipdb
from config import CONFIG


class tcc_updater(chainer.training.updaters.StandardUpdater):
    def __init__(self, iterater, optimizer, device):
        if CONFIG.alignment == "classification":
            self.alignment = cycle_back_classification
        elif CONFIG.alignment == "regression":
            self.alignment = cycle_back_regression

        super().__init__(iterater, optimizer, device=device)

    def update_core(self):
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        batch = train_iter.next()

        loss = 0
        num = 0

        for i in range(len(batch)):
            for j in range(len(batch)):
                if i != j:
                    train_batch1 = converter(batch[i], self.device)
                    train_batch2 = converter(batch[j], self.device)

                    train_batch1 = optimizer.target(train_batch1)
                    train_batch2 = optimizer.target(train_batch2)
                    loss += self.alignment(train_batch1, train_batch2)
                    num += len(train_batch1)
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
