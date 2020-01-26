import numpy as np
import chainer
import chainer.functions as F
from model import cycle_back_regression2, cycle_back_classification2, cycle_back_regression, cycle_back_classification
import ipdb
from config import CONFIG
from chainer.dataset import concat_examples


class tcc_updater(chainer.training.updaters.StandardUpdater):
    def __init__(self, iterater, optimizer, device):
        if CONFIG.alignment == "classification":
            self.alignment = cycle_back_classification2
        elif CONFIG.alignment == "regression":
            self.alignment = cycle_back_regression2

        self.converter = concat_examples
        super().__init__(iterater, optimizer, device=device)

    def update_core(self):
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        batch = train_iter.next()

        train_batch, indices = self.converter(batch, self.device)

        feature = optimizer.target(train_batch)

        loss = 0
        num = 0
        for i in range(len(batch)):
            for j in range(len(batch)):
                if i != j:
                    loss += self.alignment((feature[i], indices[i]),
                                           (feature[j], indices[j]))
                    num += len(feature[i])
        loss /= num
        optimizer.target.cleargrads()  # Clear the parameter gradients
        loss.backward()  # Backprop
        optimizer.update()  # Update the parameters
        # loss.unchain_backward()  # Truncate the graph
        chainer.reporter.report({"loss": loss}, optimizer.target)

    # def converter(batch, device):
    #     batch = np.array(batch, dtype=np.float32)
    #     return device.send(batch)
