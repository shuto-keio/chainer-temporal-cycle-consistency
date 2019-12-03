from chainer.training import extensions
from chainer import reporter as reporter_module
import cupy
import chainer
# from updater import converter
from tools import Kendalls_Tau
import numpy as np


class evaluator(extensions.Evaluator):
    def __init__(self, iterator, model, device, OPTION):
        super().__init__(iterator, model, device)
        self.device = device

    def evaluate(self):
        iterator = self._iterators['main']
        model = self._targets['main']

        iterator.reset()

        summary = reporter_module.DictSummary()

        y = []
        for i in iterator:
            with chainer.function.no_backprop_mode():
                x = converter(i[0], self.device)
                tmp = model(x)
                y.append(cupy.asnumpy(tmp.data))

        # Kendall's Tau
        tau = Kendalls_Tau(y)

        summary.add({"tau": tau})

        return summary.compute_mean()


def converter(batch, device):
    batch = np.array(batch)
    return device.send(batch)
