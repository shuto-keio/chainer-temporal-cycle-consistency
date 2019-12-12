from chainer.training import extensions
from chainer import reporter as reporter_module
import cupy
import chainer
from tools import eval_kendall_tau, eval_tsne
import numpy as np
import matplotlib.pyplot as plt
import ipdb


class evaluator(extensions.Evaluator):
    def __init__(self, iterator, model, device, epoch, out):
        self.epoch = 0
        self.epoch_interval = epoch
        self.count = 1
        self.out = out

        super().__init__(iterator, model, device)
        self.device = device

    def evaluate(self):
        self.epoch = self.count*self.epoch_interval
        self.count += 1

        iterator = self._iterators['main']
        model = self._targets['main']

        iterator.reset()

        summary = reporter_module.DictSummary()

        y = []
        for i in iterator:
            with chainer.function.no_backprop_mode():
                x = converter_batch(i, self.device)
                tmp = model(x).reshape(-1, 128)
                y.append(cupy.asnumpy(tmp.data))

        # Kendall's Tau
        tau = eval_kendall_tau(y)
        summary.add({"test/tau": tau})

        Y = eval_tsne(y, num=5)
        plt.figure(figsize=(15, 15), dpi=300)
        count = 0
        for i, j in enumerate(y[:5]):
            plt.plot(Y[:, 0][count:count+len(j)], Y[:, 1]
                     [count:count+len(j)], "-o", markersize=5)
            count += len(j)
        plt.savefig(self.out + str(self.epoch) + ".jpg")
        plt.close()
        return summary.compute_mean()

def converter(batch, device):
    batch = np.array(batch, dtype=np.float32)
    return device.send(batch)
