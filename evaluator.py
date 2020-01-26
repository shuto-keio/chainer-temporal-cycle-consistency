from chainer.training import extensions
from chainer import reporter as reporter_module
import cupy
import chainer
from model import cycle_back_regression2, cycle_back_classification2, cycle_back_regression, cycle_back_classification
from tools import eval_kendall_tau, eval_tsne
import numpy as np
import matplotlib.pyplot as plt
from chainer.dataset import concat_examples
from config import CONFIG
import ipdb


class evaluator(extensions.Evaluator):
    def __init__(self, iterator, model, device, epoch, out):
        self.epoch = 0
        self.epoch_interval = epoch
        self.count = 1
        self.out = out
        if CONFIG.alignment == "classification":
            self.alignment = cycle_back_classification2
        elif CONFIG.alignment == "regression":
            self.alignment = cycle_back_regression2

        super().__init__(iterator, model, device)
        self.device = device

    def evaluate(self):
        self.epoch = self.count*self.epoch_interval
        self.count += 1

        iterator = self._iterators['main']
        model = self._targets['main']

        iterator.reset()

        summary = reporter_module.DictSummary()

        feature = []
        indices = []

        with chainer.function.no_backprop_mode():
            for i in iterator:
                x, indices_tmp = concat_examples(i, self.device)
                tmp1 = model(x).reshape(-1, 128)
                feature.append(cupy.asnumpy(tmp1.data))
                indices.append(cupy.asnumpy(indices_tmp[0]))

            loss = 0
            num = 0
            for i in range(4):
                for j in range(4):
                    if i != j:
                        loss += self.alignment((feature[i], indices[i]),
                                               (feature[j], indices[j]))
                        num += len(feature[i])
            loss /= num

        summary.add({"test/loss": loss})
        # Kendall's Tau
        tau = eval_kendall_tau(feature)
        summary.add({"test/tau": tau})

        seq_len = CONFIG.max_img_seq
        Y = eval_tsne(feature, num=10)
        plt.figure(figsize=(15, 15), dpi=300)
        for i in range(len(Y)//seq_len):
            plt.plot(Y[i*seq_len:i*seq_len+seq_len, 0],
                     Y[i*seq_len:i*seq_len+seq_len, 1], "-o", markersize=5)
        plt.savefig(self.out + str(self.epoch) + ".jpg")
        plt.close()

        return summary.compute_mean()

# def converter(batch, device):
#     batch = np.array(batch, dtype=np.float32)
#     return device.send(batch)
