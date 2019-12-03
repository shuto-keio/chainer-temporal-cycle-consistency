import chainer
import chainer.links as L
import chainer.functions as F
import cupy

from config import CONFIG
import ipdb


class tcc(chainer.Chain):
    def __init__(self, use_bn, k):
        super().__init__()
        with self.init_scope():
            self.extracter = VGG_M(use_bn)
            self.embedder = Embedder(use_bn, k)

    def __call__(self, x):

        h = self.extracter(x)
        h = self.embedder(h)
        return h


class VGG_M(chainer.Chain):
    def __init__(self, use_bn):
        self.use_bn = use_bn
        super().__init__()

        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 64, ksize=7, stride=2, pad=3)
            self.bn1 = L.BatchNormalization(64)

            self.conv2_1 = L.Convolution2D(64, 128, ksize=3, stride=1, pad=1)
            self.bn2_1 = L.BatchNormalization(128)
            self.conv2_2 = L.Convolution2D(128, 128, ksize=3, stride=1, pad=1)
            self.bn2_2 = L.BatchNormalization(128)

            self.conv3_1 = L.Convolution2D(128, 256, ksize=3, stride=1, pad=1)
            self.bn3_1 = L.BatchNormalization(256)
            self.conv3_2 = L.Convolution2D(256, 256, ksize=3, stride=1, pad=1)
            self.bn3_2 = L.BatchNormalization(256)

            self.conv4_1 = L.Convolution2D(256, 512, ksize=3, stride=1, pad=1)
            self.bn4_1 = L.BatchNormalization(512)
            self.conv4_2 = L.Convolution2D(512, 512, ksize=3, stride=1, pad=1)
            self.bn4_2 = L.BatchNormalization(512)

    def __call__(self, x):
        h = self.conv1(x)
        if self.use_bn:
            h = self.bn1(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, ksize=3, stride=2, pad=1)

        h = self.conv2_1(h)
        if self.use_bn:
            h = self.bn2_1(h)
        h = F.relu(h)

        h = self.conv2_2(h)
        if self.use_bn:
            h = self.bn2_2(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = self.conv3_1(h)
        if self.use_bn:
            h = self.bn3_1(h)
        h = F.relu(h)
        h = self.conv3_2(h)
        if self.use_bn:
            h = self.bn3_2(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = self.conv4_1(h)
        if self.use_bn:
            h = self.bn4_1(h)
        h = F.relu(h)
        h = self.conv4_2(h)
        if self.use_bn:
            h = self.bn4_2(h)
        h = F.relu(h)

        return h


class Embedder(chainer.Chain):
    def __init__(self, use_bn, k):
        self.use_bn = use_bn
        self.k = k
        super().__init__()
        with self.init_scope():
            if CONFIG.conv_type == "2D":
                self.conv5_1 = L.Convolution2D(
                    512, 512, ksize=3, stride=1, pad=1)
            if CONFIG.conv_type == "3D":
                self.conv5_1 = L.Convolution3D(
                    512, 512, ksize=3, stride=1, pad=1)

            self.bn5_1 = L.BatchNormalization(512)

            if CONFIG.conv_type == "2D":
                self.conv5_2 = L.Convolution2D(
                    512, 512, ksize=3, stride=1, pad=1)
            elif CONFIG.conv_type == "3D":
                self.conv5_2 = L.Convolution3D(
                    512, 512, ksize=3, stride=1, pad=1)

            self.bn5_2 = L.BatchNormalization(512)

            self.fc1 = L.Linear(512, 512)
            self.fc2 = L.Linear(512, 512)
            self.fc3 = L.Linear(512, 128)

    def __call__(self, x):
        dropout_ratio = CONFIG.fc_dropout_rate
        if CONFIG.conv_type == "3D":
            x = x.transpose(1, 0, 2, 3)
            ch, time, w, h_ = x.shape
            x = x.reshape(1, ch, time, w, h_)

        h = self.conv5_1(x)
        if self.use_bn:
            h = self.bn5_1(h)
        h = F.relu(h)

        h = self.conv5_2(x)
        if self.use_bn:
            h = self.bn5_2(h)
        h = F.relu(h)

        if CONFIG.conv_type == "3D":
            h = h.reshape(ch, time, w, h_)
            h = h.transpose(1, 0, 2, 3)

        h = F.max_pooling_2d(h, ksize=h.shape[2], stride=1, pad=0)

        h = self.fc1(h)
        h = F.relu(h)
        if dropout_ratio > 0:
            h = F.dropout(h, dropout_ratio)
        h = self.fc2(h)
        h = F.relu(h)
        if dropout_ratio > 0:
            h = F.dropout(h, dropout_ratio)
        h = self.fc3(h)
        return h


def cycle_back_classification(batch1, batch2):
    loss = 0
    for i in range(len(batch1)):
        u = batch1
        v = batch2
        t = cupy.array((i,), dtype=int)

        u_i = batch1[i]

        length1 = F.sum(-(u_i-v)**2, axis=1).reshape((1, -1))
        alpha = F.softmax(length1 / CONFIG.softmax_tmp).reshape((-1))

        v_tilde = F.sum(alpha*v.T, axis=1)

        length2 = F.sum(-(u-v_tilde)**2, axis=1).reshape((1, -1))

        loss += (F.softmax_cross_entropy((length2/CONFIG.softmax_tmp), t))

    return loss


def cycle_back_regression(batch1, batch2):
    loss = 0

    for i in range(len(batch1)):
        u = batch1
        v = batch2
        t = i

        u_i = batch1[i]

        length1 = F.sum(-(u_i-v)**2, axis=1).reshape((1, -1))
        alpha = F.softmax(length1 / CONFIG.softmax_tmp).reshape((-1))

        v_tilde = F.sum(alpha*v.T, axis=1)

        length2 = F.sum(-(u-v_tilde)**2, axis=1).reshape((1, -1))
        beta = F.softmax(length2 / CONFIG.softmax_tmp)

        ave = F.sum(cupy.array(range(len(u)))*beta)
        var = F.sum(beta*(cupy.array(len(batch1)) - ave)**2)

        loss += (t-ave)**2/var + CONFIG.variance_lambda*0.5*F.log(var)

    return loss
