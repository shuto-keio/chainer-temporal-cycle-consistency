from config import CONFIG
import cupy
import chainer
import chainer.links as L
import chainer.functions as F
from chainer.links.model.vision.resnet import ResNetLayers


class tcc(chainer.Chain):
    def __init__(self, use_bn, k):
        super().__init__()
        with self.init_scope():
            if CONFIG.model == "VGGM":
                self.extracter = VGG_M(use_bn)
            elif CONFIG.model == "ResNet50":
                self.extracter = ResNet50()
            else:
                raise("model error.")
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
        batch_size, sequence_len, ch, w, h_ = x.shape
        # bathsize,time,ch,w,h >> bathsize*time,ch,w,h
        x = x.reshape(-1, ch, w, h_)

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

        h = h.reshape(batch_size, sequence_len, 512, h.shape[2], h.shape[3])

        return h


class ResNet50(chainer.Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.base = ResNetLayers(
                pretrained_model="ResNet-50-model.caffemodel", n_layers=50)

    def __call__(self, x):
        batch_size, sequence_len, ch, w, h_ = x.shape
        # bathsize,time,ch,w,h >> bathsize*time,ch,w,h
        x = x.reshape(-1, ch, w, h_)
        h = self.base(x, layers=['res4'])["res4"]
        h = h.reshape(batch_size, sequence_len, 1024, h.shape[2], h.shape[3])

        return h


class Embedder(chainer.Chain):
    def __init__(self, use_bn, k):
        self.use_bn = use_bn
        self.k = k
        super().__init__()
        with self.init_scope():
            if CONFIG.conv_type == "2D":
                self.conv5_1 = L.Convolution2D(
                    None, 512, ksize=3, stride=1, pad=1)
            elif CONFIG.conv_type in ["3D", "k"]:
                self.conv5_1 = L.Convolution3D(
                    None, 512, ksize=3, stride=1, pad=1)

            self.bn5_1 = L.BatchNormalization(512)

            if CONFIG.conv_type == "2D":
                self.conv5_2 = L.Convolution2D(
                    None, 512, ksize=3, stride=1, pad=1)
            elif CONFIG.conv_type in ["3D", "k"]:
                self.conv5_2 = L.Convolution3D(
                    None, 512, ksize=3, stride=1, pad=1)

            self.bn5_2 = L.BatchNormalization(512)

            self.fc1 = L.Linear(512, 512)
            self.fc2 = L.Linear(512, 512)
            self.fc3 = L.Linear(512, 128)

    def __call__(self, x):
        dropout_ratio = CONFIG.fc_dropout_rate

        batch_size, sequence_len, ch, w, h_ = x.shape

        if CONFIG.conv_type == "3D":
            x = x.transpose(0, 2, 1, 3, 4)
        elif CONFIG.conv_type == "2D":
            x = x.reshape(-1, ch, w, h_)
        elif CONFIG.conv_type == "k":
            x = x.reshape(-1, 2, ch, w, h_)
            x = x.transpose(0, 2, 1, 3, 4)

        h = self.conv5_1(x)
        if self.use_bn:
            h = self.bn5_1(h)
        h = F.relu(h)

        h = self.conv5_2(x)
        if self.use_bn:
            h = self.bn5_2(h)
        h = F.relu(h)

        if CONFIG.conv_type == "3D":
            # batch_size, ch, sequence_length, w, h  >> batch_size, sequence_length, ch, w, h
            h = h.transpose(0, 2, 1, 3, 4)
            # batch_size, sequence_length, ch, w, h >> batch_size*sequence_len, ch, w, h
            h = h.reshape(batch_size * sequence_len, ch, w, h_)
            h = F.max_pooling_2d(h, ksize=h.shape[-1], stride=1, pad=0)

        elif CONFIG.conv_type == "2D":
            h = F.max_pooling_2d(h, ksize=h.shape[-1], stride=1, pad=0)

        elif CONFIG.conv_type == "k":
            # batch_size, ch, sequence_length, w, h
            h = F.max_pooling_3d(
                h, ksize=(h.shape[2], h.shape[3], h.shape[4]), stride=1, pad=0)

        h = self.fc1(h)
        h = F.relu(h)
        if dropout_ratio > 0:
            h = F.dropout(h, dropout_ratio)
        h = self.fc2(h)
        h = F.relu(h)

        h = self.fc3(h)

        h = h.reshape(batch_size, -1, 128)
        return h


def cycle_back_classification(batch1, batch2):
    loss = 0
    u, _ = batch1
    v, _ = batch2

    for i in range(len(u)):
        t = cupy.array((i,), dtype=int)

        u_i = u[i]

        length1 = F.sum(-(u_i - v)**2, axis=1).reshape((1, -1))
        sim_12 = length1 / u.shape[1]
        sim_12 /= CONFIG.softmax_tmp
        alpha = F.softmax(sim_12).reshape((-1))

        v_tilde = F.sum(alpha * v.T, axis=1)

        length2 = F.sum(-(u - v_tilde)**2, axis=1).reshape((1, -1))
        sim_21 = length2 / u.shape[1]
        sim_21 /= CONFIG.softmax_tmp
        loss += (F.softmax_cross_entropy(sim_21, t))
    return loss


def cycle_back_classification_fast(batch1, batch2):
    u, u_indices = batch1
    v, v_indices = batch2

    len_u = len(u)
    len_v = len(v)

    t = cupy.arange(len(u))
    u_repeat = F.repeat(F.expand_dims(u, 0), len_v, 0).transpose(1, 0, 2)

    length1 = -F.sum((u_repeat - v)**2, axis=2)
    sim_12 = length1 / u.shape[1]
    sim_12 /= CONFIG.softmax_tmp
    alpha = F.softmax(sim_12)

    v_tilde = F.matmul(alpha, v)

    v_tilde_repeat = F.repeat(F.expand_dims(
        v_tilde, 0), len_u, 0).transpose(1, 0, 2)

    length2 = -F.sum((v_tilde_repeat - u)**2, axis=2)
    sim_21 = length2 / u.shape[1]
    sim_21 /= CONFIG.softmax_tmp
    loss = (F.softmax_cross_entropy(sim_21, t))

    return loss * len_u


def cycle_back_regression(batch1, batch2):
    loss = 0

    for i in range(len(batch1)):
        u, _ = batch1
        v, _ = batch2
        t = i

        u_i = batch1[i]

        length1 = F.sum(-(u_i - v)**2, axis=1).reshape((1, -1))
        alpha = F.softmax(length1 / CONFIG.softmax_tmp).reshape((-1))

        v_tilde = F.sum(alpha * v.T, axis=1)

        length2 = F.sum(-(u - v_tilde)**2, axis=1).reshape((1, -1))
        beta = F.softmax(length2 / CONFIG.softmax_tmp)

        ave = F.sum(cupy.array(range(len(u))) * beta)
        var = F.sum(beta * (cupy.array(len(batch1)) - ave) ** 2)

        loss += (t - ave) ** 2 / var + \
            CONFIG.variance_lambda * 0.5 * F.log(var)

    return loss


def cycle_back_regression_fast(batch1, batch2):
    var_lambda = CONFIG.variance_lambda
    u, u_indices = batch1
    v, v_indices = batch2

    len_u = len(u)
    len_v = len(v)

    t = u_indices
    import ipdb
    ipdb.set_trace()
    if CONFIG.regression.normalize_indices:
        t = t / len(u)

    u_repeat = F.repeat(F.expand_dims(u, 0), len_v, 0).transpose(1, 0, 2)

    length1 = -F.sum((u_repeat - v)**2, axis=2)
    sim_12 = length1 / u.shape[1]
    sim_12 /= CONFIG.softmax_tmp
    alpha = F.softmax(sim_12)

    v_tilde = F.matmul(alpha, v)

    v_tilde_repeat = F.repeat(F.expand_dims(
        v_tilde, 0), len_u, 0).transpose(1, 0, 2)

    length2 = -F.sum((v_tilde_repeat - u)**2, axis=2)
    sim_21 = length2 / u.shape[1]
    sim_21 /= CONFIG.softmax_tmp

    beta = F.softmax(sim_21)

    myu = F.sum(beta * t, axis=1)

    var = F.sum(((beta - t) ** 2 * beta), axis=1)

    log_var = F.log(var)

    loss = (myu - t) ** 2 * F.exp(-log_var) + var_lambda * log_var

    return F.sum(loss)
