import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys

sys.path.insert(0, '../Utilities/')

import tensorflow as tf
import numpy as np
import scipy.io
from pyDOE import lhs
import time

np.random.seed(1234)
tf.set_random_seed(1234)

class XPINN:
    # Initialize the class
    def __init__(self, xyt_0, u0, v0, xyt_b1, ub_label1, vb_label1, xyt_b2, ub_label2, vb_label2, X_f1_train, X_f2_train, X_i1_train, layers1, layers2):
        self.i = 1

        self.x0 = xyt_0[:, 0:1]
        self.y0 = xyt_0[:, 1:2]
        self.t0 = xyt_0[:, 2:3]
        self.u0 = u0
        self.v0 = v0

        self.x_b1 = xyt_b1[:, 0:1]
        self.y_b1 = xyt_b1[:, 1:2]
        self.t_b1 = xyt_b1[:, 2:3]
        self.x_b2 = xyt_b2[:, 0:1]
        self.y_b2 = xyt_b2[:, 1:2]
        self.t_b2 = xyt_b2[:, 2:3]

        self.ub_label1 = ub_label1
        self.vb_label1 = vb_label1
        self.ub_label2 = ub_label2
        self.vb_label2 = vb_label2

        self.x_f1 = X_f1_train[:, 0:1]
        self.y_f1 = X_f1_train[:, 1:2]
        self.t_f1 = X_f1_train[:, 2:3]
        self.x_f2 = X_f2_train[:, 0:1]
        self.y_f2 = X_f2_train[:, 1:2]
        self.t_f2 = X_f2_train[:, 2:3]

        self.x_i1 = X_i1_train[:, 0:1]
        self.y_i1 = X_i1_train[:, 1:2]
        self.t_i1 = X_i1_train[:, 2:3]


        self.layers1 = layers1
        self.layers2 = layers2

        self.weights1, self.biases1, self.A1 = self.initialize_NN(layers1)
        self.weights2, self.biases2, self.A2 = self.initialize_NN(layers2)

        Config = tf.ConfigProto(allow_soft_placement=True)
        Config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=Config)

        self.x0_tf = tf.placeholder(tf.float64, shape=[None, self.x0.shape[1]])
        self.y0_tf = tf.placeholder(tf.float64, shape=[None, self.y0.shape[1]])
        self.t0_tf = tf.placeholder(tf.float64, shape=[None, self.t0.shape[1]])

        self.x_b1_tf = tf.placeholder(tf.float64, shape=[None, self.x_b1.shape[1]])
        self.y_b1_tf = tf.placeholder(tf.float64, shape=[None, self.y_b1.shape[1]])
        self.t_b1_tf = tf.placeholder(tf.float64, shape=[None, self.t_b1.shape[1]])
        self.x_b2_tf = tf.placeholder(tf.float64, shape=[None, self.x_b2.shape[1]])
        self.y_b2_tf = tf.placeholder(tf.float64, shape=[None, self.y_b2.shape[1]])
        self.t_b2_tf = tf.placeholder(tf.float64, shape=[None, self.t_b2.shape[1]])

        self.ub_label1_tf = tf.placeholder(tf.float64, shape=[None, self.ub_label1.shape[1]])
        self.vb_label1_tf = tf.placeholder(tf.float64, shape=[None, self.vb_label1.shape[1]])
        self.ub_label2_tf = tf.placeholder(tf.float64, shape=[None, self.ub_label2.shape[1]])
        self.vb_label2_tf = tf.placeholder(tf.float64, shape=[None, self.vb_label2.shape[1]])

        self.x_f1_tf = tf.placeholder(tf.float64, shape=[None, self.x_f1.shape[1]])
        self.y_f1_tf = tf.placeholder(tf.float64, shape=[None, self.y_f1.shape[1]])
        self.t_f1_tf = tf.placeholder(tf.float64, shape=[None, self.t_f1.shape[1]])
        self.x_f2_tf = tf.placeholder(tf.float64, shape=[None, self.x_f2.shape[1]])
        self.y_f2_tf = tf.placeholder(tf.float64, shape=[None, self.y_f2.shape[1]])
        self.t_f2_tf = tf.placeholder(tf.float64, shape=[None, self.t_f2.shape[1]])


        self.x_i1_tf = tf.placeholder(tf.float64, shape=[None, self.x_i1.shape[1]])
        self.y_i1_tf = tf.placeholder(tf.float64, shape=[None, self.y_i1.shape[1]])
        self.t_i1_tf = tf.placeholder(tf.float64, shape=[None, self.t_i1.shape[1]])

        self.ub1_pred, self.vb1_pred = self.net_u1(self.x_f1_tf, self.y_f1_tf, self.t_f1_tf)
        self.ub2_pred, self.vb2_pred = self.net_u2(self.x_f2_tf, self.y_f2_tf, self.t_f2_tf)

        self.u1_0_pred, self.v1_0_pred = self.net_u1(self.x0_tf, self.y0_tf, self.t0_tf)

        self.u1_b_pred, self.v1_b_pred = self.net_u1(self.x_b1_tf, self.y_b1_tf, self.t_b1_tf)
        self.u2_b_pred, self.v2_b_pred = self.net_u2(self.x_b2_tf, self.y_b2_tf, self.t_b2_tf)


        self.f1_u_pred, self.f2_u_pred, self.f1_v_pred, self.f2_v_pred, \
        self.fi1_u_pred, self.fi1_v_pred,\
        self.uavgi1_pred, self.vavgi1_pred,\
        self.u1i1_pred, self.u2i1_pred, \
        self.v1i1_pred, self.v2i1_pred \
            = self.net_f(self.x_f1_tf, self.y_f1_tf, self.t_f1_tf, self.x_f2_tf, self.y_f2_tf, self.t_f2_tf,
                   self.x_i1_tf, self.y_i1_tf, self.t_i1_tf)

        self.loss1 = 1 * tf.reduce_mean(tf.square(self.u1_0_pred - self.u0)) \
                     + 1 * tf.reduce_mean(tf.square(self.v1_0_pred - self.v0)) \
                     + tf.reduce_mean(tf.square(self.u1_b_pred - self.ub_label1)) \
                     + tf.reduce_mean(tf.square(self.v1_b_pred - self.vb_label1)) \
                     + tf.reduce_mean(tf.square(self.f1_u_pred)) \
                     + tf.reduce_mean(tf.square(self.f1_v_pred)) \
                     + 1 * tf.reduce_mean(tf.square(self.fi1_u_pred)) \
                     + 1 * tf.reduce_mean(tf.square(self.fi1_v_pred)) \
                     + 1 * tf.reduce_mean(tf.square(self.u1i1_pred - self.uavgi1_pred)) \
                     + 1 * tf.reduce_mean(tf.square(self.v1i1_pred - self.vavgi1_pred))

        self.loss2 = tf.reduce_mean(tf.square(self.u2_b_pred - self.ub_label2)) \
                     + tf.reduce_mean(tf.square(self.v2_b_pred - self.vb_label2)) \
                     + tf.reduce_mean(tf.square(self.f2_u_pred)) \
                     + tf.reduce_mean(tf.square(self.f2_v_pred)) \
                     + 1 * tf.reduce_mean(tf.square(self.fi1_u_pred)) \
                     + 1 * tf.reduce_mean(tf.square(self.fi1_v_pred)) \
                     + 1 * tf.reduce_mean(tf.square(self.u2i1_pred - self.uavgi1_pred)) \
                     + 1 * tf.reduce_mean(tf.square(self.v2i1_pred - self.vavgi1_pred))

        self.loss = self.loss1 + self.loss2
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer(0.0008)
        self.train_op_Adam1 = self.optimizer_Adam.minimize(self.loss1)
        self.train_op_Adam2 = self.optimizer_Adam.minimize(self.loss2)

        init = tf.global_variables_initializer()
        self.sess.run(init)


    def initialize_NN(self, layers):
        weights = []
        biases = []
        A = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float64), dtype=tf.float64)
            a = tf.Variable(0.05, dtype=tf.float64)
            weights.append(W)
            biases.append(b)
            A.append(a)

        return weights, biases, A

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.to_double(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev)), dtype=tf.float64)

    def neural_net_tanh(self, X, weights, biases, A):
        num_layers = len(weights) + 1

        H = X
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(20 * A[l] * tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def neural_net_sin(self, X, weights, biases, A):
        num_layers = len(weights) + 1

        H = X
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.sin(20 * A[l] * tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def neural_net_cos(self, X, weights, biases, A):
        num_layers = len(weights) + 1

        H = X
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.cos(20 * A[l] * tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u1(self, x, y, t):
        uv = self.neural_net_tanh(tf.concat([x, y, t], 1), self.weights1, self.biases1, self.A1)
        u = uv[:, 0:1]
        v = uv[:, 1:2]
        return u, v

    def net_u2(self, x, y, t):
        uv = self.neural_net_sin(tf.concat([x, y, t], 1), self.weights2, self.biases2, self.A2)
        u = uv[:, 0:1]
        v = uv[:, 1:2]
        return u, v

    def net_f(self, x1, y1, t1, x2, y2, t2, xi1, yi1, ti1):
        # Sub-Net1
        u1, v1 = self.net_u1(x1, y1, t1)
        u1_x = tf.gradients(u1, x1)[0]
        v1_x = tf.gradients(v1, x1)[0]
        u1_y = tf.gradients(u1, y1)[0]
        v1_y = tf.gradients(v1, y1)[0]
        u1_t = tf.gradients(u1, t1)[0]
        v1_t = tf.gradients(v1, t1)[0]
        u1_xx = tf.gradients(u1_x, x1)[0]
        v1_xx = tf.gradients(v1_x, x1)[0]
        u1_yy = tf.gradients(u1_y, y1)[0]
        v1_yy = tf.gradients(v1_y, y1)[0]

        # Sub-Net2
        u2, v2 = self.net_u2(x2, y2, t2)
        u2_x = tf.gradients(u2, x2)[0]
        v2_x = tf.gradients(v2, x2)[0]
        u2_y = tf.gradients(u2, y2)[0]
        v2_y = tf.gradients(v2, y2)[0]
        u2_t = tf.gradients(u2, t2)[0]
        v2_t = tf.gradients(v2, t2)[0]
        u2_xx = tf.gradients(u2_x, x2)[0]
        v2_xx = tf.gradients(v2_x, x2)[0]
        u2_yy = tf.gradients(u2_y, y2)[0]
        v2_yy = tf.gradients(v2_y, y2)[0]


        # Sub-Net1, Interface 1
        u1i1, v1i1 = self.net_u1(xi1, yi1, ti1)
        u1i1_x = tf.gradients(u1i1, xi1)[0]
        v1i1_x = tf.gradients(v1i1, xi1)[0]
        u1i1_y = tf.gradients(u1i1, yi1)[0]
        v1i1_y = tf.gradients(v1i1, yi1)[0]
        u1i1_t = tf.gradients(u1i1, ti1)[0]
        v1i1_t = tf.gradients(v1i1, ti1)[0]
        u1i1_xx = tf.gradients(u1i1_x, xi1)[0]
        v1i1_xx = tf.gradients(v1i1_x, xi1)[0]
        u1i1_yy = tf.gradients(u1i1_y, yi1)[0]
        v1i1_yy = tf.gradients(v1i1_y, yi1)[0]

        # Sub-Net2, Interface 1
        u2i1, v2i1 = self.net_u2(xi1, yi1, ti1)
        u2i1_x = tf.gradients(u2i1, xi1)[0]
        v2i1_x = tf.gradients(v2i1, xi1)[0]
        u2i1_y = tf.gradients(u2i1, yi1)[0]
        v2i1_y = tf.gradients(v2i1, yi1)[0]
        u2i1_t = tf.gradients(u2i1, ti1)[0]
        v2i1_t = tf.gradients(v2i1, ti1)[0]
        u2i1_xx = tf.gradients(u2i1_x, xi1)[0]
        v2i1_xx = tf.gradients(v2i1_x, xi1)[0]
        u2i1_yy = tf.gradients(u2i1_y, yi1)[0]
        v2i1_yy = tf.gradients(v2i1_y, yi1)[0]


        # Average value
        uavgi1 = (u1i1 + u2i1) / 2

        vavgi1 = (v1i1 + v2i1) / 2

        # Residuals
        f1_u = u1_t + v1_xx + v1_yy + (3 - 2 * tf.tanh(x1) ** 2 - 2 * tf.tanh(y1) ** 2) * v1
        f2_u = u2_t + v2_xx + v2_yy + (3 - 2 * tf.tanh(x2) ** 2 - 2 * tf.tanh(y2) ** 2) * v2

        f1_v = v1_t - u1_xx - u1_yy - (3 - 2 * tf.tanh(x1) ** 2 - 2 * tf.tanh(y1) ** 2) * u1
        f2_v = v2_t - u2_xx - u2_yy - (3 - 2 * tf.tanh(x2) ** 2 - 2 * tf.tanh(y2) ** 2) * u2


        fi1_u = (u1i1_t + v1i1_xx + v1i1_yy + (3 - 2 * tf.tanh(xi1) ** 2 - 2 * tf.tanh(yi1) ** 2) * v1i1) - (u2i1_t + v2i1_xx + v2i1_yy + (3 - 2 * tf.tanh(xi1) ** 2 - 2 * tf.tanh(yi1) ** 2) * v2i1)
        fi1_v = (v1i1_t - u1i1_xx - u1i1_yy - (3 - 2 * tf.tanh(xi1) ** 2 - 2 * tf.tanh(yi1) ** 2) * u1i1) - (v2i1_t - u2i1_xx - u2i1_yy - (3 - 2 * tf.tanh(xi1) ** 2 - 2 * tf.tanh(yi1) ** 2) * u2i1)

        return f1_u, f2_u, f1_v, f2_v, \
               fi1_u, fi1_v, \
               uavgi1, vavgi1, \
               u1i1, u2i1, \
               v1i1, v2i1


    def callback(self, loss):
        print('Iter:', self.i, 'Loss:', loss)
        self.i += 1

    def train(self, nIter):
        tf_dict = {self.x0_tf: self.x0, self.y0_tf: self.y0, self.t0_tf: self.t0,
                   self.x_b1_tf: self.x_b1, self.y_b1_tf: self.y_b1, self.t_b1_tf: self.t_b1,
                   self.x_b2_tf: self.x_b2, self.y_b2_tf: self.y_b2, self.t_b2_tf: self.t_b2,
                   self.x_f1_tf: self.x_f1, self.y_f1_tf: self.y_f1, self.t_f1_tf: self.t_f1,
                   self.x_f2_tf: self.x_f2, self.y_f2_tf: self.y_f2, self.t_f2_tf: self.t_f2,
                   self.x_i1_tf: self.x_i1, self.y_i1_tf: self.y_i1, self.t_i1_tf: self.t_i1}

        MSE_history1 = []
        MSE_history2 = []

        l2_err1 = []
        l2_err2 = []

        for it in range(nIter):
            self.sess.run(self.train_op_Adam1, tf_dict)
            self.sess.run(self.train_op_Adam2, tf_dict)

            if it % 10 == 0:
                loss1_value = self.sess.run(self.loss1, tf_dict)
                loss2_value = self.sess.run(self.loss2, tf_dict)

                print(
                    'It: %d, Loss1: %.3e, Loss2: %.3e' %
                    (it, loss1_value, loss2_value))

                MSE_history1.append(loss1_value)
                MSE_history2.append(loss2_value)

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)

        return MSE_history1, MSE_history2, l2_err1, l2_err2

    def predict(self, X_star1, X_star2):

        u_star1, v_star1 = self.sess.run([self.ub1_pred, self.vb1_pred],
                                         {self.x_f1_tf: X_star1[:, 0:1], self.y_f1_tf: X_star1[:, 1:2], self.t_f1_tf: X_star1[:, 2:3]})
        u_star2, v_star2 = self.sess.run([self.ub2_pred, self.vb2_pred],
                                         {self.x_f2_tf: X_star2[:, 0:1], self.y_f2_tf: X_star2[:, 1:2], self.t_f2_tf: X_star2[:, 2:3]})

        return u_star1, u_star2, v_star1, v_star2

def analyticalSolution(x, y, t):
    u = (complex(0, 1) * np.exp((complex(0, 1) * t))) / (np.cosh(x) * np.cosh(y))
    return u

def boundary_Condition1(x, t):
    u = (complex(0, 1) * np.exp((complex(0, 1) * t))) / (np.cosh(-5) * np.cosh(x))
    return u

def boundary_Condition2(x, t):
    u = (complex(0, 1) * np.exp((complex(0, 1) * t))) / (np.cosh(5) * np.cosh(x))
    return u

def get_boundary_data(lb, ub, N_b):
    x_min, y_min, t_min = lb[0], lb[1], lb[2]
    x_max, y_max, t_max = ub[0], ub[1], ub[2]

    mesh_size = 100

    t = np.linspace(t_min, t_max, mesh_size).reshape(mesh_size, 1)

    x1 = x_min * np.ones((mesh_size, 1))
    y1 = np.random.uniform(y_min, y_max, mesh_size).reshape(mesh_size, 1)
    u1 = boundary_Condition1(y1, t)
    x1_u_train = np.hstack((x1, y1, t))

    x2 = x_max * np.ones((mesh_size, 1))
    y2 = np.random.uniform(y_min, y_max, mesh_size).reshape(mesh_size, 1)
    u2 = boundary_Condition2(y2, t)
    x2_u_train = np.hstack((x2, y2, t))

    x3 = np.random.uniform(x_min, x_max, mesh_size).reshape(mesh_size, 1)
    y3 = y_min * np.ones((mesh_size, 1))
    u3 = boundary_Condition1(x3, t)
    x3_u_train = np.hstack((x3, y3, t))

    x4 = np.random.uniform(x_min, x_max, mesh_size).reshape(mesh_size, 1)
    y4 = y_max * np.ones((mesh_size, 1))
    u4 = boundary_Condition2(x4, t)
    x4_u_train = np.hstack((x4, y4, t))

    x_u_train = np.vstack((x1_u_train, x2_u_train, x3_u_train, x4_u_train))
    u_train = np.real(np.vstack((u1, u2, u3, u4)))
    v_train = np.imag(np.vstack((u1, u2, u3, u4)))

    idx = np.random.choice(x_u_train.shape[0], N_b, replace=False)
    x_u_train = x_u_train[idx, :]
    u_train = u_train[idx, :]
    v_train = v_train[idx, :]

    return x_u_train, u_train, v_train

if __name__ == '__main__':
    N_0 = 30

    N_b1 = 150
    N_b2 = 150

    # Residual points in three subdomains
    N_f1 = 5000
    N_f2 = 5000

    # Interface points along the two interfaces
    N_I1 = 256
    N_I2 = 256

    # NN architecture in each subdomain
    L = 8
    layers1 = [3] + L * [50] + [2]
    layers2 = [3] + L * [50] + [2]

    data = scipy.io.loadmat('../Data/Schrodinger2D.mat')
    t = data['tt'].flatten()[:, None]
    x = data['xx'].flatten()[:, None]
    y = data['yy'].flatten()[:, None]
    Exact = data['u_sol']
    Exact_u = np.real(Exact)
    Exact_v = np.imag(Exact)
    Exact_h = np.sqrt(Exact_u ** 2 + Exact_v ** 2)

    t1 = t[0: 50 + 1, :]
    Exact_u1 = Exact_u[:, :, 0: 50 + 1]
    Exact_v1 = Exact_v[:, :, 0: 50 + 1]
    Exact_h1 = Exact_h[:, :, 0: 50 + 1]

    t2 = t[50: 100 + 1, :]
    Exact_u2 = Exact_u[:, :, 50: 100 + 1]
    Exact_v2 = Exact_v[:, :, 50: 100 + 1]
    Exact_h2 = Exact_h[:, :, 50: 100 + 1]

    idx = np.random.choice(x.shape[0], N_0, replace=False)
    x0 = x[idx, :]
    y0 = y[idx, :]
    x_star, y_star = np.meshgrid(x0, y0)
    x_star = x_star.flatten()[:, None]
    y_star = y_star.flatten()[:, None]

    t_star = np.zeros_like(x_star)
    xyt_0 = np.hstack((x_star, y_star, t_star))
    Exact_0 = analyticalSolution(x_star, y_star, t_star)
    u0 = np.real(Exact_0)
    v0 = np.imag(Exact_0)

    lb1 = np.array([-5, -5, 0])
    ub1 = np.array([5, 5, 0.5])
    lb2 = np.array([-5, -5, 0.5])
    ub2 = np.array([5, 5, 1])
    xyt_b1, ub_label1, vb_label1 = get_boundary_data(lb1, ub1, N_b1)
    xyt_b2, ub_label2, vb_label2 = get_boundary_data(lb2, ub2, N_b2)

    X_f1_train = lb1 + (ub1 - lb1) * lhs(3, N_f1)
    X_f2_train = lb2 + (ub2 - lb2) * lhs(3, N_f2)

    ti1 = np.zeros_like(x_star) + 0.5
    X_i1_train = np.hstack((x_star, y_star, ti1))

    model = XPINN(xyt_0, u0, v0, xyt_b1, ub_label1, vb_label1, xyt_b2, ub_label2, vb_label2, X_f1_train, X_f2_train,
                  X_i1_train, layers1, layers2)

    Max_iter = 0
    start_time = time.time()
    MSE_history1, MSE_history2, l2_err1, l2_err2 = model.train(Max_iter)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))

    # Solution prediction
    X1, Y1, T1 = np.meshgrid(x, y, t1)
    X_star1 = np.hstack((X1.flatten()[:, None], Y1.flatten()[:, None], T1.flatten()[:, None]))
    u_exact1 = Exact_u1.flatten()[:, None]
    v_exact1 = Exact_v1.flatten()[:, None]
    h_exact1 = Exact_h1.flatten()[:, None]

    X2, Y2, T2 = np.meshgrid(x, y, t2)
    X_star2 = np.hstack((X2.flatten()[:, None], Y2.flatten()[:, None], T2.flatten()[:, None]))
    u_exact2 = Exact_u2.flatten()[:, None]
    v_exact2 = Exact_v2.flatten()[:, None]
    h_exact2 = Exact_h2.flatten()[:, None]

    u_pred1, u_pred2, v_pred1, v_pred2 = model.predict(X_star1, X_star2)

    u_pred = np.vstack((u_pred1, u_pred2))
    v_pred = np.vstack((v_pred1, v_pred2))
    h_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)

    u_exact = np.vstack((u_exact1, u_exact2))
    v_exact = np.vstack((v_exact1, v_exact2))
    h_exact = np.vstack((h_exact1, h_exact2))

    error_u = np.linalg.norm(u_exact - u_pred, 2) / np.linalg.norm(u_exact, 2)
    error_v = np.linalg.norm(v_exact - v_pred, 2) / np.linalg.norm(v_exact, 2)
    error_h = np.linalg.norm(h_exact - h_pred, 2) / np.linalg.norm(h_exact, 2)
    print("All domain Error:")
    print('Error u: %e' % (error_u))
    print('Error v: %e' % (error_v))
    print('Error h: %e' % (error_h))
