import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import time
import numpy as np
from pyDOE import lhs
import torch
import scipy.io
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm, trange
from Plot import plot_x_y_3D

torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

seed = 1234
torch.set_default_dtype(torch.float)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

use_gpu = torch.cuda.is_available()


def data_device(data):
    data = torch.from_numpy(data).float()
    if use_gpu:
        return data.cuda()
    else:
        return data

class Net(nn.Module):
    def __init__(self, layers):
        super(Net, self).__init__()
        self.layers = layers
        self.iter = 0
        self.activation = nn.Tanh()
        self.linear = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linear[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linear[i].bias.data)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        x = self.basis_function(x)
        a = self.act(self.linear[0](x))
        for i in range(1, len(self.layers) - 2):
            z = self.linear[i](a)
            a = self.act(z)
        a = self.linear[-1](a)
        return a

    def act(self, X):
        return torch.erf(X)

    def basis_function(self, X):
        X = torch.reshape(X, [-1, 3, 1])
        temp1 = (-X) * torch.exp(-X * X / 2)
        temp2 = (1 - X * X) * torch.exp(-X * X / 2)
        temp3 = (3 * X - X * X * X) * torch.exp(-X * X / 2)
        temp4 = (6 * X * X - X * X * X * X - 3) * torch.exp(-X * X / 2)
        temp5 = (-X * X * X * X * X + 10 * X * X * X - 15 * X) * torch.exp(-X * X / 2)
        temp6 = (-X * X * X * X * X * X + 15 * X * X * X * X - 45 * X * X + 15) * torch.exp(-X * X / 2)
        linshi = torch.reshape(torch.concat([temp1, temp2, temp3, temp4, temp5, temp6], 2), [-1, 18])

        return linshi


class Model:
    def __init__(self, net, xyt_0, u0_label, v0_label, xyt_b, ub_label, vb_label, xyt_f, lb, ub):
        self.optimizer_LBGFS = None
        self.net = net
        self.xyt_0 = data_device(xyt_0)
        self.u0_label = data_device(u0_label)
        self.v0_label = data_device(v0_label)
        self.xyt_b = data_device(xyt_b)
        self.ub_label = data_device(ub_label)
        self.vb_label = data_device(vb_label)
        self.xyt_f = data_device(xyt_f)
        self.lb = data_device(lb)
        self.ub = data_device(ub)

        self.x_label_loss_collect = []
        self.x_f_loss_collect = []

    def train_U(self, x):
        x = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        uv = self.net(x)
        return uv[:, 0:1], uv[:, 1:2]

    def net_uv(self, x):
        if not x.requires_grad:
            x = Variable(x, requires_grad=True)
        u, v = self.train_U(x)
        return u, v

    def x_f_loss_fun(self, x):
        if not x.requires_grad:
            x = Variable(x, requires_grad=True)

        u, v = self.net_uv(x)
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, [0]]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0][:, [0]]
        u_y = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, [1]]
        v_y = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0][:, [1]]
        u_yy = torch.autograd.grad(u_y, x, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, [1]]
        v_yy = torch.autograd.grad(v_y, x, grad_outputs=torch.ones_like(v_y), create_graph=True)[0][:, [1]]
        u_t = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, [2]]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, [0]]
        v_t = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0][:, [2]]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0][:, [0]]

        f_u = u_t + v_xx + v_yy + (3 - 2 * torch.tanh(x[:, 0:1]) ** 2 - 2 * torch.tanh(x[:, 1:2]) ** 2) * v
        f_v = v_t - u_xx - u_yy - (3 - 2 * torch.tanh(x[:, 0:1]) ** 2 - 2 * torch.tanh(x[:, 1:2]) ** 2) * u

        return f_u, f_v

    def predict(self, x):
        x = data_device(x)
        u, v = self.train_U(x)
        return u.cpu().detach().numpy(), v.cpu().detach().numpy()

    # compute loss
    def PDE_loss(self):
        # u0
        u0_pred, v0_pred = self.net_uv(self.xyt_0)
        loss_0 = torch.mean((u0_pred - self.u0_label) ** 2) + torch.mean((v0_pred - self.v0_label) ** 2)

        # ub vb
        ub_pred, vb_pred = self.net_uv(self.xyt_b)
        loss_b = torch.mean((ub_pred - self.ub_label) ** 2) + torch.mean((vb_pred - self.vb_label) ** 2)

        # residual
        f_u, f_v = self.x_f_loss_fun(self.xyt_f)
        loss_f = torch.mean((f_u) ** 2) + torch.mean((f_v) ** 2)

        loss = loss_0 + loss_b + loss_f

        return loss

    def LBGFS_epoch_loss(self):
        self.optimizer_LBGFS.zero_grad()
        loss = self.PDE_loss()
        loss.backward()
        self.net.iter += 1
        print('Iter:', self.net.iter, 'Loss:', loss.item())
        return loss

    def epoch_loss(self):
        loss = self.PDE_loss()
        loss.backward()
        self.net.iter += 1
        return loss

    def train(self, iter=0):
        optimizer_adam = torch.optim.Adam(self.net.parameters())
        self.optimizer_LBGFS = torch.optim.LBFGS(self.net.parameters(),
                                                 lr=1,
                                                 max_iter=20000,
                                                 max_eval=None,
                                                 history_size=100,
                                                 tolerance_grad=1e-5,
                                                 tolerance_change=1.0 * np.finfo(float).eps,
                                                 line_search_fn="strong_wolfe")

        start_time = time.time()
        pbar = trange(iter, ncols=100)
        for i in pbar:
            optimizer_adam.zero_grad()
            loss = self.epoch_loss()
            optimizer_adam.step()
            # save loss
            pbar.set_postfix({'Iter': self.net.iter,
                              'Loss': '{0:.2e}'.format(loss.item())
                              })

        print('Adam done!')
        self.optimizer_LBGFS.step(self.LBGFS_epoch_loss)
        print('LBGFS done!')

        elapsed = time.time() - start_time
        print('Training time: %.2f' % elapsed)


def load_data(t_idx):
    data = scipy.io.loadmat('../Data/Schrodinger2D.mat')
    t = data['tt'].flatten()[:, None]  # 101
    x = data['xx'].flatten()[:, None]  # 41
    y = data['yy'].flatten()[:, None]  # 41
    Exact = data['u_sol']  # 41 x 41 x 101
    Exact_u = np.real(Exact)
    Exact_v = np.imag(Exact)
    Exact_h = np.sqrt(Exact_u ** 2 + Exact_v ** 2)

    t = t[t_idx[0]: t_idx[-1] + 1, :]
    Exact_u = Exact_u[:, :, t_idx[0]: t_idx[-1] + 1]
    Exact_v = Exact_v[:, :, t_idx[0]: t_idx[-1] + 1]
    Exact_h = Exact_h[:, :, t_idx[0]: t_idx[-1] + 1]

    return x, y, t, Exact_u, Exact_v, Exact_h


def initial_Condition(x, y):
    u = complex(0, 1) / (np.cosh(x) * np.cosh(y))
    return u


def boundary_Condition1(x, t):
    u = (complex(0, 1) * np.exp((complex(0, 1) * t))) / (np.cosh(-5) * np.cosh(x))
    return u


def boundary_Condition2(x, t):
    u = (complex(0, 1) * np.exp((complex(0, 1) * t))) / (np.cosh(5) * np.cosh(x))
    return u


def analyticalSolution(x, y, t):
    u = (complex(0, 1) * np.exp((complex(0, 1) * t))) / (np.cosh(x) * np.cosh(y))
    return u


def get_init_data(x, y, N_0):
    idx = np.random.choice(x.shape[0], N_0, replace=False)
    x = x[idx, :]
    y = y[idx, :]

    x_star, y_star = np.meshgrid(x, y)
    x_star = x_star.flatten()[:, None]
    y_star = y_star.flatten()[:, None]

    t_star = np.zeros_like(x_star)
    xyt_0 = np.hstack((x_star, y_star, t_star))
    Exact_0 = analyticalSolution(x_star, y_star, t_star)

    return xyt_0, np.real(Exact_0), np.imag(Exact_0)


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

def load_net_para(net):
    net.load_state_dict(torch.load('params/model_parameter.pkl'))

def main(subdomain, N_0, N_b, N_f, lb, ub, layers, t_idx, xyt_0=None, u0_label=None, v0_label=None):
    # load data
    x, y, t, Exact_u, Exact_v, Exact_h = load_data(t_idx)
    "xt_0"
    if xyt_0 is None:
        xyt_0, u0_label, v0_label = get_init_data(x, y, N_0)
    " xt_b "
    xyt_b, ub_label, vb_label = get_boundary_data(lb, ub, N_b)
    " xyt-f"
    xyt_f = lb + (ub - lb) * lhs(3, N_f)

    net = Net(layers)
    load_net_para(net)
    if use_gpu:
        net = net.cuda()

    model = Model(
        net=net,
        xyt_0=xyt_0,
        u0_label=u0_label,
        v0_label=v0_label,
        xyt_b=xyt_b,
        ub_label=ub_label,
        vb_label=vb_label,
        xyt_f=xyt_f,
        lb=lb,
        ub=ub,
    )

    model.train(iter=0)

    end_t = t[-1][0]
    t_boundary = np.zeros((xyt_0.shape[0], 1)) + end_t
    xyt_boundary = np.hstack((xyt_0[:, 0:2], t_boundary))
    u_boundary, v_boundary = model.predict(xyt_boundary)

    " Error "
    X, Y, T = np.meshgrid(x, y, t)
    xyt_test = np.hstack((X.flatten()[:, None], Y.flatten()[:, None], T.flatten()[:, None]))
    u_exact = Exact_u.flatten()[:, None]
    v_exact = Exact_v.flatten()[:, None]
    h_exact = Exact_h.flatten()[:, None]
    u_pred, v_pred = model.predict(xyt_test)
    h_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)
    error_u = np.linalg.norm(u_exact - u_pred, 2) / np.linalg.norm(u_exact, 2)
    error_v = np.linalg.norm(v_exact - v_pred, 2) / np.linalg.norm(v_exact, 2)
    error_h = np.linalg.norm(h_exact - h_pred, 2) / np.linalg.norm(h_exact, 2)
    print('Error u: %e' % (error_u))
    print('Error v: %e' % (error_v))
    print('Error h: %e' % (error_h))

    return xyt_boundary, u_boundary, v_boundary, u_pred, v_pred, h_pred, u_exact, v_exact, h_exact


if __name__ == '__main__':
    L = 8
    layers = [18] + L * [50] + [2]

    N_0 = 30
    N_b = 150
    N_f = 5000

    lb = np.array([-5, -5, 0])
    ub = np.array([5, 5, 0.5])
    t_idx = [0, 50]
    xyt_boundary, u_boundary, v_boundary, u1_pred, v1_pred, h1_pred, u1_exact, v1_exact, h1_exact = main(1, N_0, N_b,
                                                                                                         N_f, lb, ub,
                                                                                                         layers, t_idx)

    lb = np.array([-5, -5, 0.5])
    ub = np.array([5, 5, 1])
    t_idx = [50, 100]
    xyt_boundary, u_boundary, v_boundary, u2_pred, v2_pred, h2_pred, u2_exact, v2_exact, h2_exact = main(2, N_0, N_b,
                                                                                                         N_f, lb, ub,
                                                                                                          layers, t_idx,
                                                                                                         xyt_boundary,
                                                                                                         u_boundary,
                                                                                                         v_boundary)

    u_exact = np.vstack((u1_exact, u2_exact))
    v_exact = np.vstack((v1_exact, v2_exact))
    h_exact = np.vstack((h1_exact, h2_exact))
    u_pred = np.vstack((u1_pred, u2_pred))
    v_pred = np.vstack((v1_pred, v2_pred))
    h_pred = np.vstack((h1_pred, h2_pred))

    error_u = np.linalg.norm(u_exact - u_pred, 2) / np.linalg.norm(u_exact, 2)
    error_v = np.linalg.norm(v_exact - v_pred, 2) / np.linalg.norm(v_exact, 2)
    error_h = np.linalg.norm(h_exact - h_pred, 2) / np.linalg.norm(h_exact, 2)
    print("All domain Error:")
    print('Error u: %e' % (error_u))
    print('Error v: %e' % (error_v))
    print('Error h: %e' % (error_h))