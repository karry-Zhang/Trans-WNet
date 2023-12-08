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
from Plot import plot_x_y_analy_pred, plot_x_y

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
        X = torch.reshape(X, [-1, 2, 1])
        temp1 = (-X) * torch.exp(-X * X / 2)
        temp2 = (1 - X * X) * torch.exp(-X * X / 2)
        temp3 = (3 * X - X * X * X) * torch.exp(-X * X / 2)
        temp4 = (6 * X * X - X * X * X * X - 3) * torch.exp(-X * X / 2)
        temp5 = -(X * X * X * X * X - 10 * X * X * X + 15 * X) * torch.exp(-X * X / 2)
        temp6 = (-X * X * X * X * X * X + 15 * X * X * X * X - 45 * X * X + 15) * torch.exp(-X * X / 2)
        temp7 = (-X * X * X * X * X * X * X + 21 * X * X * X * X * X - 105 * X * X * X + 105 * X) * torch.exp(
            -X * X / 2)
        linshi = torch.reshape(torch.concat([temp1, temp2, temp3, temp4, temp5, temp6, temp7], 2), [-1, 14])

        return linshi


class Model:
    def __init__(self, net, xt0, u0, v0, xt_lb, xt_ub, xt_f, lb, ub):
        self.optimizer_LBGFS = None
        self.net = net
        self.xt0 = data_device(xt0)
        self.u0 = data_device(u0)
        self.v0 = data_device(v0)
        self.xt_lb = data_device(xt_lb)
        self.xt_ub = data_device(xt_ub)
        self.xt_f = data_device(xt_f)
        self.lb = data_device(lb)
        self.ub = data_device(ub)

        self.weight_0 = torch.tensor(0.).float().cuda()
        self.weight_b = torch.tensor(0.).float().cuda()
        self.weight_f = torch.tensor(0.).float().cuda()

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
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, [0]]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0][:, [0]]

        return u, v, u_x, v_x

    def x_f_loss_fun(self, x):
        if not x.requires_grad:
            x = Variable(x, requires_grad=True)

        u, v, u_x, v_x = self.net_uv(x)
        u_t = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, [1]]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, [0]]
        v_t = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0][:, [1]]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0][:, [0]]

        f_u = u_t + 0.5 * v_xx + (u ** 2 + v ** 2) * v
        f_v = v_t - 0.5 * u_xx - (u ** 2 + v ** 2) * u

        return f_u, f_v

    def predict(self, x):
        x = data_device(x)
        u, v = self.train_U(x)
        return u.cpu().detach().numpy(), v.cpu().detach().numpy()

    # compute loss
    def PDE_loss(self):
        # u0
        u0_pred, v0_pred = self.train_U(self.xt0)
        loss_0 = torch.mean((u0_pred - self.u0) ** 2) + torch.mean((v0_pred - self.v0) ** 2)

        # residual
        f_u, f_v = self.x_f_loss_fun(self.xt_f)
        loss_f = torch.mean((f_u) ** 2) + torch.mean((f_v) ** 2)

        # boudary
        u_lb, v_lb, u_x_lb, v_x_lb = self.net_uv(self.xt_lb)
        u_ub, v_ub, u_x_ub, v_x_ub = self.net_uv(self.xt_ub)
        loss_b = torch.mean((u_lb - u_ub) ** 2) + \
                 torch.mean((v_lb - v_ub) ** 2) + \
                 torch.mean((u_x_lb - u_x_ub) ** 2) + \
                 torch.mean((v_x_lb - v_x_ub) ** 2)

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
                                                 max_iter=50000,
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
    data = scipy.io.loadmat('../Data/NLS_pinn.mat')
    t = data['tt'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = data['uu']
    Exact_u = np.real(Exact)
    Exact_v = np.imag(Exact)
    Exact_h = np.sqrt(Exact_u ** 2 + Exact_v ** 2)

    t = t[t_idx[0]: t_idx[-1] + 1, :]
    Exact_u = Exact_u[:, t_idx[0]: t_idx[-1] + 1]
    Exact_v = Exact_v[:, t_idx[0]: t_idx[-1] + 1]
    Exact_h = Exact_h[:, t_idx[0]: t_idx[-1] + 1]

    return x, t, Exact_u, Exact_v, Exact_h


def get_init_data(x, N, Exact_u, Exact_v):
    idx_x = np.random.choice(x.shape[0], N, replace=False)
    idx_x = np.sort(idx_x)
    x0 = x[idx_x, :]
    t0 = np.zeros_like(x0)
    xt0 = np.hstack((x0, t0))
    u0 = Exact_u[idx_x, 0:1]
    v0 = Exact_v[idx_x, 0:1]

    return xt0, u0, v0


def get_init_accu_data(x, N_0, lb, ub):
    x0 = np.linspace(lb[0], ub[0], N_0)[:, None]
    t0 = np.zeros_like(x0)
    e = np.e
    u0 = 4 / (e ** x0 + e ** (-x0))
    v0 = np.zeros_like(x0)

    return x0, t0, u0, v0


def get_boundary_data(t, N_b, lb, ub):
    idx_t = np.random.choice(t.shape[0], N_b, replace=False)
    tb = t[idx_t, :]

    xt_lb = np.concatenate((0 * tb + lb[0], tb), 1)
    xt_ub = np.concatenate((0 * tb + ub[0], tb), 1)

    return xt_lb, xt_ub

def load_net_para(net):
    net.load_state_dict(torch.load('params/model_parameter.pkl'))

def main(subdomain, lb, ub, t_idx, N_0, N_b, N_f, layers, xt0=None, u0=None, v0=None):
    # load data
    x, t, Exact_u, Exact_v, Exact_h = load_data(t_idx)
    " x0, u0, v0 "
    if xt0 is None:
        print("init xt0!")
        xt0, u0, v0 = get_init_data(x, N_0, Exact_u, Exact_v)
    else:
        print("not need init!")
    " xt_b "
    xt_lb, xt_ub = get_boundary_data(t, N_b, lb, ub)
    " xt_f"
    xt_f = lb + (ub - lb) * lhs(2, N_f)
    lb = np.array([-5.0, 0.0])
    ub = np.array([5.0, np.pi / 2])

    net = Net(layers)
    load_net_para(net)
    if use_gpu:
        net = net.cuda()

    model = Model(
        net=net,
        xt0=xt0,
        u0=u0,
        v0=v0,
        xt_lb=xt_lb,
        xt_ub=xt_ub,
        xt_f=xt_f,
        lb=lb,
        ub=ub,
    )

    model.train(iter=0)

    for i in range(0, 2):
        if subdomain != 1 and i == 0:
            continue
        if i == 0:
            item = 0
        else:
            item = t.shape[0] - 1
        t_pred = np.zeros_like(x) + t[item]
        xt_pred = np.hstack((x, t_pred))
        u_pred, v_pred = model.predict(xt_pred)
        h_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)
        h_exact = Exact_h[:, item: item + 1]
        plot_x_y_analy_pred(x, h_pred, h_exact, ylim=[0, 5], title=t[item][0],
                            savePath='figure/Wave_Net/' + str(t[item][0]) + '.pdf')

    end_t = t[-1][0]
    t_boundary = np.zeros((xt0.shape[0], 1)) + end_t
    xt_boundary = np.hstack((xt0[:, 0:1], t_boundary))
    u_boundary, v_boundary = model.predict(xt_boundary)

    u_exact = Exact_u.flatten()[:, None]
    v_exact = Exact_v.flatten()[:, None]
    h_exact = Exact_h.flatten()[:, None]

    if subdomain != 1:
        t = t[1:, :]
        u_exact = Exact_u[:, 1:].flatten()[:, None]
        v_exact = Exact_v[:, 1:].flatten()[:, None]
        h_exact = Exact_h[:, 1:].flatten()[:, None]

    T_test, X_test = np.meshgrid(t, x)
    xt_test = np.hstack((X_test.flatten()[:, None], T_test.flatten()[:, None]))
    u_pred, v_pred = model.predict(xt_test)
    h_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)
    error_u = np.linalg.norm(u_exact - u_pred, 2) / np.linalg.norm(u_exact, 2)
    error_v = np.linalg.norm(v_exact - v_pred, 2) / np.linalg.norm(v_exact, 2)
    error_h = np.linalg.norm(h_exact - h_pred, 2) / np.linalg.norm(h_exact, 2)
    print('Error u: %e' % (error_u))
    print('Error v: %e' % (error_v))
    print('Error h: %e' % (error_h))

    return xt_boundary, u_boundary, v_boundary, u_pred, v_pred, h_pred, u_exact, v_exact, h_exact


if __name__ == '__main__':
    L = 6
    layers = [14] + L * [300] + [2]

    N_0 = 256
    N_b = 25
    N_f = 5000

    # 1. subdomain1
    t_idx = [0, 50]
    lb = np.array([-5.0, 0.0])
    ub = np.array([5.0, np.pi / 8])
    xt_boundary, u_boundary, v_boundary, u1_pred, v1_pred, h1_pred, u1_exact, v1_exact, h1_exact = main(1, lb, ub,
                                                                                                        t_idx, N_0, N_b,
                                                                                                        N_f, layers)

    # 2. subdomain2
    t_idx = [50, 100]
    lb = np.array([-5.0, np.pi / 8])
    ub = np.array([5.0, np.pi / 4])
    xt_boundary, u_boundary, v_boundary, u2_pred, v2_pred, h2_pred, u2_exact, v2_exact, h2_exact = main(2, lb, ub,
                                                                                                        t_idx, N_0, N_b,
                                                                                                        N_f, layers,
                                                                                                        xt_boundary,
                                                                                                        u_boundary,
                                                                                                        v_boundary)

    # 3. subdomain3
    t_idx = [100, 150]
    lb = np.array([-5.0, np.pi / 4])
    ub = np.array([5.0, 3 * np.pi / 8])
    xt_boundary, u_boundary, v_boundary, u3_pred, v3_pred, h3_pred, u3_exact, v3_exact, h3_exact = main(3, lb, ub,
                                                                                                        t_idx, N_0, N_b,
                                                                                                        N_f, layers,
                                                                                                        xt_boundary,
                                                                                                        u_boundary,
                                                                                                        v_boundary)

    # 4. subdomain4
    t_idx = [150, 200]
    lb = np.array([-5.0, 3 * np.pi / 8])
    ub = np.array([5.0, np.pi / 2])
    xt_boundary, u_boundary, v_boundary, u4_pred, v4_pred, h4_pred, u4_exact, v4_exact, h4_exact = main(4, lb, ub,
                                                                                                        t_idx, N_0, N_b,
                                                                                                        N_f, layers,
                                                                                                        xt_boundary,
                                                                                                        u_boundary,
                                                                                                        v_boundary)

    u_exact = np.vstack((u1_exact, u2_exact, u3_exact, u4_exact))
    v_exact = np.vstack((v1_exact, v2_exact, v3_exact, v4_exact))
    h_exact = np.vstack((h1_exact, h2_exact, h3_exact, h4_exact))
    u_pred = np.vstack((u1_pred, u2_pred, u3_pred, u4_pred))
    v_pred = np.vstack((v1_pred, v2_pred, v3_pred, v4_pred))
    h_pred = np.vstack((h1_pred, h2_pred, h3_pred, h4_pred))

    error_u = np.linalg.norm(u_exact - u_pred, 2) / np.linalg.norm(u_exact, 2)
    error_v = np.linalg.norm(v_exact - v_pred, 2) / np.linalg.norm(v_exact, 2)
    error_h = np.linalg.norm(h_exact - h_pred, 2) / np.linalg.norm(h_exact, 2)
    print("All domain Error:")
    print('Error u: %e' % (error_u))
    print('Error v: %e' % (error_v))
    print('Error h: %e' % (error_h))