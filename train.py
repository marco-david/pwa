# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

# Prevent OpenMP issue
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch, argparse
import numpy as np
import scipy.integrate
from datetime import datetime
solve_ivp = scipy.integrate.solve_ivp

from joblib import Parallel, delayed

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from hnn.nn_models import MLP
from hnn.hnn import HNN
from hnn.utils import L2_loss, rk4

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_dim', default=3, type=int, help='dimensionality of input tensor')
    parser.add_argument('--hidden_dim', default=200, type=int, help='hidden dimension of mlp')
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=2000, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=200, type=int, help='number of gradient steps between prints')
    parser.add_argument('--name', default='spin', type=str, help='only one option right now')
    parser.add_argument('--baseline', dest='baseline', action='store_true', help='run baseline or experiment?')
    parser.add_argument('--use_rk4', dest='use_rk4', action='store_true', help='integrate derivative with RK4')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--field_type', default='solenoidal', type=str, help='type of vector field to learn')
    parser.add_argument('--seed', default=np.random.rand(), type=int, help='random seed')
    parser.add_argument('--save_dir', default=THIS_DIR + '/network_results', type=str, help='where to save the trained model')
    parser.set_defaults(feature=True)
    return parser.parse_args()

def train(data, args):
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # init model and optimizer
    if args.verbose:
        print("Training baseline model:" if args.baseline else "Training HNN model:")

    output_dim = args.input_dim if args.baseline else 2
    nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
    model = HNN(args.input_dim, differentiable_model=nn_model,
              field_type=args.field_type, baseline=args.baseline)
    optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-4)

    # arrange data
    x = torch.tensor(data['x'], requires_grad=True, dtype=torch.float32)
    test_x = torch.tensor(data['test_x'], requires_grad=True, dtype=torch.float32)
    dxdt = torch.Tensor(data['dx'])
    test_dxdt = torch.Tensor(data['test_dx'])

    # vanilla train loop
    stats = {'train_loss': [], 'test_loss': []}
    for step in range(args.total_steps+1):

        # train step
        dxdt_hat = model.rk4_time_derivative(x) if args.use_rk4 else model.time_derivative(x)
        loss = L2_loss(dxdt, dxdt_hat)
        loss.backward() ; optim.step() ; optim.zero_grad()

        # run test data
        test_dxdt_hat = model.rk4_time_derivative(test_x) if args.use_rk4 else model.time_derivative(test_x)
        test_loss = L2_loss(test_dxdt, test_dxdt_hat)

        # logging
        stats['train_loss'].append(loss.item())
        stats['test_loss'].append(test_loss.item())
        if args.verbose and step % args.print_every == 0:
            print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, loss.item(), test_loss.item()))

    train_dxdt_hat = model.time_derivative(x)
    train_dist = (dxdt - train_dxdt_hat)**2
    test_dxdt_hat = model.time_derivative(test_x)
    test_dist = (test_dxdt - test_dxdt_hat)**2
    print('Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}'
        .format(train_dist.mean().item(), train_dist.std().item()/np.sqrt(train_dist.shape[0]),
            test_dist.mean().item(), test_dist.std().item()/np.sqrt(test_dist.shape[0])))

    return model, stats


def integrate_model(model, t_span, y0, **kwargs):
    def fun(t, np_x):
        x = torch.tensor(np_x, requires_grad=True, dtype=torch.float32).view(1, 3)
        dx = model.time_derivative(x).data.numpy().reshape(-1)
        return dx

    return solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)

def train_this_model():
#    print(f"Training model {+1} of {increments}...")

    t, pos, vel = data[:, 0], data[:,  1:4], data[:, 4:7]

    test_split = 0.1
    N_test = int(len(pos) * test_split)

    # Splitting pos
    indices = np.random.choice(len(pos), N_test, replace=False)
    test_pos = pos[indices]
    train_pos = np.delete(pos, indices, axis=0)

    # Splitting vel
    indices = np.random.choice(len(vel), N_test, replace=False)
    test_vel = vel[indices]
    train_vel = np.delete(vel, indices, axis=0)

    data2 = {'x': train_pos, 'test_x': test_pos, 'dx': train_vel, 'test_dx': test_vel}

    # TRAIN MODEL
    args = get_args()
    model, stats = train(data2, args)
    
    


def train_models(N):
    # Loop this shit N times and create an array of all the data
    data = []
    for i in range(N):
        raw_data = np.loadtxt(f'simulation-results/data{i}.csv', delimiter=',')
        data.append(raw_data)

    data = np.array(data)

    # Finding the number of time values we have for each trajectory
    increments = data.shape[1]

    # Loop the training maxT times
    models = []
    t = None
    models = Parallel(n_jobs=-1, verbose=verbosity_level, backend="multiprocessing")(
             map(delayed(train_this_model), [ data[:,i,:]for i in range(increments)]))
#    for i in range(increments):

    return models


def empty_model(args):
    output_dim = args.input_dim if args.baseline else 2
    nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
    model = HNN(args.input_dim, differentiable_model=nn_model,
                field_type=args.field_type, baseline=args.baseline)
    return model


def load_models(N, args):
    models = []
    for i in range(N):
        label = '-rk4' if args.use_rk4 else ''
        path = '{}/{}{}{}.tar'.format(args.save_dir, args.name, label, i)

        model_state_dict = torch.load(path)
        model = empty_model(args)
        model.load_state_dict(model_state_dict)
        models.append(model)
    return models


def load_t_array():
    data0 = np.loadtxt(f'simulation-results/data0.csv', delimiter=',')
    t = data0[:, 0]
    return t


def integrate_models_plot(models):
    t = load_t_array()

    # Integrate Model
    yi = np.asarray([0.0, 0.0, 1.0])
    kwargs = {'rtol': 1e-10, 'method': 'RK45'}
    # 't_eval': np.linspace(t_span[0], t_span[1], 1000)

    ys = []
    for i, model in enumerate(models):
        ys.append(yi)
        if i is len(models) - 1: break

        hnn_ivp = integrate_model(model, [t[i], t[i + 1]], yi, **kwargs)
        yi = hnn_ivp['y'][:, -1]  # all 3 dimensions, last value

    import matplotlib.pyplot as plt
    labels = ['x', 'y', 'z']
    for i in range(3):
        pos = np.array(ys)[:, i]
        plt.plot(t, pos, label=labels[i])

    plt.legend()
    plt.savefig(f'network-results/model-output-{datetime.now().strftime("%Y-%m-%d-%H:%M")}.pdf')
    plt.show()

    # plt.xlabel("$q$", fontsize=14)
    # plt.ylabel("$p$", rotation=0, fontsize=14)
    # plt.title("Hamiltonian NN", pad=10)
    # plt.show()

if __name__ == "__main__":
    N = 500

    models = train_models(N)

    #models = load_models(N, get_args())

    integrate_models_plot(models)


