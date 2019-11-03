import torch, os, re
import numpy as np

from hnn.nn_models import MLP
from hnn.hnn import HNN


# == FOR HANDLING SIMULATION DATA ===


def load_t_array(data_id, traj_id=0):
    t, _, _ = load_simulated_data(data_id, traj_id=traj_id)
    return t


def load_simulated_data(data_id, traj_id=0):
    data = np.loadtxt(f'simulation-data/simulation-data-{data_id}/data{traj_id}.csv', delimiter=',')
    return data[:, 0], data[:, 1:4], data[:, 4:7]


def find_n(directory):
    N = max([int(re.search("[a-zA-Z]*([0-9]+)\.[csvtar]*", fname).group(1)) for fname in os.listdir(directory)]) + 1
    return N


# === FOR HANDLING MODELS ===


def empty_model(args):
    output_dim = 2 #args.input_dim if args.baseline else 2
    nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
    model = HNN(args.input_dim, differentiable_model=nn_model, field_type=args.field_type) # also took baseline originally
    return model


def load_models(data_id, args):
    # FIND NUMBER OF DATA POINTS (value of N, from filenames)
    directory = f'network-models/network-model-{data_id}'
    N = find_n(directory)

    models = []
    for i in range(N):
        path = directory + f'/spin{i}.tar'
        model_state_dict = torch.load(path)

        model = empty_model(args)
        model.load_state_dict(model_state_dict)
        models.append(model)
    return models


def save_model(model, i, run_id, args):
    directory = f'network-models/network-model-{run_id}'
    os.makedirs(directory) if not os.path.exists(directory) else None

    path = directory + f'/spin{i}.tar'
    torch.save(model.state_dict(), path)