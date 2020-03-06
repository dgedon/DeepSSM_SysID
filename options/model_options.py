import argparse


def get_model_options(model_type, dataset_name, dataset_options):

    y_dim = dataset_options.y_dim
    u_dim = dataset_options.u_dim

    # model parameters
    model_parser = argparse.ArgumentParser(description='Model Parameter')
    model_parser.add_argument('--y_dim', type=int, default=y_dim, help='dimension of y')
    model_parser.add_argument('--u_dim', type=int, default=u_dim, help='dimension of u')

    if dataset_name == 'cascaded_tank': # 40, 2, 1
        model_parser.add_argument('--h_dim', type=int, default=60, help='dimension of det. latent variable h')  # 100, 50
        model_parser.add_argument('--z_dim', type=int, default=2, help='dimension of stoch. latent variable')  # 16, 25
        model_parser.add_argument('--n_layers', type=int, default=1, help='number of RNN layers (GRU)')  # 1, 3
    elif dataset_name == 'narendra_li': # 40,2,1
        model_parser.add_argument('--h_dim', type=int, default=60, help='dimension of det. latent variable h')
        model_parser.add_argument('--z_dim', type=int, default=60, help='dimension of stoch. latent variable')  # 16, 25
        model_parser.add_argument('--n_layers', type=int, default=1, help='number of RNN layers (GRU)')  # 1, 3
    elif dataset_name == 'f16gvt':
        model_parser.add_argument('--h_dim', type=int, default=40, help='dimension of det. latent variable h')  # 100, 50
        model_parser.add_argument('--z_dim', type=int, default=2, help='dimension of stoch. latent variable')  # 16, 25
        model_parser.add_argument('--n_layers', type=int, default=1, help='number of RNN layers (GRU)')  # 1, 3
    elif dataset_name == 'toy_lgssm':
        model_parser.add_argument('--h_dim', type=int, default=50, help='dimension of det. latent variable h')
        model_parser.add_argument('--z_dim', type=int, default=2, help='dimension of stoch. latent variable')  # 16, 25
        model_parser.add_argument('--n_layers', type=int, default=1, help='number of RNN layers (GRU)')  # 1, 3
    elif dataset_name == 'wiener_hammerstein': # 50, 20
        model_parser.add_argument('--h_dim', type=int, default=10, help='dimension of det. latent variable h')
        model_parser.add_argument('--z_dim', type=int, default=10, help='dimension of stoch. latent variable')  # 16, 25
        model_parser.add_argument('--n_layers', type=int, default=2, help='number of RNN layers (GRU)')  # 1, 3

    # only if type is GMM
    if model_type == 'VRNN-GMM-I' or model_type == 'VRNN-GMM':
        model_parser.add_argument('--n_mixtures', type=int, default=5, help='number Gaussian output mixtures')

    model_options = model_parser.parse_args()

    return model_options
