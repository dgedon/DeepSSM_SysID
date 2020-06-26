import argparse


def get_dataset_options(dataset_name):

    """Not used datasets"""
    """if dataset_name == 'cascaded_tank':
        dataset_parser = argparse.ArgumentParser(description='dynamic system parameter: cascaded tank')
        dataset_parser.add_argument('--y_dim', type=int, default=1, help='dimension of y')
        dataset_parser.add_argument('--u_dim', type=int, default=1, help='dimension of u')
        dataset_parser.add_argument('--seq_len_train', type=int, default=128, help='training sequence length')
        dataset_parser.add_argument('--seq_len_test', type=int, default=None, help='test sequence length')
        dataset_parser.add_argument('--seq_len_val', type=int, default=128, help='validation sequence length')
        dataset_options = dataset_parser.parse_args()

    elif dataset_name == 'f16gvt':
        dataset_parser = argparse.ArgumentParser(description='dynamic system parameter: f-16')
        dataset_parser.add_argument('--y_dim', type=int, default=3, help='dimension of y')
        dataset_parser.add_argument('--u_dim', type=int, default=1, help='dimension of u')
        dataset_parser.add_argument('--seq_len_train', type=int, default=2048, help='training sequence length')
        dataset_parser.add_argument('--seq_len_test', type=int, default=2048, help='test sequence length')
        dataset_parser.add_argument('--seq_len_val', type=int, default=2048, help='validation sequence length')
        dataset_options = dataset_parser.parse_args()"""

    if dataset_name == 'narendra_li':
        dataset_parser = argparse.ArgumentParser(description='dynamic system parameter: narendra li')
        dataset_parser.add_argument('--y_dim', type=int, default=1, help='dimension of y')
        dataset_parser.add_argument('--u_dim', type=int, default=1, help='dimension of u')
        dataset_parser.add_argument('--seq_len_train', type=int, default=2000, help='training sequence length')
        dataset_parser.add_argument('--seq_len_test', type=int, default=None, help='test sequence length')
        dataset_parser.add_argument('--seq_len_val', type=int, default=2000, help='validation sequence length')  # 512
        dataset_options = dataset_parser.parse_args()

    elif dataset_name == 'toy_lgssm':
        dataset_parser = argparse.ArgumentParser(description='dynamic system parameter: lgssm')
        dataset_parser.add_argument('--y_dim', type=int, default=1, help='dimension of y')
        dataset_parser.add_argument('--u_dim', type=int, default=1, help='dimension of u')
        dataset_parser.add_argument('--seq_len_train', type=int, default=64, help='training sequence length')
        dataset_parser.add_argument('--seq_len_test', type=int, default=None, help='test sequence length')
        dataset_parser.add_argument('--seq_len_val', type=int, default=64, help='validation sequence length')  # 512
        dataset_options = dataset_parser.parse_args()

    elif dataset_name == 'wiener_hammerstein':
        dataset_parser = argparse.ArgumentParser(description='dynamic system parameter: wiener hammerstein')
        dataset_parser.add_argument('--y_dim', type=int, default=1, help='dimension of y')
        dataset_parser.add_argument('--u_dim', type=int, default=1, help='dimension of u')
        dataset_parser.add_argument('--seq_len_train', type=int, default=2048, help='training sequence length')
        dataset_parser.add_argument('--seq_len_test', type=int, default=None, help='test sequence length')
        dataset_parser.add_argument('--seq_len_val', type=int, default=2048, help='validation sequence length')
        dataset_options = dataset_parser.parse_args()

    return dataset_options
