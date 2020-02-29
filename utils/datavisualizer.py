import matplotlib.pyplot as plt
import torch
import numpy as np
import os


# plots the resulting time sequence
def plot_time_sequence(data_y, label_y, batch, x_limit_show, options):
    # get number of data sets
    num_data = len(data_y)

    # get number of outputs
    num_outputs = data_y[0].shape[1]

    # get number of columns
    num_cols = 1

    # initialize figure
    plt.figure(figsize=(5 * num_cols, 5 * num_outputs))

    # plot outputs
    for j in range(0, num_outputs):
        # output yk
        plt.subplot(num_outputs, num_cols, num_cols * (j + 1))
        for i in range(0, num_data):
            plt.plot(data_y[i][batch, j, :].squeeze(), label='y_{}(k) {}'.format(j + 1, label_y[i]))
        plt.title('Output $y_{}(k)$'.format(j + 1))
        plt.ylabel('$y_{}(k)$'.format(j + 1))
        plt.xlabel('time steps $k$')
        plt.legend()
        plt.xlim(x_limit_show)

    if options['showfig']:
        plt.show()


# plots the resulting time sequence
def plot_time_sequence_uncertainty(data_y_true, data_y_sample, label_y, options, path_general, file_name_general,
                                   batch_show, x_limit_show):
    # storage path
    file_name = file_name_general + '_timeEval'
    path = path_general + 'timeEval/'

    # get number of outputs
    num_outputs = data_y_sample[-1].shape[1]

    # get number of columns
    num_cols = 1

    # initialize figure
    plt.figure(figsize=(5 * num_cols, 5 * num_outputs))

    # plot outputs
    for j in range(0, num_outputs):
        # output yk
        plt.subplot(num_outputs, num_cols, num_cols * (j + 1))
        if len(data_y_true) == 1:  # plot samples
            plt.plot(data_y_true[0][batch_show, j, :].squeeze(), label='y_{}(k) {}'.format(j + 1, label_y[0]))
        else:  # plot true mu /pm 3sigma
            length = len(data_y_true[0][batch_show, j, :])
            x = np.linspace(0, length - 1, length)
            mean = data_y_true[0][batch_show, j, :].squeeze()
            std = data_y_true[1][batch_show, j, :].squeeze()
            # plot mean
            plt.plot(mean, label='y_{}(k) {}'.format(j + 1, label_y[0]))
            # plot 3std around
            plt.fill_between(x, mean, mean + 3 * std, alpha=0.3, facecolor='b')
            plt.fill_between(x, mean, mean - 3 * std, alpha=0.3, facecolor='b')

        # plot samples mu \pm 3sigma
        length = len(data_y_sample[0][batch_show, j, :])
        x = np.linspace(0, length - 1, length)
        mean = data_y_sample[0][batch_show, j, :].squeeze()
        std = data_y_sample[1][batch_show, j, :].squeeze()

        # plot mean
        plt.plot(mean, label='y_{}(k) {}'.format(j + 1, label_y[1]))
        # plot 3std around
        plt.fill_between(x, mean, mean + 3 * std, alpha=0.3, facecolor='r')
        plt.fill_between(x, mean, mean - 3 * std, alpha=0.3, facecolor='r')

        # plot settings
        plt.title('Output $y_{}(k)$, {} with (h,z,n)=({},{},{})'.format((j + 1),
                                                                        options['dataset'],
                                                                        options['model_options'].h_dim,
                                                                        options['model_options'].z_dim,
                                                                        options['model_options'].n_layers))
        plt.ylabel('$y_{}(k)$'.format(j + 1))
        plt.xlabel('time steps $k$')
        plt.legend()
        plt.xlim(x_limit_show)

    # save figure
    if options['savefig']:
        # check if path exists and create otherwise
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path + file_name + '.png', format='png')
    # plot model
    if options['showfig']:
        plt.show()


# plot resulting sample distribution
def plot_sample_distribution(data_y, label_y):
    # get number of data sets
    num_data = len(data_y)

    # get number of outputs
    num_outputs = data_y[0].shape[1]

    # get number of columns
    num_cols = 1

    # initialize figure
    plt.figure(figsize=(5 * num_cols, 5 * num_outputs))

    # plot distribution
    for j in range(0, num_outputs):
        plt.subplot(num_outputs, num_cols, num_cols * (j + 1))

        """maxval = -np.inf * np.ones(num_outputs)
        # get boundaries
        for cnt in range(0, num_outputs):
            np.max([np.max(data_y[j][cnt, :]), maxval[cnt]])"""

        """bins = np.linspace(np.min([y_sample.detach().numpy()[i, :], y_test.detach().numpy()[i, :]]),
                           np.max([y_sample.detach().numpy()[i, :], y_test.detach().numpy()[i, :]]), 150)"""
        kwargs = dict(alpha=0.5, bins=150, density=True)

        for i in range(0, num_data):
            # 'flatten' out all batches for each output
            data_new = data_y[i].transpose(1, 0, 2).reshape(data_y[i].shape[1], -1)
            # get histogram
            plt.hist(data_new[j, :], **kwargs, label='y_{}(k) {}'.format(j + 1, label_y[i]))  # hist_data_y =
        plt.title('Sample distribution of $y_{}$ for all times'.format(j + 1))
        plt.ylabel('distribution $p(y_{})$'.format(j + 1))
        plt.xlabel('sample value $y_{}$'.format(j + 1))
        plt.legend()
    plt.show()
    # print(kl(y_test.detach().numpy(), y_sample.detach().numpy()))


# plot and save the loss curve
def plot_losscurve(df, options, path_general, file_name_general, removedata=True):
    # only if df has values
    if 'all_losses' in df:
        # storage path
        file_name = file_name_general + '_loss'
        path = path_general + '/loss/'

        # get data to plot loss curve
        all_losses = df['all_losses']
        all_vlosses = df['all_vlosses']
        time_el = df['train_time']

        # plot loss curve
        plt.figure(figsize=(5, 5))
        xval = np.linspace(0, options['train_options'].test_every * (len(all_losses) - 1), len(all_losses))
        plt.plot(xval, all_losses, label='Training set')
        plt.plot(xval, all_vlosses, label='Validation set')  # loss_test_store_idx,
        plt.xlabel('Number Epochs in {:2.0f}:{:2.0f} [min:sec]'.format(time_el // 60,
                                                                       time_el - 60 * (time_el // 60)))
        plt.ylabel('Loss')
        plt.title('Loss of {} with (h,z,n)=({},{},{})'.format(options['dataset'],
                                                              options['model_options'].h_dim,
                                                              options['model_options'].z_dim,
                                                              options['model_options'].n_layers))
        plt.legend()
        # save model
        if options['savefig']:
            # check if path exists and create otherwise
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(path + file_name + '.png', format='png')
        # show the model
        if options['showfig']:
            plt.show()

        # delete loss value matrices from dictionary
        if removedata:
            del df['all_losses']
            del df['all_vlosses']

    return df


# plot performance over number of training points
def plot_perf_varynumdata(k_max_train_values, all_vaf, all_rmse, all_likelihood, options, path_general):
    # plot the stuff
    plt.figure(figsize=(5 * 1, 5 * 3))

    # vaf
    plt.subplot(3, 1, 1)
    plt.plot(k_max_train_values, all_vaf)
    plt.title('VAF of {}'.format(options['dataset']))
    plt.xlabel('Training Datapoints')
    plt.ylabel('VAF [%]')

    # rmse
    plt.subplot(3, 1, 2)
    plt.plot(k_max_train_values, all_rmse)
    plt.title('RMSE of {}'.format(options['dataset']))
    plt.xlabel('Training Datapoints')
    plt.ylabel('RMSE [-]')

    # marg. likelihood
    plt.subplot(3, 1, 3)
    plt.plot(k_max_train_values, all_likelihood)
    plt.title('Marg. Likelihood of {}'.format(options['dataset']))
    plt.xlabel('Training Datapoints')
    plt.ylabel('p(yhat) [-]')

    # save figure
    if options['savefig']:
        path = path_general + 'performance/'
        file_name = 'performanceEval'
        # check if path exists and create otherwise
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path + file_name + '.png', format='png')

    # show figure
    if options['showfig']:
        plt.show()

def plot_perf_gridsearch(all_vaf, all_rmse, all_likelihood, z_values, h_values, path_general, options):

    plt.figure(figsize=(5 * 3, 5 * 1))

    # plot VAF
    plt.subplot(1, 3, 1)
    plt.imshow(all_vaf)
    plt.colorbar()
    plt.xticks(np.arange(len(z_values)), np.asarray(z_values))
    plt.yticks(np.arange(len(h_values)), np.asarray(h_values))
    plt.title('VAF gridsearch')
    plt.xlabel('z_value')
    plt.ylabel('h_value')
    # plt.show()

    # plot RMSE
    plt.subplot(1, 3, 2)
    plt.imshow(all_rmse)
    plt.colorbar()
    plt.xticks(np.arange(len(z_values)), np.asarray(z_values))
    plt.yticks(np.arange(len(h_values)), np.asarray(h_values))
    plt.title('RMSE gridsearch')
    plt.xlabel('z_value')
    plt.ylabel('h_value')
    # plt.show()

    # plot likelihood
    plt.subplot(1, 3, 3)
    plt.imshow(all_likelihood)
    plt.colorbar()
    plt.xticks(np.arange(len(z_values)), np.asarray(z_values))
    plt.yticks(np.arange(len(h_values)), np.asarray(h_values))
    plt.title('Likelihood gridsearch')
    plt.xlabel('z_value')
    plt.ylabel('h_value')
    plt.show()

    # saving path and file name
    path = path_general + 'Performance/'
    file_name = 'performanceEval'

    # save figure
    if options['savefig']:
        # check if path exists and create otherwise
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path + file_name + '.png', format='png')
    # plot model
    if options['showfig']:
        plt.show()