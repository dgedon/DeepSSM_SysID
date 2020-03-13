import matplotlib.pyplot as plt
import torch
import numpy as np
import os


# %% plots the resulting time sequence
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
    plt.figure(1, figsize=(5 * num_cols, 5 * num_outputs))

    # plot outputs
    for j in range(0, num_outputs):
        # output yk
        plt.subplot(num_outputs, num_cols, num_cols * (j + 1))
        if len(data_y_true) == 1:  # plot samples
            plt.plot(data_y_true[0][batch_show, j, :].squeeze(), label='y_{}(k) {}'.format(j + 1, label_y[0]))
        else:  # plot true mu /pm 3sigma
            length = len(data_y_true[0][batch_show, j, :])
            x = np.linspace(0, length - 1, length)
            mu = data_y_true[0][batch_show, j, :].squeeze()
            std = data_y_true[1][batch_show, j, :].squeeze()
            # plot mean
            plt.plot(mu, label='y_{}(k) {}'.format(j + 1, label_y[0]))
            # plot 3std around
            plt.fill_between(x, mu, mu + 3 * std, alpha=0.3, facecolor='b')
            plt.fill_between(x, mu, mu - 3 * std, alpha=0.3, facecolor='b')

        # plot samples mu \pm 3sigma
        length = len(data_y_sample[0][batch_show, j, :])
        x = np.linspace(0, length - 1, length)
        mu = data_y_sample[0][batch_show, j, :].squeeze()
        std = data_y_sample[1][batch_show, j, :].squeeze()

        # plot mean
        plt.plot(mu, label='y_{}(k) {}'.format(j + 1, label_y[1]))
        # plot 3std around
        plt.fill_between(x, mu, mu + 3 * std, alpha=0.3, facecolor='r')
        plt.fill_between(x, mu, mu - 3 * std, alpha=0.3, facecolor='r')

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
    plt.close(1)


# %% plot and save the loss curve
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
        plt.figure(1, figsize=(5, 5))
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
        plt.yscale('log')
        # save model
        if options['savefig']:
            # check if path exists and create otherwise
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(path + file_name + '.png', format='png')
        # show the model
        if options['showfig']:
            plt.show()
        plt.close(1)

        # delete loss value matrices from dictionary
        if removedata:
            del df['all_losses']
            del df['all_vlosses']

    return df


# %% plot performance over number of training points
def plot_perf_ndata(k_max_train_values, all_vaf, all_rmse, all_likelihood, options, path_general):
    # plot the stuff
    plt.figure(1, figsize=(5 * 1, 5 * 3))
    x = k_max_train_values

    # vaf
    mu = all_vaf.mean(0).squeeze().numpy()
    std = np.sqrt(all_vaf.var(0)).squeeze().numpy()
    plt.subplot(3, 1, 1)
    # plot mean
    plt.plot(x, mu, label='VAF $\mu\pm\sigma$')
    # plot std
    plt.fill_between(x, mu, mu + std, alpha=0.3, facecolor='b')
    plt.fill_between(x, mu, mu - std, alpha=0.3, facecolor='b')
    # plot settings
    plt.title('VAF of {}'.format(options['dataset']))
    plt.xlabel('Training Datapoints')
    plt.ylabel('VAF [%]')
    plt.legend()

    # rmse
    mu = all_rmse.mean(0).squeeze().numpy()
    std = np.sqrt(all_rmse.var(0)).squeeze().numpy()
    plt.subplot(3, 1, 2)
    # plot mean
    plt.plot(x, mu, label='RMSE $\mu\pm\sigma$')
    # plot std
    plt.fill_between(x, mu, mu + std, alpha=0.3, facecolor='b')
    plt.fill_between(x, mu, mu - std, alpha=0.3, facecolor='b')
    # plot settings
    plt.title('RMSE of {}'.format(options['dataset']))
    plt.xlabel('Training Datapoints')
    plt.ylabel('RMSE [-]')
    plt.legend()

    # marg. likelihood
    mu = -all_likelihood.mean(0).squeeze().numpy()
    std = np.sqrt((-all_likelihood).var(0)).squeeze().numpy()
    plt.subplot(3, 1, 3)
    # plot mean
    plt.plot(x, mu, label='NLL $\mu\pm\sigma$')
    # plot std
    plt.fill_between(x, mu, mu + std, alpha=0.3, facecolor='b')
    plt.fill_between(x, mu, mu - std, alpha=0.3, facecolor='b')
    # plot settings
    plt.title('NLL of {}'.format(options['dataset']))
    plt.xlabel('Training Datapoints')
    plt.ylabel('p(yhat) [-]')
    plt.legend()

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
    plt.close(1)


# %% plot performance over number of training points
def plot_perf_sizes(h_values, all_vaf, all_rmse, all_likelihood, options, path_general):
    plt.figure(1, figsize=(5 * 1, 5 * 3))
    x = h_values

    # vaf
    mu = all_vaf.mean(0).squeeze().numpy()
    std = np.sqrt(all_vaf.var(0)).squeeze().numpy()
    plt.subplot(3, 1, 1)
    # plot mean
    plt.plot(x, mu, label='VAF $\mu\pm\sigma$')
    # plot std
    plt.fill_between(x, mu, mu + std, alpha=0.3, facecolor='b')
    plt.fill_between(x, mu, mu - std, alpha=0.3, facecolor='b')
    # plot settings
    plt.title('VAF of {}'.format(options['dataset']))
    plt.xlabel('Training Datapoints')
    plt.ylabel('VAF [%]')
    plt.legend()

    # rmse
    mu = all_rmse.mean(0).squeeze().numpy()
    std = np.sqrt(all_rmse.var(0)).squeeze().numpy()
    plt.subplot(3, 1, 2)
    # plot mean
    plt.plot(x, mu, label='RMSE $\mu\pm\sigma$')
    # plot std
    plt.fill_between(x, mu, mu + std, alpha=0.3, facecolor='b')
    plt.fill_between(x, mu, mu - std, alpha=0.3, facecolor='b')
    # plot settings
    plt.title('RMSE of {}'.format(options['dataset']))
    plt.xlabel('Training Datapoints')
    plt.ylabel('RMSE [-]')
    plt.legend()

    # marg. likelihood
    mu = -all_likelihood.mean(0).squeeze().numpy()
    std = np.sqrt((-all_likelihood).var(0)).squeeze().numpy()
    plt.subplot(3, 1, 3)
    # plot mean
    plt.plot(x, mu, label='NLL $\mu\pm\sigma$')
    # plot std
    plt.fill_between(x, mu, mu + std, alpha=0.3, facecolor='b')
    plt.fill_between(x, mu, mu - std, alpha=0.3, facecolor='b')
    # plot settings
    plt.title('NLL of {}'.format(options['dataset']))
    plt.xlabel('Training Datapoints')
    plt.ylabel('p(yhat) [-]')
    plt.legend()

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
    plt.close(1)


# %% old: For gridsearch, plots the resulting matrix
def plot_perf_gridsearch(all_vaf, all_rmse, all_likelihood, z_values, h_values, path_general, options):
    plt.figure(1, figsize=(5 * 3, 5 * 1))

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
    plt.close(1)
