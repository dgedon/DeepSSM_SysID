# import generic libraries
import torch
import numpy as np
# import user-writte files
import utils.datavisualizer as dv
import utils.dataevaluater as de
from utils.utils import get_n_params
from models.model_state import ModelState
from utils.utils import compute_normalizer


def run_test(options, loaders, df, path_general, file_name_general, **kwargs):
    # switch to cpu computations for testing
    options['device'] = 'cpu'

    # %% load model

    # Compute normalizers (here just used for initialization, real values loaded below)
    if options["normalize"]:
        normalizer_input, normalizer_output = compute_normalizer(loaders['test'])
    else:
        normalizer_input = normalizer_output = None

    # Define model
    modelstate = ModelState(seed=options["seed"],
                            nu=loaders["train"].nu, ny=loaders["train"].ny,
                            model=options["model"],
                            options=options,
                            normalizer_input=normalizer_input,
                            normalizer_output=normalizer_output)
    modelstate.model.to(options['device'])

    # load model
    path = path_general + 'model/'
    file_name = file_name_general + '_bestModel.ckpt'
    modelstate.load_model(path, file_name)
    modelstate.model.to(options['device'])

    # %% plot and save the loss curve
    dv.plot_losscurve(df, options, path_general, file_name_general)

    # %% others

    if bool(kwargs):
        file_name_add = kwargs['file_name_add']
    else:
        # Default option
        file_name_add = ''
    file_name_general = file_name_add + file_name_general

    # get the number of model parameters
    num_model_param = get_n_params(modelstate.model)
    print('Model parameters: {}'.format(num_model_param))

    # %% RUN PERFORMANCE EVAL
    # %%

    # %% sample from the model
    for i, (u_test, y_test) in enumerate(loaders['test']):
        # getting output distribution parameter only implemented for selected models
        u_test = u_test.to(options['device'])
        y_sample, y_sample_mu, y_sample_sigma = modelstate.model.generate(u_test)

        # convert to numpy for evaluation
        # samples data
        y_sample_mu = y_sample_mu.detach().numpy()
        y_sample_sigma = y_sample_sigma.detach().numpy()
        # test data
        y_test = y_test.detach().numpy()
        y_sample = y_sample.detach().numpy()

    # get noisy test data for narendra_li
    if options['dataset'] == 'narendra_li':
        # original test set is unnoisy -> get noisy test set
        yshape = y_test.shape
        y_test_noisy = y_test + np.sqrt(0.1) * np.random.randn(yshape[0], yshape[1], yshape[2])
    elif options['dataset'] == 'toy_lgssm':
        # original test set is unnoisy -> get noisy test set
        yshape = y_test.shape
        y_test_noisy = y_test + np.sqrt(1) * np.random.randn(yshape[0], yshape[1], yshape[2])
    else:
        y_test_noisy = y_test

    # %% plot resulting predictions
    if options['dataset'] == 'narendra_li':
        # for narendra_li problem show test data mean pm 3sigma as well
        data_y_true = [y_test, np.sqrt(0.1) * np.ones_like(y_test)]
        data_y_sample = [y_sample_mu, y_sample_sigma]
        label_y = ['true, $\mu\pm3\sigma$', 'sample, $\mu\pm3\sigma$']
    elif options['dataset'] == 'toy_lgssm':
        # for lgssm problem show test data mean pm 3sigma as well
        data_y_true = [y_test, np.sqrt(1) * np.ones_like(y_test)]
        data_y_sample = [y_sample_mu, y_sample_sigma]
        label_y = ['true, $\mu\pm3\sigma$', 'sample, $\mu\pm3\sigma$']
    else:
        data_y_true = [y_test_noisy]
        data_y_sample = [y_sample_mu, y_sample_sigma]
        label_y = ['true', 'sample, $\mu\pm3\sigma$']
    if options['dataset'] == 'cascaded_tank':
        temp = 1024
    elif options['dataset'] == 'wiener_hammerstein':
        temp = 4000
    else:
        temp = 200
    dv.plot_time_sequence_uncertainty(data_y_true,
                                      data_y_sample,
                                      label_y,
                                      options,
                                      batch_show=0,
                                      x_limit_show=[0, temp],
                                      path_general=path_general,
                                      file_name_general=file_name_general)

    # %% compute performance values

    # compute marginal likelihood (same as for predictive distribution loss in training)
    marginal_likeli = de.compute_marginalLikelihood(y_test_noisy, y_sample_mu, y_sample_sigma, doprint=True)

    # compute VAF
    vaf = de.compute_vaf(y_test_noisy, y_sample_mu, doprint=True)

    # compute RMSE
    rmse = de.compute_rmse(y_test_noisy, y_sample_mu, doprint=True)

    # %% Collect data

    # options_dict
    options_dict = {'h_dim': options['model_options'].h_dim,
                    'z_dim': options['model_options'].z_dim,
                    'n_layers': options['model_options'].n_layers,
                    'seq_len_train': options['dataset_options'].seq_len_train,
                    'batch_size': options['train_options'].batch_size,
                    'lr_scheduler_nepochs': options['train_options'].lr_scheduler_nepochs,
                    'lr_scheduler_factor': options['train_options'].lr_scheduler_factor,
                    'model_param': num_model_param, }
    # test_dict
    test_dict = {'marginal_likeli': marginal_likeli,
                 'vaf': vaf,
                 'rmse': rmse}
    # dataframe
    df.update(options_dict)
    df.update(test_dict)

    return df
