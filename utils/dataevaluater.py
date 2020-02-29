import numpy as np
import torch
import torch.distributions as tdist


# computes the VAF (variance accounted for)
def compute_vaf(y, yhat, doprint=False):
    # reshape to ydim x -1
    num_outputs = y.shape[1]
    y = y.transpose(1, 0, 2)
    y = y.reshape(num_outputs, -1)
    yhat = yhat.transpose(1, 0, 2)
    yhat = yhat.reshape(num_outputs, -1)

    diff = y - yhat
    num = np.mean(np.linalg.norm(diff, axis=0) ** 2)
    den = np.mean(np.linalg.norm(y, axis=0) ** 2)
    vaf = 1 - num/den
    vaf = max(0, vaf*100)

    """# new method
    num = 0
    den = 0
    for k in range(y.shape[-1]):
        norm2_1 = (np.linalg.norm(y[:, k] - yhat[:, k])) ** 2
        num = num + norm2_1
        norm2_2 = (np.linalg.norm(y[:, k])) ** 2
        den = den + norm2_2
    vaf = max(0, (1 - num / den) * 100)"""

    # print output
    if doprint:
        print('VAF = {:.3f}%'.format(vaf))

    return vaf


# computes the RMSE for all outputs
def compute_rmse(y, yhat, doprint=False):
    # get sizes from data
    num_outputs = y.shape[1]

    # reshape to ydim x -1
    y = y.transpose(1, 0, 2)
    y = y.reshape(num_outputs, -1)
    yhat = yhat.transpose(1, 0, 2)
    yhat = yhat.reshape(num_outputs, -1)

    rmse = np.zeros([num_outputs])
    for i in range(num_outputs):
        rmse[i] = np.sqrt(((yhat[i, :] - y[i, :]) ** 2).mean())

    # print output
    if doprint:
        for i in range(num_outputs):
            print('RMSE y{} = {:.3f}'.format(i + 1, rmse[i]))

    return rmse


# computes the marginal likelihood of all outputs
def compute_marginalLikelihood(y, yhat_mu, yhat_sigma, doprint=False):
    # to torch
    y = torch.tensor(y, dtype=torch.double)
    yhat_mu = torch.tensor(yhat_mu, dtype=torch.double)
    yhat_sigma = torch.tensor(yhat_sigma, dtype=torch.double)

    # number of batches
    num_batches = y.shape[0]
    num_points = np.prod(y.shape)

    # get predictive distribution
    pred_dist = tdist.Normal(yhat_mu, yhat_sigma)

    # get marginal likelihood
    marg_likelihood = torch.mean(pred_dist.log_prob(y))
    # to numpy
    marg_likelihood = marg_likelihood.numpy()

    # print output
    if doprint:
        print('Marginal Likelihood / point = {:.3f}'.format(marg_likelihood))

    return marg_likelihood
