"""
67800 - Probabilistic Methods in AI
Spring 2018/19
Programming Assignment 1 - Bayesian Networks
(Complete the missing parts (TODO))
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from scipy.io import loadmat
# from scipy.misc import logsumexp
from scipy.special import logsumexp
import pandas as pd

Z_SIZE = 25
I_SIZE = 784


def get_p_z1(z1_val):
    '''
    Get the prior probability for variable Z1 to take value z1_val.
    Note: You are also welcome to access bayes_net['prior_z1'] directly...
    '''
    return bayes_net['prior_z1'][z1_val]


def get_p_z2(z2_val):
    '''
    Get the prior probability for variable Z2 to take value z2_val.
    '''
    return bayes_net['prior_z2'][z2_val]


def get_p_x_cond_z1_z2(z1_val, z2_val):
    '''
    Get the conditional probabilities of variables X_1 to X_784 to take the value 1 given z1 = z1_val and z2 = z2_val.
    '''
    return bayes_net['cond_likelihood'][(z1_val, z2_val)]


def get_pixels_sampled_from_p_x_joint_z1_z2():
    '''
    The function should return the sampled values of pixel variables (array of length 784)
    '''
    L = 25
    df_z1 = pd.DataFrame.from_dict(bayes_net['prior_z1'], 'index')
    df_z2 = pd.DataFrame.from_dict(bayes_net['prior_z2'], 'index')
    probs = np.multiply.outer(df_z1.values[:, 0], df_z2.values[:, 0])
    chosen_ix = np.random.choice(np.arange(probs.size), p=probs.flatten())
    i_ix = chosen_ix // L
    j_ix = chosen_ix % L
    i_ = df_z2.index[i_ix]
    j_ = df_z1.index[j_ix]
    P_X = get_p_x_cond_z1_z2(j_, i_)
    f = lambda p: np.random.choice([1, 0], p=[p[0], 1 - p[0]])
    X = np.apply_along_axis(f, 0, P_X)
    return X


def get_expectation_x_cond_z1_z2(z1_val, z2_val):
    '''
    This function calculates E(X1:784 | z1, z2): expectation of X1...784 w.r.t the conditional probability
    '''
    return get_p_x_cond_z1_z2(z1_val, z2_val)



def q_4():
    '''
    Plots the pixel variables sampled from the joint distribution as 28 x 28 images.
    Your job is to implement get_pixels_sampled_from_p_x_joint_z1_z2.
    '''
    plt.figure()
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(get_pixels_sampled_from_p_x_joint_z1_z2().reshape(28, 28), cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('q_4', bbox_inches='tight')
    plt.show()
    plt.close()


def q_5():
    '''
    Plots the expected images for each latent configuration on a 2D grid.
    Your job is to implement get_expectation_x_cond_z1_z2.
    '''

    canvas = np.empty((28 * len(z1_vals), 28 * len(z1_vals)))
    for i, z1_val in enumerate(z1_vals):
        for j, z2_val in enumerate(z2_vals):
            canvas[(len(z1_vals) - i - 1) * 28:(len(z1_vals) - i) * 28, j * 28:(j + 1) * 28] = \
                get_expectation_x_cond_z1_z2(z1_val, z2_val).reshape(28, 28)

    plt.figure()
    plt.axis('off')
    plt.imshow(canvas, cmap='gray')
    plt.tight_layout()
    plt.savefig('q_5', bbox_inches='tight')
    plt.show()
    plt.close()


def handle_tiny_numbers():
    float_formatter = lambda x: "%.5f" % x
    np.set_printoptions(formatter={'float_kind': float_formatter})


def get_log_likelihood(data):
    data = data.T
    _, C = data.shape
    log_val_data = np.zeros(C)
    for i, (z1, z2) in enumerate(bayes_net['cond_likelihood'].keys()):
        P_z1 = get_p_z1(z1)
        P_z2 = get_p_z2(z2)
        P_X = get_p_x_cond_z1_z2(z1, z2)
        curr_ps = np.tile(P_X, (C, 1)).T
        curr_ps[data == 0] -= np.nextafter(1, 0)
        curr_ps = np.abs(curr_ps)

        rep_z1 = np.repeat(P_z1, C)
        rep_z2 = np.repeat(P_z2, C)
        curr_ps = np.vstack((np.vstack((curr_ps, rep_z1)), rep_z2))
        curr_ps_sumed = np.sum(np.log(curr_ps), axis=0)
        log_val_data = np.vstack((log_val_data, curr_ps_sumed))
    return logsumexp(log_val_data[1:, :], axis=0)


def q_6():
    '''
    Loads the data and plots the histograms. Rest is TODO.
    Your job is to compute real_marginal_log_likelihood and corrupt_marginal_log_likelihood below.
    '''
    handle_tiny_numbers()
    mat = loadmat('q_6.mat')
    val_data = mat['val_x']
    test_data = mat['test_x']
    log_val_data = get_log_likelihood(val_data)
    avg_val = np.mean(log_val_data)
    std_val = np.std(log_val_data)
    thr = avg_val - 3 * std_val
    log_test_data = get_log_likelihood(test_data)
    corrupt_marginal_log_likelihood = log_test_data[log_test_data < thr]
    real_marginal_log_likelihood = log_test_data[log_test_data >= thr]

    plot_histogram(real_marginal_log_likelihood, title='Histogram of marginal log-likelihood for real test data',
                   xlabel='marginal log-likelihood', savefile='q_6_hist_real')

    plot_histogram(corrupt_marginal_log_likelihood,
                   title='Histogram of marginal log-likelihood for corrupted test data',
                   xlabel='marginal log-likelihood', savefile='q_6_hist_corrupt')

    plt.show()
    plt.close()


def get_px_cond_z1_z2_cuboid():
    """
    computes the conditional probability P(X|Z1,Z2)
    """
    px_data = np.swapaxes(np.array(list(bayes_net['cond_likelihood'].values())), 1, 2)[:, :, 0]
    px_keys = np.array(list(bayes_net['cond_likelihood'].keys()))
    idx_z1 = px_keys[:, 0]
    idx_z2 = px_keys[:, 1]
    mux = pd.MultiIndex.from_arrays([idx_z1, idx_z2], names=['z1', 'z2'])
    df_px = pd.DataFrame(px_data, index=mux)
    df_px.sort_values(['z1', 'z2'], axis=0, inplace=True)
    df_px_s = df_px.values
    p_cube = np.zeros((Z_SIZE, Z_SIZE, I_SIZE))
    for i in range(Z_SIZE):
        f = lambda x: Z_SIZE * x
        p_cube[i, :, :] = df_px_s[f(i):f(i + 1), :]
    z_idx = np.unique(df_px.index.get_level_values(0).values)
    return p_cube, z_idx


def get_p_joint_cuboid_for_data(data):
    """
    computes a cuboid with the joint P(X) distribution for a given dataset
    """
    N, D = data.shape
    data_cuboid = np.zeros((Z_SIZE, Z_SIZE, N))
    px_cuboid, z_idx = get_px_cond_z1_z2_cuboid()

    # create cuboid for the cond probabilities
    for i in range(N):
        print(i)
        img = data[i, :]
        tiled_img = np.tile(img, (Z_SIZE, Z_SIZE, 1))
        img_cuboid = np.copy(px_cuboid)
        np.putmask(img_cuboid, tiled_img == 0, 1 - img_cuboid)
        data_cuboid[:, :, i] = np.sum(np.log(img_cuboid), axis=2)
    # add the log probabilities of P(Z1), P(Z2)
    df_z1 = pd.DataFrame.from_dict(bayes_net['prior_z1'], 'index').sort_index(0)
    df_z2 = pd.DataFrame.from_dict(bayes_net['prior_z2'], 'index').sort_index(0)
    z_probs = np.add.outer(np.log(df_z1.values[:, 0]), np.log(df_z2.values[:, 0]))
    data_cuboid += np.moveaxis(np.tile(z_probs, (N, 1, 1)), 0, -1)
    return data_cuboid, z_idx


def get_px_for_data(data_cuboid):
    """
    get the marginal probability P(X)
    :param data_cuboid:
    :return:
    """
    return logsumexp(data_cuboid, axis=(0, 1))


def get_conditional_expectation(data):
    '''
    :param data: Row vectors of data points X (n x 784)
    :return: array of E(z1 | X = data), array of E(z2 | X = data)
    '''
    N, D = data.shape
    data_cuboid, z_idx = get_p_joint_cuboid_for_data(data)
    px = get_px_for_data(data_cuboid)

    # compute E(z1 | X = data)
    p_z1_x = logsumexp(data_cuboid, axis=1)
    p_z1_x -= np.tile(px, (Z_SIZE, 1))
    p_z1_given_x = np.exp(p_z1_x)
    p_z1_given_x = (p_z1_given_x.T * z_idx).T
    E_z1_X = np.sum(p_z1_given_x, axis=0)

    # compute E(z2 | X = data)
    p_z2_x = logsumexp(data_cuboid, axis=0)
    p_z2_x -= np.tile(px, (Z_SIZE, 1))
    p_z2_given_x = np.exp(p_z2_x)
    p_z2_given_x = (p_z2_given_x.T * z_idx).T
    E_z2_X = np.sum(p_z2_given_x, axis=0)

    return E_z1_X, E_z2_X


def q_7():
    '''
    Loads the data and plots a color coded clustering of the conditional expectations.
    Your job is to implement the get_conditional_expectation function
    '''

    mat = loadmat('q_7.mat')
    data = mat['x']
    labels = mat['y']

    mean_z1, mean_z2 = get_conditional_expectation(data)

    plt.figure()
    plt.scatter(mean_z1, mean_z2, c=np.squeeze(labels))
    plt.colorbar()
    plt.grid()
    plt.savefig('q_7', bbox_inches='tight')
    plt.show()
    plt.close()


def load_model(model_file):
    '''
    Loads a default Bayesian network with latent variables (in this case, a variational autoencoder)
    '''

    with open(model_file + '.pkl', 'rb') as infile:
        cpts = pkl.load(infile, encoding='bytes')

    model = {}
    model['prior_z1'] = cpts[0]
    model['prior_z2'] = cpts[1]
    model['cond_likelihood'] = cpts[2]

    return model


def plot_histogram(data, title='histogram', xlabel='value', ylabel='frequency', savefile='hist'):
    '''
    Plots a histogram.
    '''

    plt.figure()
    plt.hist(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(savefile, bbox_inches='tight')


def main():
    global bayes_net, z1_vals, z2_vals
    bayes_net = load_model('trained_mnist_model')
    z1_vals = sorted(bayes_net['prior_z1'].keys())
    z2_vals = sorted(bayes_net['prior_z2'].keys())
    '''
    TODO: Using the above Bayesian Network model, complete the following parts.
    '''
    q_4()
    q_5()
    q_6()
    q_7()


if __name__ == '__main__':
    main()
