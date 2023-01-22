import simulation as sim
import visualization as vis
import constants
import pickle
import numpy as np
import random
from multiprocessing import Pool
import time
import json
import re
import pandas as pd
import glob, os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.special as sc


def trunc_norm_sampling_vector(mu, sigma):
    n = len(mu)
    U = np.random.mtrand._rand.uniform(size=n)
    y = mu + sigma * sc.ndtri(U + sc.ndtr(-mu / sigma) * (1 - U))
    return y


pi_2D = True
constants.VISUALIZATION = 'visualization'


onLaptop = 'False'
if onLaptop=='False':
    constants.RESULTS = sys.argv[2]
    number = str(sys.argv[1])
    constants.RESULTS = constants.RESULTS + number
    config_file =  sys.argv[3]
    noise = str(sys.argv[4])
else:
    constants.RESULTS = 'Results_simulation_REALTEMP_'
    constants.RESULTS = constants.RESULTS + '1'
    number = '1'
    config_file = 'configs/'
    noise = '0'


if not os.path.exists(constants.RESULTS):
    os.makedirs(constants.RESULTS)
    print("Directory ", constants.RESULTS, " Created ")
else:
    print("Directory ", constants.RESULTS, " already exists")


# intialise
result = {'Name': [],
          'N status': [],
          'H_SEE': [],
          'phi_SEE': [],
          'pi_SEE': [],
          'PZ_SEE': [],
          'n_SEE': [],
          #'h_theta_SEE': [],
          'B_SEE': []
          }


result_df = pd.DataFrame(result)
# os.chdir("configs")
for file in glob.glob(config_file + "/*.json"):
    file_name = re.split("/|.json", file)[1]
    vis_1 = vis.visualization(constants.RESULTS + '/' + file_name + '_' + constants.VISUALIZATION)
    with open(file) as json_data_file:
        data = json.load(json_data_file)

    K = data['structure']['K']
    S = data['structure']['S']
    I = data['structure']['I']
    g = data['structure']['g']
    theta = data['structure']['theta']

    p_c_binom = data['C_variation']['p_c_binom']
    C_temp = data['C_variation']['C']
    repeat_temp = data['C_variation']['repeat']
    if C_temp is None:
        C = None
    else:
        C_t = []
        for ct in range(len(C_temp)):
            C_t.append(np.tile(C_temp[ct], (repeat_temp[ct], 1)))
        C = np.concatenate(C_t)
    vis_1.heatmap_seaborn(C, 'C_seaborn', 'clones', 'mutations', False, 0.5)


    sns.set_theme()
    ax = sns.heatmap(C, annot=False, linewidths=0.5)
    ax.set(xlabel='clones', ylabel='mutations')
    fig = ax.get_figure()
    fig.savefig(constants.RESULTS+'/C_seaborn_new' + '.png')
    plt.close()

    n_sampling = data['n_variation']['n_sampling']

    if data['n_variation']['n'] is not None:
        n = np.array(data['n_variation']['n'])
    else:
        n = None

    Z = data['Z_variation']['Z']
    avarage_clone_in_spot = data['Z_variation']['avarage_clone_in_spot']

    if onLaptop == 'True':
        max_iter = np.int(data['sampling']['max_iter']/10)
        min_iter = np.int(data['sampling']['min_iter']/10)
        # burn_in = np.int(data['sampling']['burn_in']/10)
        batch = np.int(data['sampling']['batch']/10)
    else:
        max_iter = np.int(data['sampling']['max_iter'])
        min_iter = np.int(data['sampling']['min_iter'])
        # burn_in = np.int(data['sampling']['burn_in'])
        batch = np.int(data['sampling']['batch'])

    phi_gamma = np.array(data['Gamma']['phi_gamma'])
        # F could be None, In that case, it will be generated using dirichlet distribution
    F_epsilon = np.tile(data['Gamma']['F_epsilon'], (K, 1))
    F_fraction =  data['Gamma']['F_fraction']

    F = np.tile(data['Gamma']['F'], (K, 1))  # np.array([[9,2],[9,2],[9,2],[9,2],[9,2],[9,2]])

    n_lambda = np.tile(data['n_variation']['n_lambda'], (S))

    p_mean = data['Y_variation']['p_mean']
    p_std = data['Y_variation']['p_std']
    p_y = trunc_norm_sampling_vector(np.array([p_mean]*g), p_std)

    b_alpha_shape = data['Y_variation']['b_alpha_shape']
    b_alpha_var = data['Y_variation']['b_alpha_var']
    b_alpha_scale = np.sqrt(b_alpha_var / b_alpha_shape)
    b_alpha = np.random.gamma(b_alpha_shape, b_alpha_scale, g)

    b_beta = data['Y_variation']['b_beta']

    if 'mean_read' in data['Gamma']:
        while True:
            sample_1 = sim.simulation(K=K, S=S, g=g, r=phi_gamma[0], q=phi_gamma[1], I=I, F=F, D=None, A=None, C=C,
                                      avarage_clone_in_spot=avarage_clone_in_spot, random_seed=random.randint(1,100), 
                                      F_epsilon= F_epsilon, n=n, p_c_binom=p_c_binom, theta=theta, Z = Z, n_lambda=n_lambda, 
                                      F_fraction=F_fraction, pi_2D=pi_2D, Y=None, p_y=p_y, b_alpha=b_alpha, b_beta=b_beta, 
                                      b_alpha_shape=b_alpha_shape, b_alpha_scale=b_alpha_scale)
            lb = np.floor(int(data['Gamma']['mean_read']) * 0.9)
            ub = np.ceil(int(data['Gamma']['mean_read']) * 1.1)
            print(np.mean(np.sum(sample_1.D, axis=0)))
            if np.mean(np.sum(sample_1.D, axis=0)) > lb and np.mean(np.sum(sample_1.D, axis=0)) < ub:
                break
    else:
        sample_1 = sim.simulation(K=K, S=S, g=g, r=phi_gamma[0], q=phi_gamma[1], I=I, F=F, D=None, A=None, C=C,
                                  avarage_clone_in_spot=avarage_clone_in_spot, random_seed=random.randint(1,100), 
                                  F_epsilon= F_epsilon, n=n, p_c_binom=p_c_binom, theta=theta, Z = Z, n_lambda=n_lambda, 
                                  F_fraction=F_fraction, pi_2D=pi_2D, Y=None, p_y=p_y, b_alpha=b_alpha, b_beta=b_beta, 
                                  b_alpha_shape=b_alpha_shape, b_alpha_scale=b_alpha_scale)

    pickle.dump(sample_1, open(constants.RESULTS + '/sample_' + file_name, 'wb'))

