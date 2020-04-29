import numpy as np
import numpy.random as npr 
import matplotlib.pyplot as plt
from data_generation import generate_data, input_and_target
import ssm
from plot import plot_most_likely_dynamics, plot_trajectory

# ENV 
npr.seed(2305)

# model parameters 
K = 2
D_obs = 3
D_latent = 2

def fit_slds(input : np.ndarray, target : np.ndarray): 
    global K, D_obs, D_latent

    rslds = ssm.SLDS(D_obs, K, D_latent,
                transitions="recurrent_only",
                dynamics="diagonal_gaussian",
                emissions="gaussian_orthog",
                single_subspace=True)

    rslds.initialize(input)

    q_elbos_lem, q_lem = rslds.fit(input, method="laplace_em",
                                variational_posterior="structured_meanfield",
                                initialize=False, num_iters=50, alpha=0.0)
    
    xhat_lem = q_lem.mean_continuous_states[0]
    zhat_lem = rslds.most_likely_states(xhat_lem, input)

    # Smooth the data under the variational posterior
    yhat_lem = rslds.smooth(xhat_lem, input)

    # compute the posterior over latent and continuous states for the targets 
    target_elbos, target_posterior = rslds.approximate_posterior(target,
                                              method="laplace_em",
                                              variational_posterior="structured_meanfield",
                                              num_iters=10)
                                              
    # Get the posterior mean of the continuous states
    target_posterior_x = target_posterior.mean_continuous_states[0]
    target_posterior_y = rslds.smooth(target_posterior_x, target)
    

    return {'training_elbos': q_elbos_lem, 'input_xhat': xhat_lem, 'input_zhat': zhat_lem, 'target_elbos': target_elbos, 'target_posterior': target_posterior_y}


if __name__=="__main__": 
    # sample from Lorentz system
    batch_size = 10
    t_steps = 1000
    data = generate_data(batch_size,t_steps)
    inputs, targets = input_and_target(data)
    
    # fit an slsd model 
    res = fit_slds(inputs[0], targets[0])

    fig, axs = plt.subplots(nrows=3, ncols=1)
    axs[0].plot(res['training_elbos'], label='Training ELBO') 
    axs[0].set_xlabel('iteration')
    axs[0].set_ylabel('ELBO')
    
    axs[1].plot(res['target_elbos'], label='Prediction ELBO')
    axs[1].set_xlabel('iteration')
    axs[1].set_ylabel('ELBO')

    axs[2].plot(targets[0][:, 0], c='b', label='true target')
    axs[2].plot(targets[0][:, 1:], c='b')
    axs[2].plot(res['target_posterior'][:, 0], linestyle='dotted', c='r', alpha=0.9, label='model posterior')
    axs[2].plot(res['target_posterior'][:, 1:], linestyle='dotted', c='r', alpha=0.9)
    axs[2].set_xlabel('time')

    plt.legend()
    plt.tight_layout() 
    plt.show() 



# # Plot some results
# plt.figure()
# plt.plot(q_elbos_svi, label="SVI")
# plt.plot(q_elbos_lem[1:], label="Laplace-EM")
# plt.legend()
# plt.xlabel("Iteration")
# plt.ylabel("ELBO")
# plt.tight_layout()

# plt.figure(figsize=[10,4])
# ax1 = plt.subplot(131)
# plot_trajectory(z, x, ax=ax1)
# plt.title("True")
# ax2 = plt.subplot(132)
# plot_trajectory(zhat_svi, xhat_svi, ax=ax2)
# plt.title("Inferred, SVI")
# ax3 = plt.subplot(133)
# plot_trajectory(zhat_lem, xhat_lem, ax=ax3)
# plt.title("Inferred, Laplace-EM")
# plt.tight_layout()

# plt.figure(figsize=(6,6))
# ax = plt.subplot(111)
# lim = abs(xhat_lem).max(axis=0) + 1
# plot_most_likely_dynamics(rslds_lem, xlim=(-lim[0], lim[0]), ylim=(-lim[1], lim[1]), ax=ax)
# plt.title("Most Likely Dynamics, Laplace-EM")

# plt.show()