import dill
import jax
import jax.numpy as np
from jax.scipy.special import logsumexp
from tqdm import tqdm
import sys

jax.config.update('jax_platform_name', 'cpu')

hyper = dict()
hyper['max_iter'] = 100

def read_data(agent, key):
    with open(f'data/icb-syntdata/agent{agent}-key{key}.obj', 'rb') as f:
        data = dill.load(f)
        data_x = np.array(data['x'])
        data_a = np.array(data['a'])
        key = data['key']
    return data, data_x, data_a

def evaluate_results(data_rhox, res_rhox, data_betas, res_betas):
    r_error = np.abs(data_rhox - res_rhox).sum()
    beta_error = np.abs((data_betas - res_betas)).sum(axis=-1).mean()
    return r_error, beta_error

T = 500
A = 2
K = 8
alpha = 20
sigma = .10

### Stationary agent '0'

@jax.jit
def _likelihood(rhox, x, a):
    q = alpha * np.einsum('ij,j->i', x, rhox)
    return q[a] - logsumexp(q)
_likelihood = jax.vmap(_likelihood, (None, 0, 0))

@jax.jit
def likelihood(rhox, data_x, data_a):
    ll = _likelihood(rhox, data_x, data_a)
    return ll.sum() 

def fit_stationary_model(data_x, data_a):
    grad_likelihood = jax.grad(likelihood, argnums=0)
    grad_likelihood = jax.jit(grad_likelihood)

    res = dict()
    # initialize rho
    res['rhox'] = -jax.numpy.ones(K) / K
    grad_mnsq = np.zeros(K)
    # perform gradient descent
    for i in range(hyper['max_iter']):
        grad = grad_likelihood(res['rhox'], data_x, data_a)
        grad_mnsq = .1 * grad**2 + 0.9 * grad_mnsq
        res['rhox'] += 0.001 * grad / (np.sqrt(grad_mnsq) + 1e-8)

    res['rhos'] = np.array([res['rhox'] for t in range(T)])
    return res

print("Simulation results for the stationary agent")
r_errors = []
beta_errors = []
for key in range(5):
    data, data_x, data_a = read_data('0', key)
    res = fit_stationary_model(data_x, data_a)
    r_error, beta_error = evaluate_results(
        data['rhox'], res['rhox'], 
        np.array([data['rhox'] for t in range(T)]), res['rhos'])
    r_errors.append(r_error)
    beta_errors.append(beta_error)
r_errors = np.array(r_errors)
beta_errors = np.array(beta_errors)
print(f"Reward accuracy: {r_errors.mean():.3f} ({r_errors.std():.3f})")
print(f"Belief accuracy: {beta_errors.mean():.3f} ({beta_errors.std():.3f})")


### Linear agent '21'

@jax.jit
def get_linear_rhos(rhox, rho0):
    lambdas = (np.arange(T) / T)
    rhos = np.ones((T, K)) * lambdas[:, np.newaxis] * rhox
    rhos += np.ones((T, K)) * (1 - lambdas)[:, np.newaxis] * rho0
    return rhos

@jax.jit
def _likelihood(rhox, x, a):
    q = alpha * np.einsum('ij,j->i', x, rhox)
    return q[a] - logsumexp(q)
_likelihood = jax.vmap(_likelihood, (0, 0, 0))

@jax.jit
def likelihood(rhox, rho0, data_x, data_a):
    rhos = get_linear_rhos(rhox, rho0)
    ll = _likelihood(rhos, data_x, data_a)
    return ll.sum() 

def fit_linear_model(data_x, data_a):
    grad_likelihood = jax.grad(likelihood, argnums=0)
    grad_likelihood = jax.jit(grad_likelihood)

    res = dict()
    # initialize rhox
    res['rhox'] = -jax.numpy.ones(K) / K
    rho0 = -jax.numpy.ones(K) / K
    grad_mnsq = np.zeros(K)
    for i in range(hyper['max_iter']):
        grad = grad_likelihood(res['rhox'], rho0, data_x, data_a)
        grad_mnsq = .1 * grad**2 + 0.9 * grad_mnsq
        res['rhox'] += 0.001 * grad / (np.sqrt(grad_mnsq) + 1e-8)

    res['rhos'] = get_linear_rhos(res['rhox'], rho0)
    return res

print("Simulation results for the linear agent")
r_errors = []
beta_errors = []
for key in range(5):
    data, data_x, data_a = read_data('21', key)
    res = fit_linear_model(data_x, data_a)
    r_error, beta_error = evaluate_results(
        data['rhos'][-1], res['rhox'], 
        np.array(data['rhos']), res['rhos'])
    r_errors.append(r_error)
    beta_errors.append(beta_error)
r_errors = np.array(r_errors)
beta_errors = np.array(beta_errors)
print(f"Reward accuracy: {r_errors.mean():.3f} ({r_errors.std():.3f})")
print(f"Belief accuracy: {beta_errors.mean():.3f} ({beta_errors.std():.3f})")


### Stepping agent
# Unknonwn t* and penv, known p0

@jax.jit
def _likelihood(rhox, x, a):
    q = alpha * np.einsum('ij,j->i', x, rhox)
    return q[a] - logsumexp(q)
_likelihood = jax.vmap(_likelihood, (None, 0, 0))

@jax.jit
def likelihood(rhox, data_x, data_a):
    ll = _likelihood(rhox, data_x, data_a)
    return ll.sum() 

def _fit_stepping_model(t, data_x, data_a):
    grad_likelihood = jax.grad(likelihood, argnums=0)
    grad_likelihood = jax.jit(grad_likelihood)

    res = dict()
    # initialize rho1, rho2
    rho0 = -jax.numpy.ones(K) / K
    res['rhox'] = -jax.numpy.ones(K) / K
    res['t_star'] = t
    grad_mnsq = np.zeros(K)

    for i in range(hyper['max_iter']):
        grad = grad_likelihood(res['rhox'], data_x[t:, :, :], data_a[t:])
        grad_mnsq = .1 * grad**2 + 0.9 * grad_mnsq
        res['rhox'] += 0.001 * grad / (np.sqrt(grad_mnsq) + 1e-8)

    ll = likelihood(rho0, data_x[:t, :, :], data_a[:t])
    ll += likelihood(res['rhox'], data_x[t:, :, :], data_a[t:])
    
    rhos1 = np.array([rho0 for i in range(t)])
    rhos2 = np.array([res['rhox'] for i in range(T - t)])
    res['rhos'] = np.concatenate([rhos1, rhos2])
    return ll, res

def fit_stepping_model(data_x, data_a):
    lls = dict()
    res_dict = dict()
    # fit the models for all possible values of t_star
    for t in tqdm(range(1, T-1)):
        lls[t] , res_dict[t] = _fit_stepping_model(t, data_x, data_a)
    # return the model with the highest likelihood
    t_star = max(lls, key=lls.get)
    return res_dict[t_star]

print("Simulation results for the stepping agent")
r_errors = []
beta_errors = []
for key in range(5):
    data, data_x, data_a = read_data('2', key)
    res = fit_stepping_model(data_x, data_a)
    r_error, beta_error = evaluate_results(
        data['rhos'][-1], res['rhox'], 
        np.array(data['rhos']), res['rhos'])
    r_errors.append(r_error)
    beta_errors.append(beta_error)
r_errors = np.array(r_errors)
beta_errors = np.array(beta_errors)
print(f"Reward accuracy: {r_errors.mean():.3f} ({r_errors.std():.3f})")
print(f"Belief accuracy: {beta_errors.mean():.3f} ({beta_errors.std():.3f})")


### Regressing agent, unknown t*, gamma and penv, known p0

def _get_linear_rhos(rhox, T, rho0):
    lambdas = (np.arange(T) / T)
    rhos = np.ones((T, K)) * lambdas[:, np.newaxis] * rhox
    rhos += np.ones((T, K)) * (1 - lambdas)[:, np.newaxis] * rho0
    return rhos

def get_linear_rhos(rhox, gamma, t, rho0):
    rhos1 = _get_linear_rhos(rhox, t, rho0)
    rhos2 = _get_linear_rhos((1 - gamma) * rho0 + rhox * gamma, T - t, rhox)
    return rhos1, rhos2

@jax.jit
def _likelihood(rhox, x, a):
    q = alpha * np.einsum('ij,j->i', x, rhox)
    return q[a] - logsumexp(q)
_likelihood = jax.vmap(_likelihood, (0, 0, 0))

def likelihood(rhox, gamma, data_x, data_a, t, rho0):
    rhos1, rhos2 = get_linear_rhos(rhox, gamma, t, rho0)
    ll = _likelihood(rhos1, data_x[:t, :, :], data_a[:t]).sum()
    ll += _likelihood(rhos2, data_x[t:, :, :], data_a[t:]).sum()
    return ll
likelihood = jax.jit(likelihood, static_argnames='t')

def _fit_regressing_model(t, data_x, data_a, rho0=-jax.numpy.ones(K) / K):
    grad_likelihood = jax.grad(likelihood, argnums=(0, 1))
    grad_likelihood = jax.jit(grad_likelihood, static_argnames='t')
    res = dict()
    res['rhox'] = -jax.numpy.ones(K) / K
    res['gamma'] = 0.2
    res['t_star'] = t

    grad_rhox_mnsq = np.zeros(K)
    grad_gamma_mnsq = 0
    for i in range(hyper['max_iter']):
        grad_rhox, grad_gamma = grad_likelihood(
            res['rhox'], res['gamma'], data_x, data_a, t, rho0)
        grad_rhox_mnsq = .1 * grad_rhox**2 + 0.9 * grad_rhox_mnsq
        grad_gamma_mnsq = .1 * grad_gamma**2 + 0.9 * grad_gamma_mnsq
        res['rhox'] += 0.001 * grad_rhox / (np.sqrt(grad_rhox_mnsq) + 1e-8)
        res['gamma'] += 0.001 * grad_gamma / (np.sqrt(grad_gamma_mnsq) + 1e-8)

    ll = likelihood(res['rhox'], res['gamma'], data_x, data_a, t, rho0)
    rhos1, rhos2 = get_linear_rhos(res['rhox'], res['gamma'], t, rho0)
    res['rhos'] = np.concatenate([rhos1, rhos2])
    return ll, res

def fit_regressing_model(data_x, data_a, rho0=-jax.numpy.ones(K) / K):
    lls = dict()
    res_dict = dict()
    # fit the models for all possible values of t_star
    for t in tqdm(range(1, T-1)):
        lls[t] , res_dict[t] = _fit_regressing_model(t, data_x, data_a, rho0)
    # return the model with the highest likelihood
    t_star = max(lls, key=lls.get)
    return res_dict[t_star]

print("Simulation results for the regressing agent")
r_errors = []
beta_errors = []
for key in range(5):
    data, data_x, data_a = read_data('3', key)
    res = fit_regressing_model(data_x, data_a)
    r_error, beta_error = evaluate_results(
        data['rhos'][T // 2], res['rhox'], 
        np.array(data['rhos']), res['rhos'])
    r_errors.append(r_error)
    beta_errors.append(beta_error)
r_errors = np.array(r_errors)
beta_errors = np.array(beta_errors)
print(f"Reward accuracy: {r_errors.mean():.3f} ({r_errors.std():.3f})")
print(f"Belief accuracy: {beta_errors.mean():.3f} ({beta_errors.std():.3f})")