import argparse
import dill
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
from jax.scipy.special import logsumexp
from tqdm import tqdm

jax.config.update('jax_platform_name', 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True)
parser.add_argument('-o', '--output', default='res/bandit.obj')
args = parser.parse_args()

hyper = dict()
hyper['n_samples'] = 200
hyper['max_iter'] = 150

with open(args.input, 'rb') as f:
    data = dill.load(f)
    data_x = np.array(data['x'])
    data_a = np.array(data['a'])
    key = data['key']

T = data_x.shape[0]
A = data_x.shape[1]
K = data_x.shape[2]
alpha = 20
sigma = .10

__betas_N = lambda t: np.einsum('i,j->ij', data_x[t,data_a[t]], data_x[t,data_a[t]])
__betas_N = jax.vmap(__betas_N)
_betas_N = __betas_N(np.arange(T-1)).cumsum(axis=0)
_betas_N = np.concatenate((np.zeros((K,K))[None,...], _betas_N))

__betas_y = lambda r, t: r * data_x[t,data_a[t]]
__betas_y = jax.vmap(__betas_y)
_betas_y = lambda rs: np.concatenate((np.zeros(K)[None,...], __betas_y(rs, np.arange(T-1)).cumsum(axis=0)))
_betas_y = jax.jit(_betas_y)
_BETAS_Y = jax.jit(jax.vmap(_betas_y))

@jax.jit
def decode(params):
    beta0 = np.exp(20 * params['beta0'])
    beta0_y = -np.ones(K)/K * beta0
    beta0_N = np.eye(K) * beta0
    return beta0_y, beta0_N

@jax.jit
def _likelihood_rs(x, a, beta_mean, beta_cov):
    q = alpha * np.einsum('ij,j->i', x, beta_mean) + np.einsum('ij,jk,ik->i', x, beta_cov, x)
    return q[a] - logsumexp(q)
_likelihood_rs = jax.vmap(_likelihood_rs)

@jax.jit
def likelihood_rs(beta0_N, beta0_y, data_x, data_a, rs):
    betas_invN = np.linalg.inv(beta0_N + _betas_N)
    betas_mean = np.einsum('ijk,ik->ij', betas_invN, beta0_y + _betas_y(rs[:-1]))
    betas_cov = betas_invN * sigma**2
    return _likelihood_rs(data_x, data_a, betas_mean, betas_cov)

@jax.jit
def _sample_rs(t, rs, rhox, data_x, data_a, beta0_N, beta0_y, key):
    x, a = data_x[t], data_a[t]
    keys = jax.random.split(key, 3)
    r = jax.random.normal(keys[0]) * sigma + np.dot(rhox, x[a])
    _r = jax.random.normal(keys[1]) * sigma + np.dot(rhox, x[a])
    temp_rs = rs.at[t].set(r)
    _temp_rs = rs.at[t].set(_r)
    like = likelihood_rs(beta0_N, beta0_y, data_x, data_a, temp_rs).sum()
    _like = likelihood_rs(beta0_N, beta0_y, data_x, data_a, _temp_rs).sum()
    cond = _like - like > np.log(jax.random.uniform(keys[2]))
    return jax.lax.select(cond, _temp_rs, temp_rs)

def sample_rs(args, key):
    (rs, rhox, data_x, data_a, beta0_N, beta0_y), key = args, key
    for t in range(T):
        rs = _sample_rs(t, rs, rhox, data_x, data_a, beta0_N, beta0_y, key)
    return rs
sample_rs = jax.vmap(sample_rs, (None, 0))

@jax.jit
def compute_rhox(RS):
    _beta_y = _BETAS_Y(RS[:, :-1])[:,-1,:].mean(axis=0)
    _beta_N = _betas_N[-1]
    rhox = np.einsum('ij,j->i', np.linalg.inv(_beta_N), _beta_y)
    return rhox

@jax.jit
def _likelihood(params, data_x, data_a, rs):
    beta0_y, beta0_N = decode(params)
    return likelihood_rs(beta0_N, beta0_y, data_x, data_a, rs).sum()
_likelihood = jax.vmap(_likelihood, (None, None, None, 0))

@jax.jit
def likelihood(params, data_x, data_a, RS):
    return _likelihood(params, data_x, data_a, RS).mean()

grad_likelihood = jax.grad(likelihood)
grad_likelihood = jax.jit(grad_likelihood)

# initialize beta, rewards and rho_env (rhox)
params = {'beta0': 10e-4}
grad_mnsq = {'beta0': 10e-4}
beta0_y, beta0_N = decode(params)
rs = jax.random.normal(key, shape=(T,))
rhox = -np.ones(K)/K

for i in tqdm(range(hyper['max_iter'])): 
    key, subkey = jax.random.split(key)
    # sample rewards
    RS = sample_rs(
        (rs, rhox, data_x, data_a, beta0_N, beta0_y), 
        jax.random.split(subkey, hyper['n_samples'])
    )
    rs = RS.mean(axis=0)
    rhox = compute_rhox(RS)
    # update params with gradient descent
    grad = grad_likelihood(params, data_x, data_a, RS)
    grad_mnsq['beta0'] = .1 * grad['beta0']**2 + .9 * grad_mnsq['beta0']
    params['beta0'] += .001 * grad['beta0'] / (np.sqrt(grad_mnsq['beta0']) + 1e-8)
    beta0_y, beta0_N = decode(params)

rhox = rhox / np.abs(rhox).sum()

res = dict()
res['rhox'] = rhox
res['beta0_y'] = beta0_y
res['beta0_N'] = beta0_N

key, subkey = jax.random.split(key)
RS = sample_rs(
        (rs, rhox, data_x, data_a, beta0_N, beta0_y), 
        jax.random.split(subkey, hyper['n_samples'])
    )
BETAS_Y = beta0_y + _BETAS_Y(RS[:, :-1])
betas_invN = np.linalg.inv(beta0_N + _betas_N)
betas_mean = np.einsum('ijk,lik->lij', betas_invN, BETAS_Y).mean(axis=0)
betas_cov = betas_invN * sigma**2

res['betas_mean'] = betas_mean
res['betas_cov'] = betas_invN

with open(args.output, 'wb') as f:
    dill.dump(res, f)