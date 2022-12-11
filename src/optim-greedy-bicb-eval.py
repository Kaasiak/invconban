import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import dill
import numpy as np

color_list = [
    '#5778a4', '#e49444', '#d1615d', '#85b6b2', 
    '#6a9f58', '#e7ca60', '#a87c9f', '#f1a2a9',
    '#967662', '#b8b0ac'
]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)

params = {
 'axes.axisbelow': False,
 'axes.edgecolor': 'lightgrey',
 'axes.facecolor': 'None',
 'axes.grid': False,
 'axes.labelcolor': 'dimgrey',
 'axes.spines.right': False,
 'axes.spines.top': False,
 'figure.facecolor': 'white',
 'lines.solid_capstyle': 'round',
 'patch.edgecolor': 'w',
 'patch.force_edgecolor': True,
 'text.color': 'black',
 'xtick.bottom': True,
 'xtick.color': 'dimgrey',
 'xtick.direction': 'out',
 'xtick.top': False,
 'ytick.color': 'dimgrey',
 'ytick.direction': 'out',
 'ytick.left': True,
 'ytick.right': False,
 'font.family': 'sans-serif'
}
plt.rcParams.update(params)

### Tables 1. and 2.

def evaluate_results(data, res):
    r_error = np.abs(data['rhox'] - res['rhox']).sum()
    try:
        x_true = data['betas_mean']
        x = res['betas_mean']
    except:
        x_true = data['rhos']
        x = res['betas_mean']
    
    x_true = x_true / np.abs(x_true).sum(axis=-1, keepdims=True)
    x = x / np.abs(x_true).sum(axis=-1, keepdims=True)
    belief_error = np.abs(x - x_true).sum(axis=-1).mean()
    return r_error, belief_error

agents = [('1', 'sampling'), ('11', 'optimistic'), ('12', 'greedy')]

for agent, agent_name in agents:
    r_errors = []
    belief_errors = []
    for agent_key in range(5):
        # load the data
        with open(f'data/icb-syntdata/agent{agent}-key{agent_key}.obj', 'rb') as f:
            data = dill.load(f)

        # load the results
        if agent == '1':
            # original BICB
            agent_name = "sampling"
            path = f'res/bicb-agent{agent}-key{agent_key}.obj'
        elif agent == '11':
            # optimistic BICB
            agent_name = "optimistic"
            path = f'res/optimistic/bicb-agent{agent}-key{agent_key}.obj'
        elif agent == '12':
            # optimistic BICB
            agent_name = "greedy"
            path = f'res/greedy/bicb-agent{agent}-key{agent_key}.obj'

        with open(path, 'rb') as f:
            res = dill.load(f)

        r_error, beta_error = evaluate_results(data, res)
        r_errors.append(r_error)
        belief_errors.append(beta_error)
    r_errors = np.array(r_errors)
    belief_errors = np.array(belief_errors)
    print(f"Smiluation results for the {agent_name} agent")
    print(f"Reward accuracy: {r_errors.mean():.3f} ({r_errors.std():.3f})")
    print(f"Belief accuracy: {belief_errors.mean():.3f} ({belief_errors.std():.3f})")


### Figure 1.

agent = '1'
agent_key = 0

with open(f'data/icb-syntdata/agent{agent}-key{agent_key}.obj', 'rb') as f:
        data = dill.load(f)

with open(f'res/bicb-agent{agent}-key{agent_key}.obj', 'rb') as f:
        bicb_res = dill.load(f)

with open(f'res/optimistic/bicb-agent{agent}-key{agent_key}.obj', 'rb') as f:
        optim_bicb_res = dill.load(f)

fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
ax[0].set_title("B-ICB")
ax[0].plot(data['betas_mean'], alpha=0.5, linestyle='--')
for i, b in enumerate(bicb_res['betas_mean'].T):
    ax[0].plot(b, color = f'C{i}', alpha=0.9)

ax[1].set_title("optimistic B-ICB")
ax[1].plot(data['betas_mean'], alpha=0.5, linestyle='--')
for i, b in enumerate(optim_bicb_res['betas_mean'].T):
    ax[1].plot(b, color = f'C{i}', alpha=0.9)
fig.supxlabel("Time")
fig.supylabel("Beliefs")
fig.tight_layout()

custom_lines = [Line2D([0], [0], color='dimgrey', alpha=0.5, linestyle='--'),
                Line2D([0], [0], color='dimgrey', alpha=0.9)]
fig.legend(custom_lines, ['$\\beta_t$', '$\\hat{\\beta}_t$'], loc='upper left')
fig.savefig("res/figures/optimistic.pdf")
plt.show()


### Table 3.

environments = [('1', 'sampling'), ('11', 'optimistic'), ('12', 'greedy')]
solvers = ['sampling', 'optimistic', 'greedy']

for solver in solvers:
    for env_key, env_name in environments:
        r_errors = []
        belief_errors = []
        for agent_key in range(5):
            # load the data
            with open(f'data/icb-syntdata/agent{env_key}-key{agent_key}.obj', 'rb') as f:
                data = dill.load(f)

            # load the results
            if solver == 'sampling':
                # original BICB
                path = f'res/bicb-agent{env_key}-key{agent_key}.obj'
            elif solver == 'optimistic':
                # optimistic BICB
                path = f'res/optimistic/bicb-agent{env_key}-key{agent_key}.obj'
            elif solver == 'greedy':
                path = f'res/greedy/bicb-agent{env_key}-key{agent_key}.obj'

            with open(path, 'rb') as f:
                res = dill.load(f)

            r_error, beta_error = evaluate_results(data, res)
            r_errors.append(r_error)
            belief_errors.append(beta_error)
        r_errors = np.array(r_errors)
        belief_errors = np.array(belief_errors)
        print(f"Smiluation results for the {agent} solver in {env_name} environment")
        print(f"Reward accuracy: {r_errors.mean():.4f} ({r_errors.std():.3f})")
        print(f"Belief accuracy: {belief_errors.mean():.4f} ({belief_errors.std():.3f})")
