import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from matplotlib import pyplot as plt
plt.rcParams["figure.dpi"] = 200

data = pd.read_csv("../data/forced_oscillations_0.3A.csv", )[["angular frequency", "average / arbitrary units"]].rename(
        columns={"angular frequency": "o", "average / arbitrary units": "x"})

with open("samples.I=03.pkl", "rb") as f:
    omega0, x0, gm = pickle.load(f)
    perm = np.random.permutation(omega0.size)[:1000]
    omega0 = omega0.flatten()[perm]
    x0 = x0.flatten()[perm]
    gm = gm.flatten()[perm]

m_omega0, m_x0, m_gm = np.mean(omega0), np.mean(x0), np.mean(gm)
d_omega0, d_x0, d_gm = np.std(omega0), np.std(x0),np.std(gm)
Q = omega0 / (2*gm)
m_Q, d_Q = np.mean(Q), np.std(Q)

def curve(omega, omega0, x0, gm):
    return x0 * omega0**2/((omega0**2-omega**2)**2 + (2*gm*omega)**2)**0.5

xx = np.linspace(0, max(data["o"]), 400)
   
means, stds = [], []
for x in tqdm(xx):
    ys = curve(x, omega0, x0, gm)
    means.append(np.mean(ys))
    stds.append(np.std(ys))

means = np.array(means)
stds = np.array(stds)

scale = 100

def init():
    plt.figure(figsize=(16/2,9/2), dpi=200)
    
    grid = plt.GridSpec(3, 12, wspace=0.9, hspace=1.5)
    ax1 = plt.subplot(grid[0,0:2])
    ax2 = plt.subplot(grid[1,0:2])
    ax3 = plt.subplot(grid[2,0:2])
    for ax in [ax1, ax2, ax3]:
        ax.set_yticks([])
        ax.tick_params(axis='both', which='major', labelsize=8)

        
    ax_main = plt.subplot(grid[:, 3:])
    plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.7)
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    ax_main.set_xlabel("$\omega \; / \; s^{-1}$ ")
    ax_main.set_ylabel("Amplitude")
    ax_main.set_ylim(-0.1, 13.5)    
    
    return [ax1, ax2, ax3], ax_main

def get_lim(points):
    return points.min(), points.max()
    
def sample(n):
    axs, ax_main = init()
    plt.suptitle(f"{n:4d} samples")
    for ax, points, name in zip(axs, [omega0, x0, gm], ["$\\omega_0$", "$X_0$", "$\\gamma$"]):
        ax.scatter(points[:n], np.zeros((n,)), s=20, alpha=0.2)
        ax.set_title(name)
        ax.set_xlim(get_lim(points))
    for i in range(n):
        ax_main.plot(xx, means+ scale*(curve(xx, omega0[i], x0[i], gm[i])-means), lw=0.9, alpha=0.8)
    ax_main.set_title("Not in scale!", fontsize=7)
    plt.savefig(f"samples_{n:03d}.png")

def sample_all():
    
    axs, ax_main = init()
    
    plt.suptitle("All samples")

    for ax, points, name in zip(axs, [omega0, x0, gm], ["$\\omega_0$", "$X_0$", "$\\gamma$"]):
        ax.hist(points, density=1, bins=20)
        ax.set_title(name)
        ax.set_xlim(get_lim(points))

    
    ax_main.fill(np.concatenate([xx, xx[::-1]]),
            np.concatenate([means - 2*scale * stds,
                            (means + 2*scale * stds)[::-1]]),
            alpha=.5, fc='orange', ec='None', label='95% confidence interval')
    ax_main.plot(xx, means, lw=1.5)
    ax_main.set_title("Not in scale!", fontsize=7)
    plt.savefig("samples_err.png")

def plot_main():
    plt.figure(figsize=(5,3), dpi=200)
    plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.7)
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.xlabel("$\omega \; / \; s^{-1}$ ")
    plt.ylabel("Amplitude")
    plt.plot(xx, means, lw=1.5)
    plt.plot(data.o, data.x, lw=0, marker="+", mec="black", ms=10, mew=2)
    plt.savefig("result.png")
    plt.show()

sample(3)
sample(10)
sample(30)
sample(100)
sample_all()
plot_main()
exit(0)
