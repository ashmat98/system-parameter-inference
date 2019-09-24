import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from matplotlib import pyplot as plt
plt.rcParams["figure.dpi"] = 200


for i in ['3', '6']:
    print(f"I=0.{i}")
    data = pd.read_csv(f"../data/forced_oscillations_0.{i}A.csv", )[["angular frequency", "average / arbitrary units"]].rename(
            columns={"angular frequency": "o", "average / arbitrary units": "x"})
    with open(f"samples.I=0{i}.pkl", "rb") as f:
        omega0, x0, gm = pickle.load(f)
        omega0 = omega0.flatten()
        x0 = x0.flatten()
        gm = gm.flatten()
    m_omega0, m_x0, m_gm = np.mean(omega0), np.mean(x0), np.mean(gm)
    d_omega0, d_x0, d_gm = np.std(omega0), np.std(x0),np.std(gm)
    Q = omega0 / (2*gm)
    m_Q, d_Q = np.mean(Q), np.std(Q)
    print("x0:     %0.4f ± %0.4f" % (m_x0, d_x0))
    print("omega0: %0.5f ± %0.5f" % (m_omega0, d_omega0))
    print("gamma:  %0.5f ± %0.5f" % (m_gm, d_gm))
    print("Q:      %0.5f ± %0.5f" % (m_Q, d_Q))
    print()