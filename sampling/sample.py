import pickle
import pystan
import os
import numpy as np
import pandas as pd

stan_code = """
data {
    int N;
    vector[N] omega;
    vector[N] x;
    real l_x;
}
transformed data {
    
}
parameters {
    real omega0;
    real x0;
    real gm;

}
transformed parameters {

}
model {
    omega0 ~ gamma(2, 3.0/2.0);
    x0 ~ gamma(2, 0.7/2);
    gm ~ gamma(2, 0.3/2);

    x ~ normal(x0*square(omega0) ./ sqrt(square(square(omega) - square(omega0)) + square(2*gm*omega)), 1/l_x);
}
generated quantities {
}
"""
def main():
    if os.path.exists("./stan_model.pkl") is False:
        print("Compilling model...")
        sm = pystan.StanModel(model_code=stan_code)
        with open("./stan_model.pkl", "wb") as f:
            pickle.dump((stan_code, sm), f)
    else:
        print("Loading compiled model...")
        with open("./stan_model.pkl", "rb") as f:
            _, sm = pickle.load(f)

    print("Reading data...")
    data03 = pd.read_csv("../data/forced_oscillations_0.3A.csv", )[["angular frequency", "average / arbitrary units"]].rename(
        columns={"angular frequency": "o", "average / arbitrary units": "x"})
    data06 = pd.read_csv("../data/forced_oscillations_0.6A.csv", )[["angular frequency", "average / arbitrary units"]].rename(
        columns={"angular frequency": "o", "average / arbitrary units": "x"})

    def sample_from_prior(data):
        fit = sm.sampling(data={"N":len(data["o"]), "omega":data["o"].values, "x":data["x"].values, "l_x":1/0.1**2,
        #                         "alpha":alpha, "beta":beta, "gamma": gamma
                            }, iter=500000, chains=8, 
                        )

        lb = fit.extract(pars=["omega0", "x0", "gm"],permuted=False, inc_warmup=False);
        return lb["omega0"], lb["x0"], lb["gm"]

        omega0, x0, gm = np.mean(lb["omega0"]), np.mean(lb["x0"]), np.mean(lb["gm"])
        d_omega0, d_x0, d_gm = np.std(lb["omega0"]), np.std(lb["x0"]),np.std(lb["gm"])
        Q = omega0 / (2*gm)
        d_Q = np.sqrt((d_omega0/omega0)**2 + (d_gm/gm)**2)*Q
        print("x0:     %0.4f ± %0.4f" % (x0, d_x0))
        print("omega0: %0.5f ± %0.5f" % (omega0, d_omega0))
        print("gamma:  %0.5f ± %0.5f" % (gm, d_gm))
        print("Q:      %0.5f ± %0.5f" % (Q, d_Q))

        

    with open("samples.I=03.pkl", "wb") as f:
        pickle.dump(sample_from_prior(data03), f)
        print("\nsamples are saved in \"samples.I=03.pkl\"")

    
    with open("samples.I=06.pkl", "wb") as f:
        pickle.dump(sample_from_prior(data06), f)
        print("\nsamples are saved in \"samples.I=06.pkl\"")

if __name__ == "__main__":
    main()