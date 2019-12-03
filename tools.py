import numpy as np
from itertools import combinations
import yaml


def Kendalls_Tau(y):
    count = 0
    Positive = 0
    Negative = 0
    for s, t in combinations(range(len(y)), 2):
        u = np.array(y[s])
        v = np.array(y[t])
        for i, j in combinations(range(len(u)), 2):
            u_i = u[i]
            u_j = u[j]

            length1 = np.sum((v - u_i)**2, axis=1)
            p = np.argmin(length1)

            length2 = np.sum((v - u_j)**2, axis=1)
            q = np.argmin(length2)

            count += 1
            if (i-j)*(p-q) > 0:
                Positive += 1
            else:
                Negative += 1
    tau = (Positive-Negative)/(0.5*count*(count-1))

    tau_pos_neg = {"tau": tau, "positive": positive, "negative": negative}
    with open(OPTION.output_dir + "tau.yaml") as f:
        f.write(yaml.dump(tau_pos_neg))

    return tau
