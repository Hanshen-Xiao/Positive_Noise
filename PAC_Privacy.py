import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
from numpy import log as ln

# compute the entropy of a truncated Gaussian N(mu, std) in the range [0, R]
def entropy (mu, std, R):
    a = - mu / std
    b = (R - mu) / std
    gauss = stats.norm(0, 1)
    
    pa, pb = gauss.pdf(a), gauss.pdf(b)
    ca, cb = gauss.cdf(a), gauss.cdf(b)
    
    z = cb - ca
    v1 = (pa - pb) / z
    v2 = (a * pa - b * pb) / z
    
    mean = mu + std * v1
    var = (std * std) * (1 + v2 - v1 * v1)
    
    v3 = (2 * math.pi * math.e) ** 0.5 * std * z
    etp = ln(v3) + v2 / 2
    
    return mean, var, etp

# compute the best mu given std, range R, budget S, and y's variance vary
def entp_std (std, R, S, vary):
    def good (S1, S2):
        return (S2 <= S1 and S2 >= 0.99 * S1)
    
    mean, var, etp = entropy(0, std, R)
    if var + mean ** 2 > S:
        return -1, -1
    
    mean, var, etp = entropy(R / 2, std, R)
    netp = 0.5 * ln(2 * math.pi * math.e * (var + vary))
    if mean ** 2 + var <= S:
        return R / 2, netp - etp
    
    left, right = 0, R / 2
    best_etp = float("inf")
    while right - left > 0.001 * (S ** 0.5):
        mid = (right + left) / 2
        
        mean, var, etp = entropy(mid, std, R)
        netp = 0.5 * ln(2 * math.pi * math.e * (var + vary))
        
        if mean ** 2 + var < S:
            left = mid
        else:
            right = mid
            
    return mid, netp - etp

# compute the optimal mu and std given R, S and vary
def find (R, S, vary):
    left, right = 0.1, S ** 0.5
    mu1, e1 = entp_std (left, R, S, vary)
    mu2, e2 = entp_std (right, R, S, vary)
    
    while right - left > 0.001 * (S ** 0.5):
        nright = right * 0.66 + left * 0.34
        nleft = right * 0.34 + left * 0.66
        
        mu1, e1 = entp_std (nleft, R, S, vary)
        mu2, e2 = entp_std (nright, R, S, vary)
        if e1 < e2:
            right = nright
        else:
            left = nleft
        
    return (left + right) / 2, (mu1 + mu2) / 2, (e1 + e2) / 2


# this experiment verifies that given mu, the objective function reduces with std
def experiment1 (R, S, mu, vary)
    n = 100
    results = [0] * n
    s_val = [0] * n
    etp = [0] * n
    netp = [0] * n
    for i in range(n):
        std = (i + 1)
        mean, var, etp[i] = entropy(mu, std, R)
        v3 = 2 * math.pi * math.e * (var + vary)
        netp[i] = 0.5 * ln(v3)
        
        if var + mean ** 2 <= S:
            s_val[i] = var
            results[i] = netp[i] - etp[i]
    plt.plot(results)
    return results, s_val



#experiment1(100, 10000, 50, 1000)


R = 20
vary = 20.76
n = 80
results = {}
gaussian = [0] * n
for R in [5, 10, 20]:
    results[R] = [0] * n
    for S in range(0, n, 1):
        std, mu, e = find(R, S + 1, vary)
        results[R][S] = e
        
for S in range(0, n, 1):
    gaussian[S] = 0.5 * ln(1 + vary / (S + 1))
for R in [5, 10, 20]:
    plt.plot(results[R])
plt.plot(gaussian)
