import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
from numpy import log as ln

# return the mean, variance, and entropy of a truncated Gaussian N(mu, std) in the range [0, R]
def entropy (mu, std, R):
    a = - mu / std
    b = (R - mu) / std
    gauss = stats.norm(0, 1)
    
    pa, pb = gauss.pdf(a), gauss.pdf(b)
    ca, cb = gauss.cdf(a), gauss.cdf(b)
    
    z = cb - ca
    v1 = (pa - pb) / z
    v2 = (a * pa - b * pb) / z
    
    # compute the mean and variance of the truncated Gaussian
    mean = mu + std * v1
    var = (std * std) * (1 + v2 - v1 * v1)
    
    # compute the entropy
    v3 = (2 * math.pi * math.e) ** 0.5 * std * z
    etp = ln(v3) + v2 / 2
    
    return mean, var, etp

# given std, range R, budget S, and y's variance vary
# find a truncated Gaussian N(mu, std) in the range [0, R] such that 
#   (1) its second moment is less than S and (2) its entropy is the smallest
# return mu and the entropy gap
def entp_std (std, R, S, vary):
    def good (S1, S2):
        return (S2 <= S1 and S2 >= 0.99 * S1)
    
    mean, var, etp = entropy(0, std, R)
    # if the second moment when mu=0 is already larger than S, then such a mu does not exist
    if var + mean ** 2 > S:
        return -1, -1
    
    mean, var, etp = entropy(R / 2, std, R)
    netp = 0.5 * ln(2 * math.pi * math.e * (var + vary))
    # if the second moment when mu=R/2 is smaller than S, simply return mu=R/2
    if mean ** 2 + var <= S:
        return R / 2, netp - etp
    
    # since the entropy decreases with mu and the second moment increase with mu in [0, R/2]
    # use binary search between [0, R/2] to find a mu such that the resulted second moment is S
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

# given range R, budget S, and y's variance vary
# find a truncated Gaussian N(mu, std) in the range [0, R] such that 
#   (1) its second moment is less than S and (2) its entropy is the smallest
# return mu, std and the entropy gap
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
def experiment1 (R, S, mu, vary):
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

def plot (R, vary, S):
    # find the optimal leakage under R, vary and s in [0, S]
    results = {}
    for R in [5, 10, 20]:
        results[R] = [0] * S
        for s in range(0, S, 1):
            std, mu, e = find(R, s + 1, vary)
            results[R][s] = e
            
    # compute the leakage for simple gaussian distribution
    gaussian = [0] * S
    for s in range(0, S, 1):
        gaussian[s] = 0.5 * ln(1 + vary / (s + 1))

    # plot the graphs
    for R in [5, 10, 20]:
        plt.plot(results[R])
    plt.plot(gaussian)
    plt.show()


#experiment1(100, 10000, 50, 1000)
#plot(R, vary, S)

# set the value for range R, y's variance vary and maximum second moment S
R = 20
vary = 20.76
S = 80 
std, mu, e = find(R, S, vary)
shorten = lambda x: int(x * 1000) / 1000.00
# shorten = lambda x: x
print(f"the optimal truncated Gaussian in range [0, {R}] " +
    f"under second moment budget {S} and y's variance {vary} is achieved when")
print(f"    mu = {shorten(mu)} and std = {shorten(std)}, the resulted  entropy is {shorten(e)}")

