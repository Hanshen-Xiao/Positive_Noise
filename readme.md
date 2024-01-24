We include the codes for the paper "On the Foundation of One-Sided Noise and Side-Channel Leakage Control".

maxLeakage.py: implementation of Algorithm 2 and 4 in the paper.
Given a cost matrix, a budget upper bound and a prior distribution, print out the optimal allocation array and the corresponding leakage. The following input parameters can be adjusted:
line 50: n, number of inputs
line 51: k, budget upper bound
line 53: matrix, the cost matrix
line 55: prior, the prior distribution


maxLeakage_plot.py: code for creating Fig 4(c) in the paper. Compared to maxLeakage.py, the dynamic programming dp(*,*) is moved out as an independent function. This is because we need to call maximumLeakage() multiple times, moving out dp(*,*) would greatly enhance performance.


PAC.py: implementation of Algorithm 3 in the paper.
Given range R, budget S and y's variance vary, find a truncated Gaussian N(mu, std) in the range [0, R] such that (1) its second moment is less than S and (2) its entropy is the smallest. The following input parameters can be adjusted:
line 134: R, the range upper bound
line 135: vary, the variance for y
line 136: S, the upper bound for the second moment

functions:
1. entropy(mu, std, R): return the mean, variance, and entropy of a truncated Gaussian N(mu, std) in the range [0, R].
2. entp_std (std, R, S, vary): given std, range R, budget S and y's variance vary, find a truncated Gaussian N(mu, std) in the range [0, R] such that (1) its second moment is less than S and (2) its entropy is the smallest
3. find (R, S, vary): given range R, budget S and y's variance vary, find a truncated Gaussian N(mu, std) in the range [0, R] such that (1) its second moment is less than S and (2) its entropy is the smallest.
4.experiment1 (R, S, mu, vary): this experiment verifies that given mu, the objective function reduces with std.




