from functools import lru_cache
import random
import matplotlib.pyplot as plt

n = 100 # total number of inputs
# matrix: matrix[i][j] represents the cost of matching i to j
matrix = [[j - i for j in range(n)] for i in range(n)]
#matrix = [[j for j in range(n)] for i in range(n)]

# prior: prior[i] represents the prior probability of i
prior = [1 / n for i in range(n)]

# compute the maximum leakage given a cost matrix, a total cost value k, and the prior distribution
# k is the cost - 1 as the last place is 1 by default
def maximumLeakage (k):
    n = len(matrix)
    min_cost = dp(n - 1, k - 1)
    return min_cost

# this is an implementation of the dynamic programming algorithm (Alg 2) in the paper
@lru_cache(maxsize = 1000000)
def dp (i, j):
    if j == 0:
        return sum([prior[pos] * matrix[pos][i] for pos in range(i)])
    if i <= j:
        return sum([prior[pos] * matrix[pos][pos] for pos in range(i)])

    s = 0
    min_cost = float('inf')
    min_cut = -1
    for pos in range(i - 1, 0, -1):
        cost = dp(pos, j - 1) + s
        s += matrix[pos][i] * prior[pos]

        if cost < min_cost:
            min_cost = cost
            min_cut = pos
    return min_cost


# given a budget list, we compute the maximum leakage for each budget value in the list
# the returned leakage value is in the costs function
budgets = [1 + i for i in range(40)]
leaks = [0] * len(budgets)
for i, b in enumerate(budgets):
    leaks[i] = maximumLeakage(int(b))
# plot the leakage, this produce the curves in Fig 4(c)
plt.plot(leaks)
plt.show()





