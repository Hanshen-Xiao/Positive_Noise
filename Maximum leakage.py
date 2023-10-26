from functools import lru_cache
import random
import matplotlib.pyplot as plt

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

# compute the maximum leakage given a cost matrix, a total cost value k, and the prior distribution
# k is the cost - 1 as the last place is 1 by default
def maximumLeakage (k):
    n = len(matrix)
    min_cost = dp(n - 1, k - 1)
    return min_cost

n = 100
matrix = [[i * i] * n for i in range(n)]
budgets = [1 + i for i in range(40)]
prior = [1 / n for i in range(n)]
costs = [0] * len(budgets)
for i, b in enumerate(budgets):
    costs[i] = func(int(b))

plt.plot(costs)





