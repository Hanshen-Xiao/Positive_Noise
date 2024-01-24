from functools import lru_cache
import random
import matplotlib.pyplot as plt

# given a cost matrix, the budget value k, the last place is 1 by default
def maximumLeakage (matrix, k, prior):
    n = len(matrix)
    # link[i][j] stores the non-zero slot immediate before i if we have j budgets for positions 1 to i
    link = [[-1] * (k + 1) for _ in range(n + 1)]

    # an implementation of the dynamic programming algorithm (Alg 2) in the paper
    @lru_cache(maxsize = 1000000)
    def dp (i, j):
        if j == 0:
            link[i][j] = -1
            return sum([prior[pos] * matrix[pos][i] for pos in range(i + 1)])
        if i <= j:
            link[i][j] = i - 1
            return sum([prior[pos] * matrix[pos][pos] for pos in range(i + 1)])

        s = matrix[i][i] * prior[i]
        min_cost = float('inf')
        min_cut = -1
        for pos in range(i - 1, 0, -1):
            cost = dp(pos, j - 1) + s
            s += matrix[pos][i] * prior[pos]

            if cost < min_cost:
                min_cost = cost
                min_cut = pos
        link[i][j] = min_cut
        return min_cost

    # since the last place in the vector is 1 by default, we call dp with (n-1, k-1)
    min_cost = dp(n - 1, k - 1)
    
    pos = n - 1
    p = [0] * n
    p[-1] = 1
    # the dp only returns the minimal leakage, use the link table to recover the optimal vector
    for j in range(k - 1, 0, -1):
        new_pos = link[pos][j]
        p[new_pos] = 1
        pos = new_pos
    if sum(p) <= k:
        for i in range(k - sum(p)):
            p[i] = 1
    return link, p, min_cost

n = 7 # total number of inputs
k = 3 # the budget value
# matrix: matrix[i][j] represents the cost of matching i to j
matrix = [[j * j for j in range(n)] for i in range(n)] 
# prior distribution for the inputs
prior = [2/7, 1/7, 0, 1/7, 2/7, 0, 1/7]
link, p, min_cost = maximumLeakage(matrix, k, prior)
print(f"the optimal allocation vector is {p}, the leakage is {min_cost}")

