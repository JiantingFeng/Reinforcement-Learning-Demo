#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of Markov Reward Process with Bootstrapping
"""

import numpy as np

# Number of states
N = 5
# Number of iterations
T = 1000
# Discount factor gamma
gamma = 0.99
# Transition matrix generated from a dirichlet distribution
P = np.random.dirichlet(np.ones(N), size=(N, N))[0]
#  print(P)
# Reward Matrix
R = np.random.randint(0, 10, size=(N, 1))
# Exact solution of the value function
V = np.linalg.inv(np.eye(N) - gamma * P).dot(R)
V_old = np.zeros((N, 1))

for _ in range(T):
    V_new = R + gamma * np.dot(P, V_old)
    if np.linalg.norm(V_new - V_old) < 1e-6:
        break
    V_old = V_new

print("Exact solution: {}".format(V.T[0]))
print("Approximate solution: {}".format(V_old.T[0]))
print("Relative error: {}".format(np.linalg.norm(V_old - V) / np.linalg.norm(V)))
