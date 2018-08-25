#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import sys

initialTrajFilename = "../../build/initial_traj.csv"

# Read file
initialTraj = np.genfromtxt(initialTrajFilename, delimiter=',')

indexReordering = [0, 2, 4, 1, 3, 5, 8, 9, 6, 7]
indexReorderingArray = np.argsort(indexReordering)
initialTrajOrdered = initialTraj[:,indexReorderingArray]

dt = 0.01

numPoints = initialTrajOrdered.shape[0]
#numPoints = initialTrajOrdered.shape[0] - 1
time = np.linspace(0, dt * (numPoints - 1), numPoints)

# TODO Maybe read these values from a list (the body names)
fig = plt.figure()
fig.suptitle("Initial Trajectory")

labels = ["xref", "dxref", "psiref", "dpsiref", "thref", "dthref", "ddthref", "tau0ref", "ref"]


for i in range(0, 8):
    ax = fig.add_subplot(330 + i + 1)

    x = time
    y1 = initialTrajOrdered[:, i]

    ax.plot(x, y1, label=labels[i])
    ax.legend()

i = i + 1
ax = fig.add_subplot(330 + i + 1)
x = initialTrajOrdered[:, i]
y = initialTrajOrdered[:, i+1]
ax.plot(x, y, label=labels[i])
ax.legend()

plt.show()
