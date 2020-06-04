#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from sys import argv

if (len(argv) == 1 or len(argv) > 2):
    print('example of use \n python plot_matrices.py <name_of_output_png_file>)')

data = input()

data = data.split()
num_of_rows = int(data[0])
num_of_cols = int(data[1])

data = data[2:]
data = [int(i) for i in data]
data = np.array(data)
data = data.reshape(num_of_rows, num_of_cols)

print(data)

plt.figure()
plt.imshow(data, cmap='gray')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig(argv[1])
