#!/usr/bin/env/ python3

import numpy as np

# set dimension boundaries
DIM_X = 256
#DIM_Y = ?
#DIM_Z = ?

c = 0
data = np.load('PARAMETER.npy')
#print("Eigen::MatrixXd {")
print("{")
for cuboid in data:
    for rect in cuboid:
        for row in rect:
            for col in row:
                if(c is DIM_X):
                    #print("},\nEigen::MatrixXd {")
                    print("},\n{")
                    c = 0
                if(c < DIM_X - 1):
                    print("    " + str(col) + ",")
                else:
                    print("    " + str(col) )
                c = c + 1
print("}")

