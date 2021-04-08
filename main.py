# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 12:38:34 2021

@author: Kot
"""

#podstawy
"""
a = 3
print(a)

string = "raz dwa trzy"
print(string)

a = 1 + 1/2 + 1/3 + 1/4 + 1/5
print(a)
print("%.4f" % a)
"""

#tablice
"""
import numpy as np
A = np.array([[1,2,3] , [7,8,9]])
print(A)
print(A[1])
print(A[1][2])

A = np.array([[1,2,\
               3],
              [7,8,9]])
print(A)

v = np.arange(1,7)
print(v)

v = np.arange(1,10,3)
print(v)

v = np.linspace(1,3,4)
print(v)

A = np.ones((2,3))
print(A)

A = np.eye(2)
print(A)


U = np.block([[
np.block([
np.block([[np.linspace(1,3,3)],
[np.zeros((2,3))]]) ,
np.ones((3,1))])
],
[np.array([100, 3, 1/2, 0.333])]] )
print(U)

print(U[0,2])
print(U[3,0])

A = [[1,2,3],[4,5,6]]
Q = np.delete(A, 1, 0)
print(Q)

print(np.size(A))
print(np.shape(A))
"""

#operacje na macierzach
"""
import numpy as np
A = np.array([[1,0,0],
              [2,3,-1],
              [0,7,2]
              ])

B = np.array([[3,20,1],
              [3,5,-2],
              [-2,-7,3]
              ])

print(A+B)

print(A+2)

M1 = A@B
print(M1)
M2 = B@A
print(M2)

M1 = A*B
print(M1)
M2 = B*A
print(M2)

M1 = A/B
print(M1)

C = np.linalg.solve(A,M1)
print(C)

P = np.linalg.matrix_power(A,2)
print(P)

print(P.transpose())
"""

#operacje logiczne
"""
import numpy as np
A = [[1,0,3],[4,5,6]]
B = [[3,0,1],[6,5,6]]

print(A==B)
print(A<B)


print(np.logical_not(A))
print(np.logical_or(A,B))

print(np.nonzero(A))

print(np.max(A))
"""

#matplotlib
import matplotlib.pyplot as plt
import numpy as np
"""
x = [1,2,3]
y = [4,6,5]
plt.plot(x,y)
plt.show()
"""
x = np.arange(0.0, 2.0, 0.01)
y = np.sin(2.0*np.pi*x)
plt.plot(x,y)
plt.show()


