# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 13:42:23 2021

@author: Kot
"""

import numpy as np

A1 = np.array([np.linspace(1,5,5),np.linspace(5,1,5)])
A2 = np.zeros((3,2))
A3 = np.ones((2,3))*2
A4 = np.linspace(-90,-70,3)
A5 = np.ones((5,1))*10

A = np.block([[A3], [A4]])
A = np.block([A2,A])
A = np.block([[A1],[A]])
A = np.block([A,A5])
#print(A)

#zad4
B = A[1] + A[3]
print(B)

#zad5
C = list(map(np.max, zip(*A)))
#print(C)

#zad6
D = np.delete(B,0)
D = np.delete(D,len(D)-1)
print(D)

#zad7
D[D==4] = 0
print(D)