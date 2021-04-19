# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 13:42:23 2021

@author: Kot
"""

import numpy as np
import matplotlib.pyplot as plt
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

#zad8
min_v = min(C)
max_v = max(C)
E = [x for x in C if x != min_v and x != max_v]

#zad9
j = np.array([])
min_v = A.min()
max_v = A.max()
for x in range(np.shape(A)[0]):
    if min_v in A[x,:]:
        if max_v in A[x,:]:
            print(A[x,:])

#zad10
M1 = D * E
M2 = D @ E



#zad11

def funkcja(n):
    m = np.random.randint(-1, 11, [n, n])
    return m, np.trace(m)


print(funkcja(4))

#zad12
def funkcja2(macierz):
    size = np.shape(macierz)
    macierz = macierz * (1-np.eye(size[0], size[0]))
    macierz = macierz * (1-np.fliplr(np.eye(size[0], size[0])))
    return macierz

m = funkcja(4)[0]
print(funkcja2(m))



#zad13
def funkcja3(macierz):
    suma=0
    
    size = np.shape(macierz)
    
    for i in range(size[0]):
        if i%2==1:
            suma = suma + np.sum(macierz[i, :])
            
    return suma

print(funkcja3(m))

#zad14
def funkcja4(x):
    return np.cos(2*x)

x = np.arange(-10,10,0.1)
y = funkcja4(x)
plt.figure(0)
plt.plot(x,y,'r--')


#zad15
import klasa

x = np.arange(-10,10,0.1)
y = klasa.lambda_f(x)
plt.figure(1)
plt.plot(x,y,'g+')

#zad17
x = np.arange(-10,10,0.1)
y1 = klasa.lambda_f(x)
y2 = funkcja4(x)
y3 = y1+y2
plt.figure(2)
plt.plot(x,y3,'b+')

#zad18
A = np.array([[10,5,1,7],[10,9,5,5],[1,6,7,3],[10,0,1,5]])
B = np.array([[34],[44],[25],[27]])
X = np.linalg.inv(A) @ B
#heheheheheheheheh




#zad19
i = 1000000
x = np.linspace(0, 2*np.pi, i)
y = np.sin(x)
calka = np.sum(2*np.pi/i*y)

