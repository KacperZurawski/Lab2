# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 19:32:36 2021

@author: Kot
"""

import numpy as np

def lambda_f(x):
    y = np.zeros(len(x))
 
    for i in range(len(x)):
       
        if x[i] < 0:
            y[i] = np.sin(x[i])
        else:
             y[i] = np.sqrt(x[i])
    return y

