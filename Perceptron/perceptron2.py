# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 14:16:04 2017

@author: Jones
"""
import numpy as np

entradas = np.array([-1, 1])
pesos = np.array([0.5, 0.5])

def soma(e, p):
    return e.dot(p)
# dot product / produto escalar
        
s = soma(entradas, pesos)
print(s)

def stepFunction(soma):
    if (soma >= 1):
        return 1
    return 0

r = stepFunction(s)
print(r)