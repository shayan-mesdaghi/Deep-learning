# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 23:57:01 2022

@author: SHM
"""

import numpy as np

x0= np.array(120)# tensor 0D
print('x0 dimentions: ', x0.ndim)
print('x0 shape: ',x0.shape)
print('x0 type: ', x0.dtype)

x1= np.array([[1, 2, 3]])
print('x0 dimentions: ', x1.ndim)
print('x0 shape: ',x1.shape)
print('x0 type: ', x1.dtype)

x2= np.array([[1, 2, 3],
              [4, 5,6],
              [7,8,9]])
print('x2 dimentions: ', x2.ndim)
print('x2 shape: ',x2.shape)
print('x2 type: ', x2.dtype)