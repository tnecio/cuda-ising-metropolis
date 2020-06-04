#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from sys import argv

import pandas as pd


# In[8]:


test = pd.read_csv("tmp.csv")


# In[33]:


import matplotlib
matplotlib.rcParams['font.size'] = 14


# In[34]:


plt.figure(figsize=(5, 4))
plt.plot(test.Temperature[1:], np.abs(test.Magnetisation[1:]))
plt.xlabel("Temperatura")
plt.ylabel("Magnetyzacja\n(wartość bezwzględna)")
plt.tight_layout()
plt.savefig("images/plot_magn.png")


# In[35]:


plt.figure(figsize=(5, 4))
plt.plot(test.Temperature[1:], test.Susceptibility[1:])
plt.xlabel("Temperatura")
plt.ylabel("Podatność magnetyczna")
plt.tight_layout()
plt.savefig("images/plot_susc.png")


# In[36]:


plt.figure(figsize=(5, 4))
plt.plot(test.Temperature[1:], test.SpecificHeat[1:])
plt.xlabel("Temperatura")
plt.ylabel("Ciepło właściwe")
plt.tight_layout()
plt.savefig("images/plot_heat.png")


# In[38]:


plt.figure(figsize=(5, 4))
plt.plot(test.Temperature[1:], test.Energy[1:] / 10 ** 4)
plt.xlabel("Temperatura")
plt.ylabel("Energia całkowita układu ($\\times 10^4$)")
plt.tight_layout()
plt.savefig("images/plot_energy.png")


# In[ ]:




