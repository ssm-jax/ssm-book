#!/usr/bin/env python
# coding: utf-8

# # Implementing discrete HMMs in Numpy 
# 
# We start with a simple numpy implementation.

# In[1]:


print(42)


# In[2]:




# Install necessary libraries

try:
    import jax
except:
    # For cuda version, see https://github.com/google/jax#installation
    get_ipython().run_line_magic('pip', 'install --upgrade "jax[cpu]"')
    import jax

try:
    import jsl
except:
    get_ipython().run_line_magic('pip', 'install git+https://github.com/probml/jsl')
    import jsl

try:
    import rich
except:
    get_ipython().run_line_magic('pip', 'install rich')
    import rich


# In[3]:


import abc
from dataclasses import dataclass
import functools
import itertools

from typing import Any, Callable, NamedTuple, Optional, Union, Tuple


import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import inspect
import inspect as py_inspect

from rich import inspect as r_inspect
from rich import print as r_print

def print_source(fname):
    r_print(py_inspect.getsource(fname))


# In[4]:


import jsl
import jsl.hmm.hmm_numpy_lib as hmm_lib_np
#import jsl.hmm.hmm_lib as hmm_lib_jax

normalize = hmm_lib_np.normalize_numpy
print_source(normalize)
#print_source(hmm_lib_np.normalize_numpy)


# Here are some handy utility functions we have already defined.

# In[5]:


normalize = hmm_lib_np.normalize_numpy
print_source(normalize)
#print_source(hmm_lib_np.normalize_numpy)


# We first create the "Ocassionally dishonest casino" model from {cite}`Durbin98`.
# 
# ```{figure} /figures/casino.png
# :scale: 50%
# :name: casino
# 
# Illustration of the casino HMM.
# ```
# 
# 

# In[6]:



# state transition matrix
A = np.array([
    [0.95, 0.05],
    [0.10, 0.90]
])

# observation matrix
B = np.array([
    [1/6, 1/6, 1/6, 1/6, 1/6, 1/6], # fair die
    [1/10, 1/10, 1/10, 1/10, 1/10, 5/10] # loaded die
])

pi = np.array([1, 1]) / 2

(nstates, nobs) = jnp.shape(B)
for i in range(nstates):
    A[i,:] = normalize(A[i,:])[0]
    B[i,:] = normalize(B[i,:])[0]


# Let's bundle the parameters into a structure.

# In[7]:



class HMMNumpy(NamedTuple):
    trans_mat: np.array  # A : (n_states, n_states)
    obs_mat: np.array  # B : (n_states, n_obs)
    init_dist: np.array  # pi : (n_states)
        

params_numpy = HMMNumpy(A, B, pi)
print(params_numpy)


# Function to sample a single sequence of hidden states and discrete observations.

# In[8]:


hmm_sample = hmm_lib_np.hmm_sample_numpy
print_source(hmm_sample)


# Let's sample from this model.

# In[9]:


seq_len = 20
state_seq, obs_seq = hmm_sample(params_numpy, seq_len, random_state=0)
print(state_seq)
print(obs_seq)

