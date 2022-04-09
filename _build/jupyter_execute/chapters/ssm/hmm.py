#!/usr/bin/env python
# coding: utf-8

# (sec:hmm-ex)=
# # Hidden Markov Models
# 
# In this section, we introduce Hidden Markov Models (HMMs).

# ## Boilerplate

# In[1]:


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



# In[2]:


# Import standard libraries

import abc
from dataclasses import dataclass
import functools
import itertools

from typing import Any, Callable, NamedTuple, Optional, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np


import jax
import jax.numpy as jnp
from jax import lax, vmap, jit, grad
from jax.scipy.special import logit
from jax.nn import softmax
from functools import partial
from jax.random import PRNGKey, split

import inspect
import inspect as py_inspect
from rich import inspect as r_inspect
from rich import print as r_print

def print_source(fname):
    r_print(py_inspect.getsource(fname))


# ## Utility code

# In[3]:




def normalize(u, axis=0, eps=1e-15):
    '''
    Normalizes the values within the axis in a way that they sum up to 1.
    Parameters
    ----------
    u : array
    axis : int
    eps : float
        Threshold for the alpha values
    Returns
    -------
    * array
        Normalized version of the given matrix
    * array(seq_len, n_hidden) :
        The values of the normalizer
    '''
    u = jnp.where(u == 0, 0, jnp.where(u < eps, eps, u))
    c = u.sum(axis=axis)
    c = jnp.where(c == 0, 1, c)
    return u / c, c


# (sec:casino-ex)=
# ## Example: Casino HMM
# 
# We first create the "Ocassionally dishonest casino" model from {cite}`Durbin98`.
# 
# ```{figure} /figures/casino.png
# :scale: 50%
# :name: casino-fig
# 
# Illustration of the casino HMM.
# ```
# 
# There are 2 hidden states, each of which emit 6 possible observations.

# In[4]:


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

pi, _ = normalize(np.array([1, 1]))
pi = np.array(pi)


(nstates, nobs) = np.shape(B)



# Let's make a little data structure to store all the parameters.
# We use NamedTuple rather than dataclass, since we assume these are immutable.
# (Also, standard python dataclass does not work well with JAX, which requires parameters to be
# pytrees, as discussed in https://github.com/google/jax/issues/2371).

# In[5]:



class HMM(NamedTuple):
    trans_mat: jnp.array  # A : (n_states, n_states)
    obs_mat: jnp.array  # B : (n_states, n_obs)
    init_dist: jnp.array  # pi : (n_states)

params = HMM(A, B, pi)
print(params)
print(type(params.trans_mat))


# ## Sampling from the joint
# 
# Let's write code to sample from this model. First we code it in numpy using a for loop. Then we rewrite it to use jax.lax.scan, which is faster.

# In[6]:



def hmm_sample_numpy(params, seq_len, random_state=0):

    def sample_one_step_(hist, a, p):
        x_t = np.random.choice(a=a, p=p)
        return np.append(hist, [x_t]), x_t

    np.random.seed(random_state)

    trans_mat, obs_mat, init_dist = params.trans_mat, params.obs_mat, params.init_dist
    n_states, n_obs = obs_mat.shape

    state_seq = np.array([], dtype=int)
    obs_seq = np.array([], dtype=int)

    latent_states = np.arange(n_states)
    obs_states = np.arange(n_obs)

    state_seq, zt = sample_one_step_(state_seq, latent_states, init_dist)
    obs_seq, xt = sample_one_step_(obs_seq, obs_states, obs_mat[zt])

    for _ in range(1, seq_len):
        state_seq, zt = sample_one_step_(state_seq, latent_states, trans_mat[zt])
        obs_seq, xt = sample_one_step_(obs_seq, obs_states, obs_mat[zt])

    return state_seq, obs_seq


# In[7]:


seq_len = 20
state_seq, obs_seq = hmm_sample_numpy(params, seq_len, random_state=0)
print(state_seq)
print(obs_seq)


# Now let's write a JAX version.

# In[8]:


#@partial(jit, static_argnums=(1,))
def hmm_sample(params, seq_len, rng_key):

    trans_mat, obs_mat, init_dist = params.trans_mat, params.obs_mat, params.init_dist
    n_states, n_obs = obs_mat.shape

    initial_state = jax.random.categorical(rng_key, logits=logit(init_dist), shape=(1,))
    obs_states = jnp.arange(n_obs)

    def draw_state(prev_state, key):
        logits = logit(trans_mat[:, prev_state])
        state = jax.random.categorical(key, logits=logits.flatten(), shape=(1,))
        return state, state

    rng_key, rng_state, rng_obs = jax.random.split(rng_key, 3)
    keys = jax.random.split(rng_state, seq_len - 1)

    final_state, states = jax.lax.scan(draw_state, initial_state, keys)
    state_seq = jnp.append(jnp.array([initial_state]), states)

    def draw_obs(z, key):
        obs = jax.random.choice(key, a=obs_states, p=obs_mat[z])
        return obs

    keys = jax.random.split(rng_obs, seq_len)
    obs_seq = jax.vmap(draw_obs, in_axes=(0, 0))(state_seq, keys)

    return state_seq, obs_seq


# In[9]:


seq_len = 20
state_seq, obs_seq = hmm_sample(params, seq_len, PRNGKey(1))
print(state_seq)
print(obs_seq)

