#!/usr/bin/env python
# coding: utf-8

# # Implementing discrete HMMs in Numpy 
# 
# We start with a simple numpy implementation.

# In[1]:


import jax
import numpy as np
import jax.numpy as jnp

from dataclasses import dataclass


# In[2]:


import jsl


# In[3]:


def normalize(u, axis=0, eps=1e-15):
    u = np.where(u == 0, 0, np.where(u < eps, eps, u))
    c = u.sum(axis=axis)
    c = np.where(c == 0, 1, c)
    return u / c, c


# We first create the "Ocassionally dishonest casino" model from {cite}`Durbin98`.

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

pi = np.array([1, 1]) / 2


# Let's bundle the parameters into a structure.

# In[5]:


@dataclass
class HMMNumpy:
    trans_mat: np.array  # A : (n_states, n_states)
    obs_mat: np.array  # B : (n_states, n_obs)
    init_dist: np.array  # pi : (n_states)
        


# In[6]:


(nstates, nobs) = jnp.shape(B)
for i in range(nstates):
    A[i,:] = normalize(A[i,:])[0]
    B[i,:] = normalize(B[i,:])[0]
    
params_numpy = HMMNumpy(A, B, pi)
print(params_numpy)


# Function to sample a single sequence of hidden states and discrete observations.

# In[7]:


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

    for t in range(1, seq_len):
        #print(t, zt, trans_mat[zt])
        state_seq, zt = sample_one_step_(state_seq, latent_states, trans_mat[zt])
        obs_seq, xt = sample_one_step_(obs_seq, obs_states, obs_mat[zt])

    return state_seq, obs_seq


# In[8]:


seq_len = 20
state_seq, obs_seq = hmm_sample_numpy(params_numpy, seq_len, random_state=0)
print(state_seq)
print(obs_seq)


# In[9]:


#from jsl.hmm.hmm_numpy_lib import HMMNumpy, hmm_forwards_backwards_numpy, hmm_loglikelihood_numpy
import jsl.hmm.hmm_numpy_lib as hmm_np

state_seq, obs_seq = hmm_np.hmm_sample_numpy(params_numpy, seq_len, random_state=0)
print(state_seq)
print(obs_seq)

    


# In[ ]:





# In[ ]:




