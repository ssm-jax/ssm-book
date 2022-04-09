#!/usr/bin/env python
# coding: utf-8

# # Boilerplate

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


# Import standrd libraries

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


# # Hidden Markov Models
# 
# We first create the "Ocassionally dishonest casino" model from {cite}`Durbin98`.
# 
# ```{figure} /figures/casino.png
# :scale: 50%
# :name: casino
# 
# Illustration of the casino HMM.
# ```
