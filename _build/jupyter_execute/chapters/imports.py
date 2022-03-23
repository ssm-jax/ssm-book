#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Install necessary libraries


get_ipython().run_line_magic('pip', 'install --upgrade pip')

try:
    import jax
except:
    # For cuda version, see https://github.com/google/jax#installation
    get_ipython().run_line_magic('pip', 'install --upgrade "jax[cpu]"')
    import jax

try:
  import optax
except:
  get_ipython().run_line_magic('pip', 'install --upgrade git+https://github.com/deepmind/optax.git')
  import optax

try:
    import jaxopt
except:
    get_ipython().run_line_magic('pip', 'install  --upgrade git+https://github.com/google/jaxopt.git')
    import jaxopt


try:
    import flax
except:
    get_ipython().run_line_magic('pip', 'install --upgrade git+https://github.com/google/flax.git')
    import flax

try:
    import distrax
except:
    get_ipython().run_line_magic('pip', 'install --upgrade git+https://github.com/deepmind/distrax.git')
    import distrax

try:
    import blackjax
except:
    get_ipython().run_line_magic('pip', 'install --upgrade git+https://github.com/blackjax-devs/blackjax.git')
    import blackjax

try:
    import jsl
except:
    get_ipython().run_line_magic('pip', 'install git+https://github.com/probml/jsl')
    import jsl


# In[2]:


import abc
from dataclasses import dataclass
import functools
import itertools

from typing import Any, Callable, NamedTuple, Optional, Union, Tuple

import inspect
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

def print_source(fname):
    print('source code of ', fname)
    #txt = inspect.getsource(fname)
    (lines, line_num) = inspect.getsourcelines(fname)
    for line in lines:
        print(line.strip('\n'))


# In[ ]:



   

