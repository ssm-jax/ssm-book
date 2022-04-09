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

try:
    import rich
except:
    get_ipython().run_line_magic('pip', 'install rich')
    import rich
    


# In[2]:


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


# In[3]:



   
def print_source_old(fname):
    print('source code of ', fname)
    #txt = inspect.getsource(fname)
    (lines, line_num) = inspect.getsourcelines(fname)
    for line in lines:
        print(line.strip('\n'))


# In[4]:


import jsl
import jsl.hmm.hmm_numpy_lib as hmm_lib_np
#import jsl.hmm.hmm_lib as hmm_lib_jax

normalize = hmm_lib_np.normalize_numpy
print_source(normalize)
#print_source(hmm_lib_np.normalize_numpy)

