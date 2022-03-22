---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Kevin's noodling

In this chapter, we do blah.
For more details, see  [](sec:bar), where we discuss bar.


## Python

```{code-cell}
from matplotlib import rcParams, cycler
import matplotlib.pyplot as plt
import numpy as np
plt.ion()
```

```{code-cell}
# Fixing random state for reproducibility
np.random.seed(19680801)

N = 10
data = [np.logspace(0, 1, 100) + np.random.randn(100) + ii for ii in range(N)]
data = np.array(data).T
cmap = plt.cm.coolwarm
rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, N)))


from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4),
                Line2D([0], [0], color=cmap(.5), lw=4),
                Line2D([0], [0], color=cmap(1.), lw=4)]

fig, ax = plt.subplots(figsize=(10, 5))
lines = ax.plot(data)
ax.legend(custom_lines, ['Cold', 'Medium', 'Hot']);
```

## Images

[](https://myst-parser.readthedocs.io/en/latest/_static/logo-wide.svg)

![](https://myst-parser.readthedocs.io/en/latest/_static/logo-wide.svg)

<img src="https://github.com/probml/probml-notebooks/blob/main/images/cat_dog.jpg"
style="height:200">


## Math

$$
a x^2 + bx+ c = 0
$$

$$
\begin{align}
0 &= a x^2 + bx+ c \\
0 &= a x^2 + bx+ c 
\end{align}
$$

## Refs

For more details, see {cite}`holdgraf_evidence_2014` and foo.

