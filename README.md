A toolbox for maximum entropy analysis (MEM)

This toolbox provides a collection of functions for forward modelling of 
datasets using maximum entropy or Tikhonov regularization. The analysis is applicable to any
situation where a hidden variable produces a distribution of the
measured quantity. To avoid overfitting, the entropy (for MEM) or norm (fo Tikhonov regularization) of
the distribution of the hidden variable is maximized.

**How to use:**
The toolbox contains all required functions to solve the MEM problem as described in:

Vinogradov, S. A., & Wilson, D. F. (2000). Recursive Maximum Entropy Algorithm and its Application to the Luminescence Lifetime Distribution Recovery. Applied Spectroscopy, 54(6), 849-855. https://doi.org/10.1366/0003702001950210


See the Jupyer notebooks for exemplary inference of the distribution of 
fluorescence lifetime from fluorescence decays obtained by
time-correlated single photon couting (TCSPC), or the interdye distance distribution from
colocalization distances obtained from fluorescence imaging.

For other use cases, it is only required to define a custom kernel function.
All other steps of the workflow remain unchanged.