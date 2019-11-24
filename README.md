# Reservoir Memory Machines

Copyright (C) 2019  
Benjamin Paaßen, Alexander Schulz  
AG Machine Learning  
Bielefeld University  

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

## Introduction

This is a reference implementation of _reservoir memory machines_ (RMMs)
as proposed in the ESANN 2020 paper of the same name (submitted).
We also provide here the experimental code for the experiments shown in the
paper.

## Quickstart Guide

To train your own RMM, use the following code.

```
from rmm.rmm import RMM
model = RMM(m = my_number_of_neurons, K = my_memory_size)
model.fit(X, Y)
Yhat = model.predict(X)
```

where `X` is the input time series and `Y` is the desired output time series,
both with time steps as rows.

## Contents

This repository contains the following files.

* `copy.ipynb` : The experimental code for the _copy_ task from the paper.
* `latch.ipynb` : The experimental code for the _latch_ task from the paper.
* `LICENSE.md` : A copy of the [GNU General Public License, Version 3][GPLv3].
* `README.md` : This file.
* `repeat_copy.ipynb` : The experimental code for the _repeat copy_ task from
    the paper.
* `rmm/crj.py` : An implementation of [cycle reservoirs with jumps (Rodan and Tiňo, 2012)][CRJ].
* `rmm/esn.py` : An implementation of [echo state networks (Jaeger and Haas, 2004)][ESN].
* `rmm/hyperopt.py` : Utility functions for hyperparameter optimization.
* `rmm/rmm.py` : An implementation of reservoir memory machines.
* `runtime.ipynb` : The experimental code for the runtime experiments from the
    paper.
* `tests/crj_test.py` : Unit tests for `rmm/crj.py`.
* `tests/esn_test.py` : Unit tests for `rmm/esn.py`.
* `tests/hyperopt_test.py` : Unit tests for `rmm/hyperopt.py`.
* `tests/rmm_test.py` : Unit tests for `rmm/rmm.py`.

## Licensing

This library is licensed under the [GNU General Public License Version 3][GPLv3].

## Dependencies

This library depends on [NumPy][np] for matrix operations, on [scikit-learn][scikit]
for the base interfaces and on [SciPy][scipy] for optimization.

[scikit]: https://scikit-learn.org/stable/ "Scikit-learn homepage"
[np]: http://numpy.org/ "Numpy homepage"
[scipy]: https://scipy.org/ "SciPy homepage"
[GPLv3]: https://www.gnu.org/licenses/gpl-3.0.en.html "The GNU General Public License Version 3"
[CRJ]:https://doi.org/10.1162/NECO_a_00297 "Rodan and Tino (2012). Simple Deterministically Constructed Cycle Reservoirs with Regular Jumps. Neural Compuation, 24(7), 1822-1852. doi:10.1162/NECO_a_00297"
[ESN]:https://doi.org/10.1126/science.1091277 "Jaeger and Haas (2004). Harnessing nonlinearity: Predicting chaotic systems and saving energy in wireless communication. Science, 304(5667), 78-80. doi:10.1126/science.1091277"
