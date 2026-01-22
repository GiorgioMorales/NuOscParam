[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GiorgioMorales/NuOscParam/blob/main/DemoSHTransformer.ipynb)


# Neutrino Parameter Estimation from Oscillation Probability Maps

Neutrino oscillations encode fundamental information about neutrino masses and mixing parameters, offering a unique window into physics beyond the Standard Model.
Estimating these parameters from oscillation probability maps is, however, computationally challenging due to the maps’ high dimensionality and nonlinear dependence on the underlying physics.
In this work, we introduce a data-driven framework that reformulates atmospheric neutrino oscillation parameter inference as a supervised regression task over structured oscillation maps. 
We propose a hierarchical transformer architecture that explicitly models the two-dimensional structure of these maps, capturing angular dependencies at fixed energies and global correlations across the energy spectrum. 
To improve physical consistency, the model is trained using a surrogate simulation constraint that enforces agreement between the predicted parameters and the reconstructed oscillation patterns. 
Furthermore, we introduce a neural network-based uncertainty quantification mechanism that produces distribution-free prediction intervals with formal coverage guarantees. 


<figure style="display: flex; flex-direction: column; align-items: center;">
    <img src="results/Neutrino.jpg" alt="figure" width="80%">
    <figcaption style="text-align: center; margin-top: 5px; font-style: italic;">
        Overview of the proposed $\nu$ oscillation parameters estimation.
    </figcaption>
</figure>


## Installation

The following libraries have to be installed:
* [Git](https://git-scm.com/download/) 
* [Pytorch](https://pytorch.org/get-started/locally/)

To install the package, run `pip install -q git+https://github.com/GiorgioMorales/NuOscParam` in the terminal.

## Usage

### Generate Neutrino Oscillation Maps

We use a simulator that generates 9 oscillation probability maps:

| Transition ↓ / Source → | $\nu_e$ (source) | $\nu_\mu$ (source) | $\nu_\tau$ (source) |
|--------------------------|------------------|--------------------|---------------------|
| **$\nu_e$ (detected)**  | $P(\nu_e \leftarrow \nu_e)$ | $P(\nu_e \leftarrow \nu_\mu)$ | $P(\nu_e \leftarrow \nu_\tau)$ |
| **$\nu_\mu$ (detected)** | $P(\nu_\mu \leftarrow \nu_e)$ | $P(\nu_\mu \leftarrow \nu_\mu)$ | $P(\nu_\mu \leftarrow \nu_\tau)$ |
| **$\nu_\tau$ (detected)** | $P(\nu_\tau \leftarrow \nu_e)$ | $P(\nu_\tau \leftarrow \nu_\mu)$ | $P(\nu_\tau \leftarrow \nu_\tau)$ |

In all cases, the $\nu$ oscillation parameters follow the order: `[theta12, theta23, theta13, delta_cp, m21, m31]`.

**Generate Maps Using Known Oscillation Parameters**

To use a exact simulator, initiate the `OscIterableDataset` class with the following parameters:

**Parameters** (for now):

*   `mode`: It can take the values `vacuum`, which will use a simulator that produces $\nu$ oscillation maps in vacuum, or `earth` (default), which will use a simulator that produces $\nu$ oscillation maps after Earth-matter effect. 
*   `cropR`: It determines the number of rows ($\theta$) bins are used. Min:1, Max:120, Default: 80.
*   `cropC`: It determines the number of columns (energy) bins are used. Min:1, Max:120, Default: 30.


```python
from NuOscParam.utils import *
from NuOscParam.Data.DataRanges import *
from NuOscParam.Data.Simulator import Simulator
generator = Simulator(ranges=NEUTRINO_RANGES, mode='earth', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
```

To get the actual maps given a set of $\nu$ oscillation parameters, we call the `get_maps` method.
The following example considers two sets of 6 $\nu$ oscillation parameters:

```python
import numpy as np

batch = np.array([[3.2434875e+01, 5.0723694e+01, 8.4253168e+00, 1.8559186e+02, 6.9126829e-05, 2.4357950e-03],
              [33.003803, 51.50117, 8.243288, 243.95, 7.993506e-05, 0.0024312385]])
maps = generator.get_maps(batch)
# Optionally, plot maps
for i in range(len(batch)):
    plot_osc_maps(maps[i, :, :, :].permute(1, 2, 0), title=f"Oscillation Maps. Sample {i+1}")
```

**Generate Random Maps**

To generate random $\nu$ oscillation maps and corresponding oscillation parameters, we use the
`OscIterableDataset` iterable class. Its parameters are the same as those from the `Simulator` class: 

```python
generator = iter(OscIterableDataset(ranges=NEUTRINO_RANGES, mode='earth',
                                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
```

To generate random maps, we call the `next` command iterably:

```python
X_test, Osc_params = [], []
n_samples = 3  # Generate three random maps
for i in range(n_samples):
    xtest, _, osc_pars, _ = next(generator)  # Generate 1 sample
    X_test.append(xtest)
    Osc_params.append(osc_pars)
    # Plot oscillation maps
    input_image = xtest[0, :, :, :].permute(1, 2, 0)
    plot_osc_maps(input_image, title=f"Oscillation Maps. Sample {i+1}")
X_test = torch.cat(X_test, dim=0)
```