# <img src="https://raw.githubusercontent.com/willwerj/gliquid_python/main/assets/gliq_logo_tall.png" width="300" height="430">
# G<sub>Liquid</sub>: DFT-Referenced Thermodynamic Modeling

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/willwerj/gliquid_python/blob/main/LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/willwerj/gliquid_python/blob/main/notebooks/colab_demo.ipynb)

G<sub>Liquid</sub> fits **DFT-referenced liquid free energies** to binary phase diagrams. From
those fits it interpolates ternary liquidus surfaces and predicts liquid parameters for
unassessed systems with a bundled machine-learning model. Browse the fitted and predicted
systems in the [interactive matrix](https://viz.whsunresearch.group/gliquid/interactive-matrix.html).

## Installation

G<sub>Liquid</sub> supports Python 3.10–3.13. Install it into a fresh environment:

```bash
conda create --name gliquid-env python=3.12
conda activate gliquid-env
pip install gliquid
```

Optional extras:

| Extra | Install | Adds |
|---|---|---|
| `ml` | `pip install "gliquid[ml]"` | ML predictions of liquid parameters (`ProductionModelRunner`) |
| `mpds` | `pip install "gliquid[mpds]"` | downloading phase diagrams from MPDS that are not already cached |
| `notebook` | `pip install "gliquid[notebook]"` | Jupyter and the interactive widgets used in the notebooks |

### Data and notebooks

The package does not include per-system data (DFT convex hulls and digitized phase diagrams).
Clone the repository for the example notebooks and the cached data they use:

```bash
git clone https://github.com/willwerj/gliquid_python.git
```

Then point G<sub>Liquid</sub> at the clone's `cache/` folder:

```python
import gliquid.config as cfg
cfg.set_cache_dir("path/to/gliquid_python/cache")
```

Without this, systems are fetched as needed and saved to `./gliquid_cache`.

### API keys

To fit systems not in the repo cache, you'll need both MP and MPDS API keys.
If you only interpolate from the fitted parameters in our published dataset ([interactive matrix](https://viz.whsunresearch.group/gliquid/interactive-matrix.html)), you only need the MP API key.

- **Materials Project** (DFT data): get a key at [materialsproject.org](https://next-gen.materialsproject.org/api).
- **MPDS** (phase diagrams): needs the `mpds` extra and an MPDS account (paid subscription at [mpds.io](https://mpds.io/#start)).

```python
import os
os.environ["NEW_MP_API_KEY"] = "your Materials Project key"
os.environ["MPDS_API_KEY"] = "your MPDS key"
```

## Quick start

```python
from gliquid.binary import BinaryLiquid, BLPlotter

bl = BinaryLiquid.from_cache("Cu-Mg")
fits = bl.fit_parameters(n_opts=5)
best = min(fits, key=lambda fit: fit["mae"])
print(f"liquidus MAE {best['mae']:.0f} K, MAPE {best['mape']:.1f}%")

BLPlotter(bl).show("fit+liq")
```

The [notebooks](https://github.com/willwerj/gliquid_python/tree/main/notebooks) walk through fitting (`fitting_demo`), ternary interpolation
(`interpolation_demo`) and adjusting a prediction to experiment (`GaRu_adjustment`,
`PbTb_adjustment`). To run them locally, install the `notebook` extra and start Jupyter from
the `notebooks` folder.

## Google Colab

Open [colab_demo.ipynb](https://colab.research.google.com/github/willwerj/gliquid_python/blob/main/notebooks/colab_demo.ipynb)
for a complete example that runs without API keys. Each of the other notebooks has a commented
Colab setup cell at the top; uncomment it to run that notebook on Colab. For use in Colab, install
G<sub>Liquid</sub> without extras: Colab already has XGBoost and the notebook widgets.

## Citing

If G<sub>Liquid</sub> contributes to published work, please cite it using
[CITATION.cff](https://github.com/willwerj/gliquid_python/blob/main/CITATION.cff) (GitHub's "Cite this repository" button).

## Contributing

Pull requests are welcome. [CONTRIBUTING.md](https://github.com/willwerj/gliquid_python/blob/main/CONTRIBUTING.md) covers development setup and
running the tests. Release history is in [CHANGELOG.md](https://github.com/willwerj/gliquid_python/blob/main/CHANGELOG.md).

## License

[MIT](https://github.com/willwerj/gliquid_python/blob/main/LICENSE)

## Acknowledgements

This project is made possible by funding from the U.S. Department of Energy (DOE) Office of
Science, Basic Energy Sciences Award No. DE-SC0021130 and the National Science Foundation (NSF)
Award No. OAC-2209423.
