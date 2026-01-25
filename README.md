# How Seasonality and Intertidal Positioning Impact Anemone *Anthopleura sola* Symbiosis with Photosynthetic Algae 
An analysis on a population of solitary anemones, measuring the health and status of their symbiotic relationship with alga *Breviolum muscatinei* under the backdrop of seasonal changes and multiple, distinct atmospheric rivers  

**Motivation**: Study how the symbiotic relation between the starburst anemone and its brown algae varies within a population and within individuals over seasonal changes.

**Author**: Jesse Espinoza

---

## Repository Structure

```text
sfsu-masters-anthopleura-sola/
├── abiotic_data/      # Cleaned temperature and salinity data from both on and off site loggers
                       # Also includes cleaned terrestrial precipitation data
├── biotic_data/       # Processed symbiotic dependent variables (algal density, chlorophyll concentration)
├── figures/           # Figures at various stages (pre/post annotations and final formatted TIFF figures)
├── notebooks/         # Jupyter notebooks for abiotic/biotic analyses and visualizations
├── utils/             # Python modules containing functions used in the analysis and visualizations
├── environment.yml    # Conda environment specification
└── README.md          # This file
```

---

## Installation Instructions

This project uses a Conda environment defined in `environment.yml`. Follow these steps to set it up:

### 1. Install Conda

Install either [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/).

To verify Conda is installed:
```bash
conda --version
```

### 2. Create and Activate the Environment

From the repository root, create the environment:
```bash
conda env create -f environment.yml
```

Then activate it:
```bash
conda activate asola-analysis
```

### 3. (Optional) Install Mamba for Faster Package Management

Mamba is a faster alternative to conda that speeds up environment solving and package installation:

```bash
conda install mamba -c conda-forge -y
mamba env update --file environment.yml --prune -y
```

### 4. Clean Up (Optional)

Remove unnecessary packages and caches:
```bash
conda clean --all -y
```

You're now ready to run the analysis!
---

## Usage

Open the Jupyter notebooks in the `notebooks/` directory to run the analyses:

```bash
jupyter notebook
```

Or from VS Code, select the `asola-analysis` kernel when opening any `.ipynb` file.

---

## Project Overview

This analysis examines *Anthopleura sola* (starburst anemone) populations and their symbiotic relationship with *Breviolum muscatinei* (brown algae). Key measurements include:

- **Symbiont Density**: Quantification of algal cells within anemone tissue
- **Chlorophyll Concentration**: Assessment of photosynthetic capacity
- **Temporal Variation**: Changes across seasons and atmospheric river events
- **Environmental Factors**: Temperature, salinity, and precipitation data

---

## Notes

- All data files are placed in `abiotic_data/` and `biotic_data/` directories
- Output figures are organized in `figures/` with subdirectories for draft and final versions
- The `utils/` module contains reusable functions—import them in notebooks with `from utils.functions import ...`