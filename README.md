# LocalMorphQ — Getting Started

This guide explains how to set up the environment and reproduce the data and figures from the paper: 

Nowak et al., <em>Unraveling the actin cytoskeletal role in the morphogenesis of pavement cells using mask-based approaches.</em> (Submitted)

---

## Requirements

- Python 3.10 or higher
- Git

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/jnowak90/LocalMorphQ
cd LocalMorphQ
```

### 2. Create and activate a virtual environment

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Repository structure

```
├── main.py
├── requirements.txt
├── README.md
│
├── config/
│   ├── config_Example.yaml
│   └── config_Template.yaml
│
├── src/
│   ├── __init__.py
│   ├── Processing.py
│   ├── Utils.py
│   ├── GraVisExtraction.py
│   ├── GraVisPreProcessing.py
│   └── GraVisUtils.py
|
├── data/
│   ├── AFs/
│   └── MTs/
|
└── figures/
    ├── Figure1.py
    ├── Figure2.py
    ├── Figure3.py
    ├── Figure4.py
    └── FiguresSupplementary.py
```

---

## Usage

### Step 1 — Process the data

Run `main.py` with the path to your data folder and your config file:

```bash
python main.py --data /path/to/data --config /path/to/config.yaml
```

| Argument | Description |
|---|---|
| `--data` | Path to the input data folder |
| `--config` | Path to the YAML configuration file |


To test the code, use:
```bash
python main.py --data data/ --config config/config_Example.yaml
```

### Step 2 — Reproduce the data

This will generate the processed output files required for figure reproduction.

To replicate the data provided in the study, first download the data from Dryad and save in the data/ folder. Copy the config files into the config/ folder.

Then, use:
```bash
python main.py --data data/Dryad/Cotyledons/ --config config/config_Cotyledons.yaml
python main.py --data data/Dryad/Leaves/ --config config/config_Leaves.yaml
```

### Step 3 — Reproduce the figures

Each figure script takes the path to the processed data folder as input. Run them individually:

```bash
python figures/Figure1.py --data data/Dryad
python figures/Figure2.py --data data/Dryad
python figures/Figure3.py --data data/Dryad
python figures/Figure4.py --data data/Dryad
python figures/FigureSupplementary.py --data /path/to/data
```

Figures will be saved to a separate 'Plots' folder.

---

## Deactivating the virtual environment

Once you are done, deactivate the virtual environment with:

```bash
deactivate
```
