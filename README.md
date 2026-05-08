# LocalMorphQ — Getting Started

This guide explains how to set up the environment and reproduce the data and figures from the paper.

---

## Requirements

- Python 3.10 or higher
- Git

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
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
├── main.py                   # Data processing entry point
├── Processing.py             # Core processing functions
├── Utils.py                  # General utility functions
├── GraVisPreprocessing.py    # GraVis preprocessing pipeline
├── GraVisExtraction.py       # GraVis extraction pipeline
├── GraVisUtils.py            # GraVis utility functions
├── Figure1.py                # Figure 1 reproduction
├── Figure2.py                # Figure 2 reproduction
├── Figure3.py                # Figure 3 reproduction
├── Figure4.py                # Figure 4 reproduction
├── FigureSupplementary.py    # Supplementary figures reproduction
├── requirements.txt          # Python dependencies
└── README.md
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

This will generate the processed output files required for figure reproduction.

### Step 2 — Reproduce the figures

Each figure script takes the path to the processed data folder as input. Run them individually:

```bash
python Figure1.py --data /path/to/data
python Figure2.py --data /path/to/data
python Figure3.py --data /path/to/data
python Figure4.py --data /path/to/data
python FigureSupplementary.py --data /path/to/data
```

Figures will be saved to the output folder defined in your config file.

---

## Deactivating the virtual environment

Once you are done, deactivate the virtual environment with:

```bash
deactivate
```
