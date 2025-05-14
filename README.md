# ai-rec-me

## Overview


## Prerequisites
Confirm you have Python 3 installed
- Run `python3 --version`
- If you don't see something like `Python 3.10.x or higher`, install or upgrade via Homebrew with `brew install python`

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Directories:
- `data/raw/`: Store original CSVs, JSONs, etc., exactly as you downloaded them.
- `data/processed/`: Put cleaned or transformed files here (e.g. after tokenizing or filtering)
- `notebooks/`: You'll use Jupyter notebooks here for exploration and prototyping
- `src/data/`: Scripts to load and preprocess
- `src/models/`: Training and evaluation logic
- `src/utils/`: Any helper functions (e.g., metrics, plotting)
- `tests/`: Write automated tests here
- `.venv/`: Contains an isolated Python interpreter and site packages
  - Generate by running `python3 -m venv .venv`

## Activating/Deactivating the Virtual Environment
- Activate: Once you have your `.venv/` directory, run `source .venv/bin/activate`
  - `(.venv)` should appear in your terminal
  - Activating ensures that any `pip install` or `python` command uses only the packages inside `.venv`, keeping the project's deps isolated.
- Deactivate: `deactivate`

## With venv Active
- Make sure you have the latest installer: `pip install --upgrade pip`
  - You should see `Successfully install pip-XX.X.X`

### Install code dependencies
- Data handling: `numpy`, `pandas`
- Plotting: `matplotlib`
- Collaborative filtering: `scikit-surprise`
- General ML: `scikit-learn`
- Interactive exploration: `jupyterlab`
- Testing: `pytest`
- Code Formatting `black`
- Linting: `flake8`
- Install with `pip install` in venv

### Pin your dependencies
- Freeze the exact versions you install so anyone (or any CI) can reproduce the environment
- `pip freeze > requirements.txt`


