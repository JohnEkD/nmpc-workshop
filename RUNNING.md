# Running the notebooks

The four notebooks sit at the top level of the repository. Exercises 0a and 0b also
need the do-mpc library, which `requirements.txt` includes and its setup cell
installs in Colab. Beside them are two
folders: `lib/` holds the code the notebooks use (`robot_lab.py`,
`drone_lab.py` and the shared `common.py`), and `videos/` holds the videos they
play. Keep all of these together. Each notebook's first cell uses the files in
`lib/`, or, if one is missing, downloads it from this repository.

## Colab

Click an "open" link in the README and run the first cell. It installs what is
needed, fetches the files in `lib/` and sets up 3-D rendering. It should end by
printing `ready`.

The opening video and the short animations play from `videos/`, which Colab
fetches from this repository when they are needed.

## VS Code

1. Install the **Python** and **Jupyter** extensions from Microsoft.
2. `File > Open Folder` and choose the top of this repository.
3. Make an environment and install what is needed. In the VS Code terminal:

       python -m venv .venv
       .venv\Scripts\activate        # Windows
       source .venv/bin/activate     # macOS and Linux
       pip install -r requirements.txt

4. Open a notebook. Top right, click **Select Kernel**, then
   **Python Environments**, then `.venv`.
5. Run the first cell. It should print `ready`. The `%pip install` line in it
   does nothing harmful: everything is already installed.
6. Work through the notebook cell by cell. **Run All** stops at the first task,
   which is where you start.

### If the first cell says the library was not found

The `lib` folder, holding `common.py` and `robot_lab.py` (or `drone_lab.py`),
must sit beside the notebook, and the notebook must run from its own folder. VS Code does this by default. If
you have changed it, set `"jupyter.notebookFileRoot": "${fileDirname}"` in the
VS Code settings.

### The first run downloads the models

About 15 MB from the DeepMind MuJoCo Menagerie, into `mujoco_menagerie/` beside
the notebooks. After that they are cached and everything works offline, apart
from the videos if the `videos` folder is missing.

### Videos and animations

The `imageio[ffmpeg]` package in `requirements.txt` includes its own ffmpeg, so
nothing else is needed.

### Rendering

On your own computer MuJoCo usually picks a working graphics backend by itself.
If 3-D rendering fails, run one of these **before** the first cell, then
restart the kernel:

    import os; os.environ["MUJOCO_GL"] = "egl"       # Linux with a GPU
    import os; os.environ["MUJOCO_GL"] = "osmesa"    # Linux without a screen

## Jupyter Lab

    pip install -r requirements.txt
    jupyter lab

## Anaconda

    conda env create -f environment.yml
    conda activate nmpc-workshop
    jupyter lab

`mujoco` and `casadi` are installed with pip inside the conda environment,
because their authors publish the pip versions first. The file installs the
conda packages first, which is the safe order.

To use the environment in VS Code, pick `nmpc-workshop` under **Select Kernel**.
If it is not listed, run `python -m ipykernel install --user --name
nmpc-workshop` with the environment active, then reload the window.

**Caution:** `environment.yml` has not been built on a clean machine. Build it
once before a session, because a package problem in front of a class is hard
to recover from. `requirements.txt` was tested in a clean environment with
Python 3.12, and both notebooks ran without errors.

## Tested with

Python 3.12, MuJoCo 3.13, CasADi 3.8, NumPy 2.5, Matplotlib 3.11.
