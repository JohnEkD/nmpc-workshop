<img src="assets/logo.png" width="460">

# NMPC Workshop

Two exercises on nonlinear model predictive control, built line by line. Each
applies the method to a robot model from the DeepMind MuJoCo Menagerie: CasADi
solves the optimisation, MuJoCo provides the simulated plant.

| exercise | files | open in Colab |
| --- | --- | --- |
| mobile robot | `NMPC_robot_workshop.ipynb`, `lib/robot_lab.py`, `lib/common.py` | [open](https://colab.research.google.com/github/JohnEkD/nmpc-workshop/blob/main/NMPC_robot_workshop.ipynb) |
| quadrotor | `NMPC_drone_workshop.ipynb`, `lib/drone_lab.py`, `lib/common.py` | [open](https://colab.research.google.com/github/JohnEkD/nmpc-workshop/blob/main/NMPC_drone_workshop.ipynb) |

**New to this?** Exercises 0a and 0b cover the same two vehicles with the
controller supplied, and need no programming. They are in a separate
repository: [JohnEkD/nmpc-intro](https://github.com/JohnEkD/nmpc-intro).

Take the mobile robot before the quadrotor; the quadrotor notebook assumes it.
The robot notebook builds the method from nothing in five coding tasks. The
quadrotor notebook adds rigid-body dynamics and underactuation in three coding
tasks, and supplies the parts that are the same as before.

## Structure

```
NMPC_robot_workshop.ipynb   the notebooks: open these
NMPC_drone_workshop.ipynb
lib/                        the code the notebooks use: no need to open
videos/                     the videos the notebooks play
```

Each notebook opens with a short video of the finished controller: the
MuJoCo simulation beside the controller's own view. The videos are stored in
the `videos` folder, so they play at once; the same folder holds the short
animations in section 2 and 3 of the robot notebook. Keep that folder beside the
notebooks (on GitHub, at the top level of the repository). If it is
missing, the notebook makes the video itself, which takes up to half a minute.

Both notebooks follow the same sections.

1. The problem: the application, the system, the assumptions, the task
2. The model, built up a step at a time
3. The approach and the algorithm
4. Building the parts, one task per piece of the problem
5. The complete algorithm, assembled from your own functions
6. On the simulated robot or vehicle, in MuJoCo
7. The effect of the horizon
8. In a future workshop: what comes next
9. References

## Running it

**Colab.** Click a link above and run the first cell. It installs what is
needed and downloads the files it needs from `lib/` in this repository.

**Locally.** Clone the repository, then:

    pip install -r requirements.txt
    jupyter lab

`RUNNING.md` has the details for VS Code, Jupyter Lab and Anaconda
(`environment.yml`).

The robot models download on first use into `mujoco_menagerie/` and are cached.

## Working through it

Every coding task states the question, gives the equation, then leaves the
code cell blank. A worked solution sits underneath (with a hint, except for
the last quadrotor task), and a check must print **PASS** before you continue.

Tasks marked **Explain** ask for a sentence or two instead of code. Write your
answer in the cell provided, then open the answer underneath and compare.

Run the cells in order. **Run All will stop at the first task**, which is where
you start.

Values marked `EDIT` are meant to be changed.
