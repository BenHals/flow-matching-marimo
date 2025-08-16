import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import math
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import lib.data
    import lib.vector_field
    import lib.ode

    return lib, mo, np, plt


@app.cell
def _(mo):
    n_spirals = mo.ui.slider(start=1, stop=6)

    n_spirals

    return (n_spirals,)


@app.cell
def _(lib, n_spirals):
    points = lib.data.make_spiral(n_spiral_arms=n_spirals.value, n_points=250)

    return (points,)


@app.cell
def _(plt, points):
    plt.scatter([p[0] for p in points], [p[1] for p in points])
    return


@app.cell
def _(lib, points):
    val = lib.data.sample_dataset(points)
    val
    return


@app.cell
def _(lib):
    ds = lib.data.make_gaussian(n_points=100)
    return (ds,)


@app.cell(hide_code=True)
def _(mo):
    resample_button = mo.ui.button(label="Resample")

    return (resample_button,)


@app.cell(hide_code=True)
def _(ds, lib, resample_button):
    resample_button.value
    x_init = lib.data.sample_dataset(ds)

    return (x_init,)


@app.cell(hide_code=True)
def _(mo):
    theta_slider = mo.ui.slider(start=1, stop=100, label="theta")

    return (theta_slider,)


@app.cell(hide_code=True)
def _(lib, theta_slider, x_init):
    n_steps = 100
    vf = lib.vector_field.ConvergingVectorField(theta_slider.value / 100.0)
    traj = lib.ode.solve_for_trajectory_at_timesteps(init_location=x_init, vector_field=vf, timesteps=[i / n_steps for i in range(n_steps)])
    traj
    return (traj,)


@app.cell
def _(mo):
    t_slider = mo.ui.slider(start=1, stop=100, label="timestep")

    return (t_slider,)


@app.cell
def _(mo, resample_button, t_slider, theta_slider):
    mo.vstack([resample_button, theta_slider, t_slider])
    return


@app.cell(hide_code=True)
def _(np, plt, t_slider, traj):

    t = t_slider.value / 100.0
    idx_in_view = np.where(traj.timesteps < t)
    ts = traj.timesteps[idx_in_view]
    vals = traj.locations[idx_in_view][:, 0]

    fig, ax = plt.subplots()
    ax.plot(ts, vals)
    ax.set_xlim((0, 1))
    ax.set_ylim((-2, 2))
    ax

    return


if __name__ == "__main__":
    app.run()
