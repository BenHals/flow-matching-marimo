import marimo
import marimo as mo
import numpy as np
import torch

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import math

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    import lib.data
    import lib.models
    import lib.ode
    import lib.vector_field

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
    n_steps = 25
    vf = lib.vector_field.ConvergingVectorField(theta_slider.value / 100.0)
    traj = lib.ode.solve_for_trajectory_at_timesteps(
        init_location=x_init,
        vector_field=vf,
        timesteps=[i / n_steps for i in range(n_steps)],
    )
    traj
    return n_steps, traj, vf


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
    _ts = traj.timesteps[idx_in_view]
    vals = traj.locations[idx_in_view][:, 0]

    fig, ax = plt.subplots()
    ax.plot(_ts, vals)
    ax.set_xlim((0, 1))
    ax.set_ylim((-2, 2))
    ax

    return (t,)


@app.cell
def _(lib, n_steps, np, plt, t, vf):
    n_trajectories = 20
    trajectory_range = (-2, 2)
    _ts = [i / n_steps for i in range(n_steps)]
    trajectories = [
        lib.ode.solve_for_trajectory_at_timesteps(
            init_location=i / 100, vector_field=vf, timesteps=_ts
        )
        for i in range(
            trajectory_range[0] * 100,
            trajectory_range[1] * 100,
            int(
                (trajectory_range[1] * 100 - trajectory_range[0] * 100) / n_trajectories
            ),
        )
    ]
    _, ax_traj = plt.subplots()

    for _traj in trajectories:
        _idx_in_view = np.where(_traj.timesteps < t)
        _ts = _traj.timesteps[_idx_in_view]
        _vals = _traj.locations[_idx_in_view]
        ax_traj.plot(_ts, _vals)
    ax_traj.set_xlim((0, 1))
    ax_traj.set_ylim((-2, 2))
    ax_traj
    return


@app.cell
def _(lib, mo, np):
    spiral_trajectories = []
    init_dataset = lib.data.make_spiral(3, 300)
    final_dataset = lib.data.make_gaussian(300)
    _n_steps = 1000
    spiral_slider = mo.ui.slider(
        start=1, stop=_n_steps - 1, step=5, full_width=True, label="timestep"
    )
    for j in range(300):
        z = lib.data.sample_dataset(init_dataset)
        start = lib.data.sample_dataset(final_dataset)

        vector_field = lib.vector_field.GaussianConditionalVectorField(
            alpha=lib.alpha_beta.AlphaBasic(), beta=lib.alpha_beta.BetaBasic(), z=z
        )

        trajectory = lib.ode.solve_for_trajectory_at_timesteps(
            start, vector_field, [i / _n_steps for i in range(_n_steps)]
        )
        spiral_trajectories.append(trajectory)
    spiral_trajectory_locations = np.stack([t.locations for t in spiral_trajectories])
    spiral_trajectory_locations.shape

    return spiral_slider, spiral_trajectory_locations


@app.cell
def _(plt, spiral_trajectory_locations):
    spiral_fig, spiral_ax = plt.subplots()

    spiral_scatter = spiral_ax.scatter(
        spiral_trajectory_locations[:, 0, 0],
        spiral_trajectory_locations[:, 0, 1],
        animated=True,
    )
    spiral_ax.set_xlim((-2, 2))
    spiral_ax.set_ylim((-2, 2))
    spiral_ax.set_axis_off()
    return spiral_ax, spiral_scatter


@app.cell
def _(spiral_slider):
    spiral_slider
    return


@app.cell
def _(spiral_ax, spiral_scatter, spiral_slider, spiral_trajectory_locations):
    spiral_scatter.set_offsets(spiral_trajectory_locations[:, spiral_slider.value])
    spiral_ax

    return


@app.cell
def _(lib):
    xlim = [-1.0, 1.0]
    ylim = [-1.0, 1.0]
    n_grid_points = 25
    n_timesteps = 20
    n_simulations = 500

    grid_points = np.meshgrid(
        np.linspace(xlim[0], xlim[1], n_grid_points),
        np.linspace(ylim[0], ylim[1], n_grid_points),
    )
    grid_points = torch.from_numpy(
        np.stack((grid_points[0], grid_points[1]), -1).reshape((-1, 2))
    ).float()

    spiral_vector_field_model = lib.models.load_pretrained_spiral_model()

    timesteps = np.linspace(0.0, 1.0, n_timesteps)
    step_length = timesteps[1] - timesteps[0]

    v_t = [
        spiral_vector_field_model(
            grid_points,
            torch.tensor(t).view((1, 1)).expand((grid_points.shape[0], 1)).float(),
        ).detach()
        for t in timesteps
    ]

    sim_t = torch.randn((n_simulations, 2))
    sim_locations = [sim_t]
    for t_prev, t_next in zip(timesteps[:-1], timesteps[1:]):
        print(sim_locations)
        x_t = sim_locations[-1]
        x_next = (
            x_t
            + step_length
            * spiral_vector_field_model(
                x_t,
                torch.tensor(t_prev).view((1, 1)).expand((n_simulations, 1)).float(),
            ).detach()
        )
        sim_locations.append(x_next)

    return v_t, timesteps


@app.cell
def _(lib):
    sim_slider = mo.ui.slider(0, n_timesteps - 1, full_width=True)
    sim_slider
    return


@app.cell
def _(lib):
    sim_fig, _ax = plt.subplots()
    _ax.quiver(
        grid_points[:, 0],
        grid_points[:, 1],
        v_t[sim_slider.value][:, 0],
        v_t[sim_slider.value][:, 1],
    )
    _ax.scatter(
        sim_locations[sim_slider.value][:, 0],
        sim_locations[sim_slider.value][:, 1],
    )
    _ax.set_xlim(xlim)
    _ax.set_ylim(ylim)
    sim_fig
    return


if __name__ == "__main__":
    app.run()
