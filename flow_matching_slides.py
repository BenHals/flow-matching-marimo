import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import math
    from copy import deepcopy

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    import lib.data
    import lib.mnist
    import lib.models
    import lib.ode
    import lib.vector_field

    return deepcopy, lib, mo, np, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Generative Models
    Generative models learn the distribution of a data, $p(X)$. Usually the distribution we want to learn corresponds to some interesting subset of things in the world.

    For example, we could consider the set of all possible 2D points. This isn't particularly interesting to model. Instead, we might consider our target to be the distribution of 2D points that make up a spiral (shown below). 

    A generative model lets us sample points that fall into the target distribution, i.e., lets us generate new points in our spiral. Generative models might also let us directly calculate the probability of a given sample, e.g., calculate $p(X=x)$ for a given sample $x \in X$ (but not all generative models let us do this).
    """
    )
    return


@app.cell
def _(mo):
    n_spirals = mo.ui.slider(start=1, stop=6, label="Number of spiral arms:")
    return (n_spirals,)


@app.cell(hide_code=True)
def _(lib, mo, n_spirals, plt):

    _points = lib.data.make_spiral(n_spiral_arms=n_spirals.value, n_points=250)
    _spiral_fig1 = plt.scatter([p[0] for p in _points], [p[1] for p in _points])
    mo.md(f"{n_spirals} \n {mo.as_html(_spiral_fig1)}")
    return


@app.cell
def _(mo):
    mo.md(
        """
    # Challenge
    Generative modelling is difficult for a few reasons:
    1. The Normalizing constant: The target in generative modelling is learning the probability distribution $p(X)$. A probability distribution has a few constraints, the probabilities assigned should be positive and add to 1. Adding to 1 is the difficulty here, how should we enforce that over all possible outputs, the sum is 1? This is challenging when there is usually an infinite number of outputs, and we only have positive examples.
    2. The target to learn is much more detailed compared to a discriminative model, which may just need to learn a decision boundary between two classes. Often the discriminative target is in a much smaller dimension.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        f"""
    # Flow models
    A flow model is a generative modelling approach that tackles the challenge or normalization by starting with a simple 'known' distribution and transforming it into the target distribution in a way that maintains the properties that we want.

    For example, if we start with a Gaussian distribution then we know it meets the constraints of a probability distribution and we can easily sample from it and calculate statistics. If we can transform samples taken from a gaussian into samples from the target distribution, we can use this as a way to instead take samples from the target!

    Flow matching is a method of learning a model to do this transformation.

    ## Vector fields, flows and trajectories

    The basics of how a flow model works starts with a vector field. Consider a space at a point in time. A vector field describes how each point in that space will move between now and the next point in time. If we place an initial point into the space at a starting time $t_0$, then simulate how the vector field moves the point over time, we get a _trajectory_, a description of the point's position over time. If we placed an initial point at every possible starting point and simulated them all, that is a _flow_, essentially a collection of trajectories. This kind of simulation is an Ordinary Differential Equation (ODE), i.e., the solution to an ODE at a particular starting point is a trajectory.

    The idea of 'flow matching' is to learn a vector field that will transform all points in the initial distribution into samples from the target distribution if we solve the ODE from time $t_0$ to $t_1$.

    Below shows a 1D point (on the y-axis) over time (x-axis) as you move the timestep slider. The vector field in this case pushes the point toward 0, with a strength set by theta. The top plot shows a trajectory (use the resample button to select a random initial point) and the bottom plot is the flow.
    """
    )
    return


@app.cell
def _(lib, mo):
    ds = lib.data.make_gaussian(n_points=100)
    resample_button = mo.ui.button(label="Resample")
    theta_slider = mo.ui.slider(start=1, stop=100, label="theta")
    t_slider = mo.ui.slider(start=1, stop=100, value=10, label="timestep")


    return ds, resample_button, t_slider, theta_slider


@app.cell(hide_code=True)
def _(ds, lib, resample_button):
    resample_button.value
    x_init = lib.data.sample_dataset(ds)

    return (x_init,)


@app.cell(hide_code=True)
def _(lib, theta_slider, x_init):
    n_steps = 25
    vf = lib.vector_field.ConvergingVectorField(theta_slider.value / 100.0)
    traj = lib.ode.solve_for_trajectory_at_timesteps(
        init_location=x_init,
        vector_field=vf,
        timesteps=[i / n_steps for i in range(n_steps)],
    )
    return n_steps, traj, vf


@app.cell
def _(mo, resample_button, t_slider, theta_slider):
    mo.vstack([resample_button, theta_slider, t_slider])
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Trajectory
    This is a single trajectory showing how a point in 1D space moves from it's starting position over time (on the x-axis) as it follows a vector field.
    """
    )
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
def _(mo):
    mo.md(
        """
    ## Flow
    A flow is just the set of trajectories from all starting positions.
    """
    )
    return


@app.cell(hide_code=True)
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
def _(mo):
    mo.md(
        """
    # Flow Matching Model
    This plot shows the output of transforming samples from a unit gaussian into our target spiral distribution by solving the ODE of a learned vector field.

    Our model here learns the vector field function, where the input is a spatial location at a specific time, $u(x, t)$ and the output is the velocity at that position in time, i.e., how the point moves over the next timestep.
    """
    )
    return


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
def _(lib, np, torch):
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

    return grid_points, n_timesteps, sim_locations, v_t, xlim, ylim


@app.cell
def _(mo):
    mo.md(
        """
    ## Vector field
    This plot shows the learned vector field. Notice how it moves over time, initially it points towards the center to bring points in, then later on specific locations push outward to form the spiral arms.
    """
    )
    return


@app.cell
def _(mo, n_timesteps):
    sim_slider = mo.ui.slider(0, n_timesteps - 1, full_width=True)
    sim_slider
    return (sim_slider,)


@app.cell
def _(grid_points, plt, sim_locations, sim_slider, v_t, xlim, ylim):
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


@app.cell
def _(mo):
    mo.md(
        """
    # Conditional Flow Models
    Previously, we trained models that were just based on spatial location and time. However, sometimes we want to condition our model on something else, such as a class label.

    Here our target dataset is the MNIST handwritten digit dataset. Our class in this case is the digit being written. As a side note, we are now dealing with image data. While this seems quite different, and we need to plot it differently, it's fundamentally the same kind of data as the previous spiral dataset. Before, each data point was one 2D vector, and we had a 2D velocity pushing it. Now, with MNIST data each image is 28x28 pixels, or 784 total pixels. Now each datapoint is a 784 dimension vector, and it's velocity is the same. Rather than pushing points around a 2D plot, we can show velocity as positively or negatively changing the value of a single pixel (plot to the right below). Another difference is before we were plotting many points in the dataset to show how they converged to the overall distribution. With images, we can only show one at a time (imaging one point moving on it's own in the above plots), but we show that it ends up in the target distribution by visually looking correct.

    ## How do we train these models?

    In theory all we need to do is give the model another input, the conditioned class, during training. It can then learn the effect, and at inference time we can give it the class we want when solving the ODE. However, this usually doesn't lead to great results. Partly because the model will often just learn a good 'average' output to minimize loss. Instead, an approach called 'classifier-free guidance' mixes the velocity of a conditioned model (using class) and an unconditioned model (not using class). The idea is that the difference between these two values points in a direction that is _specifically_ going to move us in the direction of the class, i.e., a stronger conditioning signal. The nice thing about this approach is we don't even have to learn a second model, we can just have a 'null' class during training and mix it in during training!

    Use the drop down below to select a class conditioning signal at inference time.
    """
    )
    return


@app.cell(hide_code=True)
def _(lib, mo):
    mnist_model = lib.models.load_pretrained_mnist_model()
    mnist_class = mo.ui.dropdown([*range(10)])
    mnist_t = mo.ui.slider(0, 50, full_width=True)

    mo.md(f"{mnist_class} \n {mnist_t}")

    return mnist_class, mnist_model, mnist_t


@app.cell(hide_code=True)
def _(deepcopy, mnist_class, mnist_model, np, torch):
    _n_steps = 50
    _num_classes = 10
    classifier_free_guidance_mix = 3.0
    t_steps = np.linspace(0.0, 1.0, _n_steps + 1)
    step_size = t_steps[1] - t_steps[0]
    x_0 = torch.randn((1, 1, 28, 28)).float()
    _x_t = x_0
    class_tensor = (
        torch.tensor([mnist_class.value]) if mnist_class.value is not None else None
    )
    null_class = torch.tensor([_num_classes])
    _x_t_seq = [deepcopy(_x_t)]
    _v_seq = [torch.zeros_like(_x_t)]
    for _t in torch.from_numpy(t_steps).float():
        t_tensor = _t.view(1, 1).float()
        with torch.no_grad():
            unconditional_prediction = mnist_model(_x_t, t_tensor, null_class)

            if class_tensor is not None:
                conditional_prediction = mnist_model(_x_t, t_tensor, class_tensor)
                pred = unconditional_prediction + classifier_free_guidance_mix * (
                    conditional_prediction - unconditional_prediction
                )
            else:
                pred = unconditional_prediction
        _x_t += step_size * pred
        _x_t_seq.append(deepcopy(_x_t))
        _v_seq.append(deepcopy(pred))

    mnist_seq = _x_t_seq
    mnist_vs = _v_seq
    return mnist_seq, mnist_vs


@app.cell
def _(mnist_seq, mnist_t, mnist_vs, mo, plt):
    mnist_out = mnist_seq[mnist_t.value]
    mnist_out_v = mnist_vs[mnist_t.value]
    _fig, _axes = plt.subplots()
    _axes.imshow(mnist_out.reshape((28, 28)), cmap="grey")
    _fig_v, _axes_v = plt.subplots()
    _axes_v.imshow(mnist_out_v.reshape((28, 28)), cmap="plasma")
    _axes.axis("off")
    _axes_v.axis("off")

    mnist_fig = _fig
    mnist_fig_v = _fig_v

    mo.hstack([mnist_fig, mnist_fig_v])

    return


@app.cell
def _(mo):
    mo.md(
        """
    # Latent diffusion models
    One issue with diffusion models is that the distribution we might want to generate might be quite high dimensional, e.g., high-res images and videos. This means a large and expensive model is required, because the input and output are high-dimensional.

    A solution here recognizes that many of these high dimention objects are actually described by a much simpler underlying structure. E.g., a high-res video may be billions of pixel values, but actually easily describably in a shorter way, e.g., a sentence. The idea of Latent diffusion models is to learn an 'encoder/decoder' model that transforms our distribution into a smaller latent embedding that describes the underlying features of the data. Then we can learn to generate _the latent_ instead of the actual data, and decode it back up to the final distribution.

    Usually the encoder/decoder is learned as some kind of autoencoder model, where we train by putting data through the encoder to the latent, then decoding and comparing to the original. If the models are good, we should be able to reconstruct.

    ## Properties of the latent
    There are some desirable properties for the latent to have:
    - Denoising: We want a latent space that is not sensitive to noise when we decode it. If small changes in latents mean huge changes in the output image, then it's difficult to learn. Something that changes smoothly as the latent changes is nice.
    - Has a structure: When we learn the diffusion model, there does need to be _some_ structure. If the latent space was perfectly gaussian, then we actually couldn't learn a diffusion model
    - Class conditional structure

    Using a model like a VAE or VQVAE can give us some of these properties.
    """
    )
    return


@app.cell(hide_code=True)
def _(lib, mo):
    latent_encoder, celebA_model = lib.models.load_pretrained_celebA_model()
    celebA_resample = mo.ui.button(label="Resample")
    celebA_t = mo.ui.slider(0, 50, full_width=True)

    mo.md(f"{celebA_resample} \n {celebA_t}")
    return celebA_model, celebA_resample, celebA_t, latent_encoder


@app.cell(hide_code=True)
def _(celebA_model, celebA_resample, deepcopy, np, torch):
    celebA_resample.value
    _n_steps = 50
    cA_t_steps = np.linspace(0.0, 1.0, _n_steps + 1)
    cA_step_size = cA_t_steps[1] - cA_t_steps[0]
    cA_x_0 = torch.randn((1, 4, 8, 8)).float()
    _x_t = cA_x_0
    _x_t_seq = [deepcopy(_x_t)]
    _v_seq = [torch.zeros_like(_x_t)]
    for _t in torch.from_numpy(cA_t_steps).float():
        cA_t_tensor = _t.view(1, 1).float()
        with torch.no_grad():
            _cA_pred = celebA_model(_x_t, cA_t_tensor).detach()
        _x_t += cA_step_size * _cA_pred
        _x_t_seq.append(deepcopy(_x_t))
        _v_seq.append(deepcopy(_cA_pred))

    cA_seq = _x_t_seq
    cA_vs = _v_seq
    return cA_seq, cA_vs


@app.cell
def _(cA_seq, cA_vs, celebA_t, latent_encoder, mo, plt):
    cA_out_latent = cA_seq[celebA_t.value]
    cA_out = latent_encoder.forward_decode_quantize(cA_out_latent).detach()
    cA_out_v = cA_vs[celebA_t.value]
    _fig, _axes = plt.subplots()
    _axes.imshow(cA_out.reshape((3, 128, 128)).permute(1, 2, 0), cmap="grey")
    _fig_v, _axes_v = plt.subplots()
    # _axes_v.imshow(mnist_out_v.reshape((28, 28)), cmap="plasma")
    _axes.axis("off")
    _axes_v.axis("off")

    cA_fig = _fig
    cA_fig_v = _fig_v

    mo.hstack([cA_fig, cA_fig_v])

    return


if __name__ == "__main__":
    app.run()
