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

    return lib, mo, plt


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


if __name__ == "__main__":
    app.run()
