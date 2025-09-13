import matplotlib.pyplot as plt

from lib import data
from lib.alpha_beta import AlphaBasic, BetaBasic
from lib.ode import solve_for_trajectory_at_timesteps
from lib.vector_field import ConvergingVectorField, GaussianConditionalVectorField


def converging():
    print("Hello from flow-matching!")
    init_dataset = data.make_spiral(3, 100)
    final_dataset = data.make_gaussian(100)

    sample = data.sample_dataset(final_dataset)
    vector_field = ConvergingVectorField(0.5)

    trajectory = solve_for_trajectory_at_timesteps(
        sample, vector_field, [i / 10.0 for i in range(10)]
    )

    plt.scatter(trajectory.timesteps, trajectory.locations[:, 0])
    plt.show()


def main():
    init_dataset = data.make_spiral(3, 100)
    final_dataset = data.make_gaussian(100)

    for j in range(100):
        z = data.sample_dataset(init_dataset)
        start = data.sample_dataset(final_dataset)

        vector_field = GaussianConditionalVectorField(
            alpha=AlphaBasic(), beta=BetaBasic(), z=z
        )

        n_steps = 1000
        trajectory = solve_for_trajectory_at_timesteps(
            start, vector_field, [i / n_steps for i in range(n_steps)]
        )
        print(z, trajectory.locations[-1])

        # for i in range(0, len(trajectory.timesteps), 100):
        #     plt.scatter(
        #         [trajectory.locations[i, 0]],
        #         [trajectory.locations[i, 1]],
        #         color="blue",
        #         alpha=i / (2 * n_steps),
        #     )
        plt.plot(
            [trajectory.locations[0, 0], trajectory.locations[-1, 0]],
            [trajectory.locations[0, 1], trajectory.locations[-1, 1]],
            alpha=0.25,
        )
        plt.scatter(
            [trajectory.locations[-1, 0]],
            [trajectory.locations[-1, 1]],
            color="red",
        )
    plt.show()


if __name__ == "__main__":
    main()
