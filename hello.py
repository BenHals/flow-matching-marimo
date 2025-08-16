import matplotlib.pyplot as plt

from lib import data
from lib.ode import solve_for_trajectory_at_timesteps
from lib.vector_field import ConvergingVectorField


def main():
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


if __name__ == "__main__":
    main()
