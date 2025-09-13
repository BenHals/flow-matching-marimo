from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from lib import domain
from lib.trajectory import Trajectory
from lib.vector_field import VectorField


def solve_for_trajectory_at_timesteps(
    init_location: domain.Location,
    vector_field: VectorField,
    timesteps: Sequence[float],
) -> Trajectory:
    t = 0.0
    x = init_location
    location_path = [x]
    output_timesteps = [t]
    full_timesteps = [*timesteps]
    if full_timesteps[0] == 0.0:
        _ = full_timesteps.pop(0)
    if full_timesteps[-1] != 1.0:
        full_timesteps.append(1.0)
    for t_next in full_timesteps[1:]:
        timestep_size = t_next - t
        velocity = vector_field.get_vector(x, t)
        x = x + timestep_size * velocity
        t = t_next
        location_path.append(x)
        output_timesteps.append(t)
    return Trajectory(np.stack(location_path), np.stack(output_timesteps))
