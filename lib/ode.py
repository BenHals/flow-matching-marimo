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
    for t_next in timesteps:
        if t_next == t:
            continue
        step_size = t_next - t
        direction = vector_field.get_vector(x, t)
        x_next = x + step_size * direction
        x = x_next
        t = t_next
        location_path.append(x)
        output_timesteps.append(t)
    return Trajectory(np.stack(location_path), np.stack(output_timesteps))
