import warp as wp
from .kernel_funcs import sph_W, sph_dWdr


@wp.kernel
def kinematic_update(
    positions_prev: wp.array(dtype=wp.vec3),
    velocities_prev: wp.array(dtype=wp.vec3),
    pressure_accs: wp.array(dtype=wp.vec3),
    gravity: wp.vec3,
    dt: float,
    boundary_min: wp.vec3,
    boundary_max: wp.vec3,
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    x_i_prev = positions_prev[i]
    v_i_prev = velocities_prev[i]

    v_i_new = v_i_prev + dt * (pressure_accs[i] + gravity)
    x_i_new = x_i_prev + dt * v_i_new

    x_i_new = wp.min(wp.max(x_i_new, boundary_min), boundary_max)
    v_i_new = (x_i_new - x_i_prev) / dt

    velocities[i] = v_i_new
    positions[i] = x_i_new


@wp.kernel
def compute_densities(
    grid: wp.uint64,
    positions: wp.array(dtype=wp.vec3),
    kernel_radius: float,
    rest_density: float,
    density_err_min: float,
    density_err_max: float,
    m: float,
    delta: float,
    densities: wp.array(dtype=wp.float32),
    density_errors: wp.array(dtype=wp.float32),
    pressures: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    x_i = positions[i]

    query = wp.hash_grid_query(grid, x_i, kernel_radius * 2.0)
    j = int(0)
    rho_i = float(0.0)

    while wp.hash_grid_query_next(query, j):
        W = sph_W(x_i - positions[j], kernel_radius)
        rho_i += m * W

    rho_err_i = rho_i - rest_density
    rho_err_i = wp.clamp(
        rho_err_i, density_err_min * rest_density, density_err_max * rest_density
    )
    delta_p_i = delta * rho_err_i
    p_i = pressures[i] + delta_p_i

    densities[i] = rho_i
    density_errors[i] = rho_err_i
    pressures[i] = p_i


@wp.kernel
def compute_pressure_accs(
    grid: wp.uint64,
    positions: wp.array(dtype=wp.vec3),
    densities: wp.array(dtype=wp.float32),
    pressures: wp.array(dtype=wp.float32),
    kernel_radius: float,
    m: float,
    pressure_accs: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    x_i = positions[i]
    p_i = pressures[i]
    rho_i = densities[i]

    query = wp.hash_grid_query(grid, x_i, kernel_radius * 2.0)
    j = int(0)
    acc_i = wp.vec3(0.0, 0.0, 0.0)

    while wp.hash_grid_query_next(query, j):
        if i != j:
            w = sph_dWdr(x_i - positions[j], kernel_radius)
            p_j = pressures[j]
            rho_j = densities[j]
            acc_i += -m * (p_i / (rho_i**2.0) + p_j / (rho_j**2.0)) * w

    pressure_accs[i] = acc_i
