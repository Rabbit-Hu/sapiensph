import warp as wp


@wp.kernel
def update_particle_render_meshes(
    positions: wp.array(dtype=wp.vec3),
    positions_begin: int,
    particle_render_V: wp.array(dtype=wp.vec3),
    n_particle_render_V: int,
    particle_diameter: float,
    render_V: wp.array(dtype=float, ndim=2),
):
    i = wp.tid() + positions_begin
    for j in range(n_particle_render_V):
        v = particle_render_V[j] * particle_diameter + positions[i]
        render_V[i * n_particle_render_V + j, 0] = v[0]
        render_V[i * n_particle_render_V + j, 1] = v[1]
        render_V[i * n_particle_render_V + j, 2] = v[2]
