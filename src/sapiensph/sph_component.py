import warp as wp
import numpy as np
import sapien
import typing
from typing import List, Union
from sapien.render import RenderCudaMeshComponent
from .utils.array import wp_slice
from .kernels.render_kernels import *

if typing.TYPE_CHECKING:
    from .sph_system import SPHSystem


class SPHComponent(sapien.Component):
    def __init__(self):
        super().__init__()

        self.id_in_sys = None
        self.particles_ptr_in_sys = None

        self.positions = None
        self.bmin = None
        self.bmax = None

    def set_particles(self, positions: np.ndarray):
        self.positions = positions

    def set_box(self, bmin: np.ndarray, bmax: np.ndarray):
        self.bmin = bmin
        self.bmax = bmax

    def on_add_to_scene(self, scene: sapien.Scene):
        system: SPHSystem = scene.get_system("sph")
        system.register_sph_component(self)
        self.create_render_component(
            system.particle_render_V_np, system.particle_render_F_np
        )

    def create_render_component(self, V, F, color=[0.3, 0.5, 0.7, 1.0]):
        assert self.particles_ptr_in_sys is not None
        n_particles = self.particles_ptr_in_sys[1] - self.particles_ptr_in_sys[0]
        n_render_V = len(V) * n_particles
        n_render_F = len(F) * n_particles
        render_F = F + np.arange(n_particles)[:, None, None] * len(V)

        self.render_comp = RenderCudaMeshComponent(n_render_V, n_render_F)
        self.render_comp.set_vertex_count(n_render_V)
        self.render_comp.set_triangle_count(n_render_F)
        self.render_comp.set_triangles(render_F.reshape(-1, 3))
        self.render_comp.set_material(sapien.render.RenderMaterial(base_color=color))

        self.entity.add_component(self.render_comp)

    def update_render(self):
        assert self.render_comp is not None

        s: SPHSystem = self.entity.scene.get_system("sph")
        n_particles = self.particles_ptr_in_sys[1] - self.particles_ptr_in_sys[0]
        with wp.ScopedDevice(s.device):
            interface = self.render_comp.cuda_vertices.__cuda_array_interface__
            dst = wp.array(
                ptr=interface["data"][0],
                dtype=wp.float32,
                shape=interface["shape"],
                strides=interface["strides"],
                owner=False,
            )
            wp.launch(
                kernel=update_particle_render_meshes,
                dim=n_particles,
                inputs=[
                    s.positions,
                    self.particles_ptr_in_sys[0],
                    s.particle_render_V,
                    len(s.particle_render_V),
                    s.config.particle_diameter / 2.0,
                ],
                outputs=[dst],
            )
            self.render_comp.notify_vertex_updated()
