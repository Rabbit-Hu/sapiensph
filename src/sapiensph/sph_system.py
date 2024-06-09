import warp as wp
import numpy as np
import weakref
import sapien

from .sph_component import *
from .utils.array import wp_slice
from .kernels.step_kernels import *


class SPHConfig:
    def __init__(self):
        ################ Memory config ################
        self.max_particles = 1 << 20

        ################ Solver config ################
        self.time_step = 1e-3
        self.n_pci_iters = 10
        self.hash_grid_dim = 128
        self.kernel_size = 1e-2
        self.particle_diameter = self.kernel_size
        self.density_err_min = -0.01
        self.density_err_max = 1e9

        ################ Physics config ################
        self.gravity = np.array([0, 0, -9.8], dtype=np.float32)
        self.rest_density = 1e3
        # self.delta_const = 2.353827575843501
        # delta (in the PCISPH paper) = delta_const * h**8 / beta
        self.max_velocity = (
            1.0  # estimated max velocity for collision detection, TODO: add clamping
        )
        self.damping = 0.0  # damping energy: 0.5 * damping * v^2, TODO: add damping
        self.boundary_min = np.array([-0.01, -0.01, 0.0], dtype=np.float32)
        self.boundary_max = np.array([0.11, 0.25, 1e9], dtype=np.float32)
        

class SPHSystem(sapien.System):
    def __init__(
        self,
        particle_render_V,
        particle_render_F,
        config: SPHConfig = None,
        device: str = "cuda:0",
    ):
        super().__init__()
        wp.init()
        self.name = "sph"

        assert particle_render_V.shape[1] == 3
        assert particle_render_F.shape[1] == 3
        with wp.ScopedDevice(device):
            self.particle_render_V = wp.array(particle_render_V, dtype=wp.vec3)
            self.particle_render_F = wp.array(particle_render_F, dtype=wp.uint32)
            self.particle_render_V_np = particle_render_V
            self.particle_render_F_np = particle_render_F

        self.config = config or SPHConfig()
        self.device = device

        self.components = []

        self._init_arrays()
        self._init_counters()
        self._init_constants()

    def get_name(self):
        return self.name

    def _init_arrays(self):
        config = self.config
        MP = config.max_particles

        with wp.ScopedDevice(self.device):
            self.positions = wp.zeros(MP, dtype=wp.vec3)
            self.positions_prev = wp.zeros(MP, dtype=wp.vec3)
            self.velocities = wp.zeros(MP, dtype=wp.vec3)
            self.velocities_prev = wp.zeros(MP, dtype=wp.vec3)
            self.pressure_accs = wp.zeros(MP, dtype=wp.vec3)
            self.densities = wp.zeros(MP, dtype=wp.float32)
            self.density_errors = wp.zeros(MP, dtype=wp.float32)
            self.pressures = wp.zeros(MP, dtype=wp.float32)

            hgdim = config.hash_grid_dim
            self.hash_grid = wp.HashGrid(hgdim, hgdim, hgdim)

    def _init_counters(self):
        self.n_particles = 0

    def _init_constants(self):
        def W(r, h):
            q = np.linalg.norm(r) / h
            if q < 1.0:
                return 1.0 / (np.pi * h ** 3.0) * (1.0 - 1.5 * q ** 2.0 + 0.75 * q ** 3.0)
            elif q < 2.0:
                return 1.0 / (np.pi * h ** 3.0) * 0.25 * (2.0 - q) ** 3.0
            else:
                return 0.0
            
        def dWdr(r, h):
            q = np.linalg.norm(r) / h
            if q < 1e-9 or q >= 2.0:
                return np.array([0.0, 0.0, 0.0])
            elif q < 1.0:
                return 1.0 / (np.pi * h ** 4.0) * (-3.0 * q + 2.25 * q ** 2.0) * r / np.linalg.norm(r)
            else:
                return 1.0 / (np.pi * h ** 4.0) * -0.75 * (2.0 - q) ** 2.0 * r / np.linalg.norm(r)
            
        h = self.config.kernel_size
        sum_W = 0.0
        sum_grad = np.array([0.0, 0.0, 0.0])
        sum_grad_dot_grad = 0.0
        
        for qx in np.arange(-1, 2, 0.5):
            for qy in np.arange(-1, 2, 0.5):
                for qz in np.arange(-1, 2, 0.5):
                    q = np.array([qx, qy, qz], dtype=np.float32)
                    r = q * h
                    sum_W += W(r, h)
                    grad = dWdr(r, h)
                    sum_grad += grad
                    sum_grad_dot_grad += np.dot(grad, grad)
                    
        rho_0 = self.config.rest_density
        self.m = rho_0 * self.config.particle_diameter ** 3.0
        self.beta = 2.0 * (self.config.time_step * self.m / rho_0) ** 2.0
        self.delta = 1.0 / (self.beta * (sum_grad_dot_grad + np.dot(sum_grad, sum_grad)))
        # print(f"sum_W = {sum_W}, m = {self.m}, beta = {self.beta}, delta = {self.delta}")

    def _add_particles(self, positions: np.ndarray):
        p_begin = self.n_particles
        p_end = p_begin + len(positions)
        assert (
            p_end <= self.config.max_particles
        ), f"Too many particles ({p_end} > {self.config.max_particles})!"
        self.n_particles = p_end

        with wp.ScopedDevice(self.device):
            wp_slice(self.positions, p_begin, p_end).assign(positions)
            wp_slice(self.velocities, p_begin, p_end).zero_()
            wp_slice(self.pressure_accs, p_begin, p_end).zero_()
            wp_slice(self.densities, p_begin, p_end).zero_()
            wp_slice(self.density_errors, p_begin, p_end).zero_()
            wp_slice(self.pressures, p_begin, p_end).zero_()

        return p_begin, p_end

    def _add_particles_by_box(self, bmin: np.ndarray, bmax: np.ndarray):
        bmin = np.array(bmin, dtype=np.float32)
        bmax = np.array(bmax, dtype=np.float32)
        assert bmin.shape == (3,)
        assert bmax.shape == (3,)

        d = self.config.particle_diameter 
        r = d * 0.5
        xs = np.arange(bmin[0] + r, bmax[0] - r + 1e-9, d) + r
        ys = np.arange(bmin[1] + r, bmax[1] - r + 1e-9, d) + r
        zs = np.arange(bmin[2] + r, bmax[2] - r + 1e-9, d) + r
        positions = np.array(np.meshgrid(xs, ys, zs)).reshape(3, -1).T
        return self._add_particles(positions)

    def _register_component_get_id(self, comp: SPHComponent):
        component_id = len(self.components)
        self.components.append(weakref.proxy(comp))

        return component_id

    def register_sph_component(self, comp: SPHComponent):
        comp.id_in_sys = self._register_component_get_id(comp)
        if comp.positions is not None:
            comp.particles_ptr_in_sys = self._add_particles(comp.positions)
        elif comp.bmin is not None and comp.bmax is not None:
            comp.particles_ptr_in_sys = self._add_particles_by_box(comp.bmin, comp.bmax)
        else:
            raise ValueError("SPHComponent must have particles or box")

    def update_render(self):
        for comp in self.components:
            if isinstance(comp, SPHComponent):
                comp.update_render()

    def _build_hash_grid(self):
        conf = self.config
        with wp.ScopedDevice(self.device):
            self.hash_grid.build(
                wp_slice(self.positions, 0, self.n_particles),
                2.0 * (conf.kernel_size + conf.max_velocity * conf.time_step),
            )

    def _kinematic_update(self):
        with wp.ScopedDevice(self.device):
            wp.launch(
                kernel=kinematic_update,
                dim=self.n_particles,
                inputs=[
                    self.positions_prev,
                    self.velocities_prev,
                    self.pressure_accs,
                    self.config.gravity,
                    self.config.time_step,
                    self.config.boundary_min,
                    self.config.boundary_max,
                ],
                outputs=[self.positions, self.velocities],
            )

    def _compute_densities(self):
        with wp.ScopedDevice(self.device):
            wp.launch(
                kernel=compute_densities,
                dim=self.n_particles,
                inputs=[
                    self.hash_grid.id,
                    self.positions,
                    self.config.kernel_size,
                    self.config.rest_density,
                    self.config.density_err_min,
                    self.config.density_err_max,
                    self.m,
                    self.delta,
                ],
                outputs=[
                    self.densities,
                    self.density_errors,
                    self.pressures,
                ],
            )

        # print(f"densities: {self.densities.numpy()[:self.n_particles]}")
        # print(f"density_errors: {self.density_errors.numpy()[:self.n_particles]}")
        # print(f"pressures: {self.pressures.numpy()[:self.n_particles]}")

    def _compute_pressure_accs(self):
        with wp.ScopedDevice(self.device):
            wp.launch(
                kernel=compute_pressure_accs,
                dim=self.n_particles,
                inputs=[
                    self.hash_grid.id,
                    self.positions,
                    self.densities,
                    self.pressures,
                    self.config.kernel_size,
                    self.m,
                ],
                outputs=[self.pressure_accs],
            )

        # print(f"pressure_accs: {self.pressure_accs.numpy()[:self.n_particles]}")

    def step(self):
        NP = self.n_particles

        wp.copy(self.positions_prev, self.positions, count=NP)
        wp.copy(self.velocities_prev, self.velocities, count=NP)

        self._build_hash_grid()

        wp_slice(self.pressures, 0, NP).zero_()
        wp_slice(self.pressure_accs, 0, NP).zero_()

        for pci_iter in range(self.config.n_pci_iters):
            with wp.ScopedDevice(self.device):
                self._kinematic_update()
                self._compute_densities()
                self._compute_pressure_accs()

        self._kinematic_update()
