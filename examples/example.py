import warp as wp
import numpy as np
import sapien
import trimesh
from PIL import Image
import os

import sapiensph
from sapiensph.sph_system import SPHConfig, SPHSystem
from sapiensph.sph_component import SPHComponent


# sapien.render.set_viewer_shader_dir("rt")
# sapien.render.set_camera_shader_dir("rt")
# sapien.render.set_ray_tracing_samples_per_pixel(8)
# sapien.render.set_ray_tracing_denoiser("oidn")


def init_camera(scene: sapien.Scene):
    cam_entity = sapien.Entity()
    cam = sapien.render.RenderCameraComponent(1024, 1024)
    cam.set_near(1e-3)
    cam.set_far(1000)
    cam_entity.add_component(cam)
    cam_entity.name = "camera"
    cam_entity.set_pose(
        sapien.Pose(
            # [-0.362142, 0.0550001, 0.172571], [0.995491, -5.96046e-08, 0.0948573, 0]
            [-0.275494, 0.0519914, 0.173081], [0.996565, 0.0031984, -0.0466232, 0.068371]
        )
    )
    scene.add_entity(cam_entity)
    return cam


def main():
    scene = sapien.Scene()
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([1, 0, -1], [1, 1, 1], True)
    scene.add_ground(0.0)

    cam = init_camera(scene)

    particle_mesh = trimesh.load_mesh(os.path.join("assets", "sphere.obj"))

    config = SPHConfig()
    system = SPHSystem(
        particle_mesh.vertices, particle_mesh.faces, config=config, device="cuda:0"
    )
    scene.add_system(system)

    sph_comp = SPHComponent()
    sph_comp.set_box(np.array([0.0, 0.0, 0.1]), np.array([0.1, 0.1, 0.4]))
    sph_entity = sapien.Entity()
    sph_entity.add_component(sph_comp)
    scene.add_entity(sph_entity)

    # system._build_hash_grid()
    # system._compute_densities()
    # print(system.densities.numpy()[:system.n_particles])

    viewer = sapien.utils.Viewer()
    viewer.set_scene(scene)
    viewer.set_camera_pose(cam.get_entity_pose())
    viewer.window.set_camera_parameters(1e-3, 1000, np.pi / 2)
    viewer.paused = True

    system.update_render()
    scene.update_render()
    viewer.render()
    
    render_every = 20
    save_render = False
    save_dir = "output/example"
    os.makedirs(save_dir, exist_ok=True)
    
    # print(f"System particles: {system.n_particles}")
    
    for time_step in range(4000):
        with wp.ScopedTimer("time step"):
            system.step()

        if time_step % render_every == 0:
            system.update_render()
            scene.update_render()
            viewer.render()
            
            if save_render:
                cam.take_picture()
                rgba = cam.get_picture("Color")
                rgba = np.clip(rgba, 0, 1)[:, :, :3]
                rgba = Image.fromarray((rgba * 255).astype(np.uint8))
                rgba.save(os.path.join(save_dir, f"step_{(time_step + 1) // render_every:04d}.png"))


# ffmpeg -framerate 50 -i output/example/step_%04d.png -c:v libx264 -crf 0 output/example.mp4

# ffmpeg -framerate 50 -i output/example/step_%04d.png -filter_complex "[0:v] palettegen" output/example_palette.png
# ffmpeg -framerate 50 -i output/example/step_%04d.png -i output/example_palette.png -filter_complex "fps=25,scale=512:-1 [new];[new][1:v] paletteuse" output/example.gif
# gifsicle -O3 output/example.gif --lossy=100 -o output/example_reduced.gif

if __name__ == "__main__":
    main()
