import time

import jax
import mujoco
import mujoco.viewer
from mujoco import mjx


XML = r"""
<mujoco>
  <asset>
    <texture builtin="gradient" height="100" rgb1="0.3 0.5 0.9" rgb2="0 0.3 0.8"
             type="skybox" width="100"/>
  </asset>
  <worldbody>
    <light directional="true" diffuse="0.8 0.8 0.8" pos="0 0 3" dir="0 0 -1"/>
    <light directional="true" diffuse="0.2 0.2 0.2" pos="-1 1 1" dir="1 -1 -1"/>
    <geom name="floor" type="plane" size="10 10 0.1" rgba="0 0.7 0.2 1"/>
    <body pos="0 0 2.0">
      <freejoint/>
      <geom size=".15" mass="1" type="sphere" rgba="0.9 0.1 0.1 1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
mjx_model = mjx.put_model(model)


@jax.vmap
def batched_step(vel):
    mjx_data = mjx.make_data(mjx_model)
    qvel = mjx_data.qvel.at[0].set(vel)
    mjx_data = mjx_data.replace(qvel=qvel)
    pos = mjx.step(mjx_model, mjx_data).qpos[0]
    return pos


vel = jax.numpy.arange(0.0, 1.0, 0.01)
pos = jax.jit(batched_step)(vel)
print(pos)


def run_simulation(vel, duration=1.0):
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    # Set initial velocity
    data.qvel[0] = vel

    # Create viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set camera
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20

        # Simulation loop
        step_size = model.opt.timestep
        num_steps = int(duration / step_size)

        for i in range(num_steps):
            # Step physics
            mujoco.mj_step(model, data)

            # Update viewer
            viewer.sync()
            time.sleep(0.01)  # Add a small delay to make visualization visible


# Try with a specific velocity
run_simulation(0.5)
