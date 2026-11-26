"""Go1 policy in plain MuJoCo: the observation and action pipeline, rebuilt outside mjlab.

Everything the policy needs is read from the ONNX metadata, so this file has no mjlab import.
"""
# [permanent robotics] Porting a policy = rebuilding its observation and action contract exactly.
import numpy as np
import mujoco
import onnx
import onnxruntime as ort

GRAVITY_W = np.array([0.0, 0.0, -1.0])

# mjlab sets these at runtime; the exported scene.xml carries none of them (it says Euler, pyramidal
# cone, impratio 1, timestep 0.002). Values from mjlab 1.6.1's MujocoCfg for the Go1 velocity task.
MJLAB_SIM = {
    "timestep": 0.005,
    "integrator": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
    "cone": mujoco.mjtCone.mjCONE_ELLIPTIC,
    "solver": mujoco.mjtSolver.mjSOL_NEWTON,
    "impratio": 10.0,
    "iterations": 10,
    "ls_iterations": 20,
    "ls_tolerance": 0.01,
    "tolerance": 1e-8,
    "ccd_iterations": 50,
}


def read_meta(onnx_path):
    meta = {p.key: p.value for p in onnx.load(onnx_path).metadata_props}
    floats = lambda k: np.array([float(x) for x in meta[k].split(",")])
    return {
        "joint_names": meta["joint_names"].split(","),
        "default_joint_pos": floats("default_joint_pos"),
        "action_scale": floats("action_scale"),
        "kp": floats("joint_stiffness"),
        "kd": floats("joint_damping"),
        "obs_names": meta["observation_names"].split(","),
    }


class Go1Port:
    """Plain-MuJoCo twin of Mjlab-Velocity-Flat-Unitree-Go1 (actor side only)."""

    def __init__(self, scene_xml, onnx_path, timestep=0.005, decimation=4, prefix="robot/", sim=None):
        self.m = mujoco.MjModel.from_xml_path(scene_xml)
        for k, v in {**MJLAB_SIM, **(sim or {}), "timestep": timestep}.items():
            setattr(self.m.opt, k, v)
        self.d = mujoco.MjData(self.m)
        self.decimation = decimation
        self.meta = read_meta(onnx_path)
        names = [prefix + j for j in self.meta["joint_names"]]
        # Policy order (FR_hip, FR_thigh, FR_calf, ...) is NOT the actuator order in the XML.
        self.qadr = np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in names])
        self.vadr = np.array([self.m.jnt_dofadr[self.m.joint(n).id] for n in names])
        self.act_id = np.array([self.m.actuator(n).id for n in names])
        s = lambda n: slice(self.m.sensor_adr[self.m.sensor(prefix + n).id],
                            self.m.sensor_adr[self.m.sensor(prefix + n).id] + 3)
        self.s_linvel, self.s_angvel = s("imu_lin_vel"), s("imu_ang_vel")
        self.sess = ort.InferenceSession(onnx_path)
        self.last_action = np.zeros(12)
        self.command = np.zeros(3)  # vx, vy, wz in the body frame

    def reset(self, keyframe=0):
        mujoco.mj_resetDataKeyframe(self.m, self.d, keyframe)
        self.last_action[:] = 0
        mujoco.mj_forward(self.m, self.d)

    def obs(self):
        d = self.d
        quat = d.qpos[3:7]
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, quat)
        proj_g = mat.reshape(3, 3).T @ GRAVITY_W
        return np.concatenate([
            d.sensordata[self.s_linvel],
            d.sensordata[self.s_angvel],
            proj_g,
            d.qpos[self.qadr] - self.meta["default_joint_pos"],
            d.qvel[self.vadr],
            self.last_action,
            self.command,
        ]).astype(np.float32)

    def policy(self, obs):
        return self.sess.run(None, {"obs": obs[None]})[0][0]

    def apply(self, action):
        target = self.meta["default_joint_pos"] + self.meta["action_scale"] * action
        self.d.ctrl[self.act_id] = target
        self.last_action = action.astype(np.float64)

    def step(self):
        """One policy step = decimation physics steps."""
        a = self.policy(self.obs())
        self.apply(a)
        for _ in range(self.decimation):
            mujoco.mj_step(self.m, self.d)
        return a
