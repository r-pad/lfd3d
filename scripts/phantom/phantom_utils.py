import mink
import mujoco
import numpy as np
import open3d as o3d
import torch
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R
from torch import pi
from tqdm import tqdm

ALOHA_GRIPPER_MIN, ALOHA_GRIPPER_MAX = 0, 0.041
HUMAN_GRIPPER_IDX = np.array([343, 763, 60])
ALOHA_REST_QPOS = np.array(
    [0, -1.73, 1.49, 0, 0, 0, 0, 0, 0, -1.73, 1.49, 0, 0, 0, 0, 0]
)


def inverse_kinematics(model, configuration, eef_pos, gripper_rot):
    pose_matrix = np.eye(4)
    pose_matrix[:3, 3] = eef_pos
    pose_matrix[:3, :3] = gripper_rot
    ee_pose_se3 = mink.lie.se3.SE3.from_matrix(pose_matrix)

    n_iter = 500
    dt = 0.01
    thresh = 1e-4

    ee_task = mink.FrameTask(
        frame_name="right/gripper",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
    )
    ee_task.set_target(ee_pose_se3)
    # A posture task to encourage the IK to solve for more "natural" poses
    # Left arm is at rest whereas the right arm is upright in an L shape
    TARGET_POSE = ALOHA_REST_QPOS.copy()
    TARGET_POSE[8:] = 0
    posture_task = mink.PostureTask(model, cost=0.05)
    posture_task.set_target(TARGET_POSE)

    ke_task = mink.KineticEnergyRegularizationTask(cost=1e-3)
    ke_task.set_dt(dt)

    for i in range(n_iter):
        vel = mink.solve_ik(
            configuration, [ee_task, ke_task, posture_task], dt=dt, solver="daqp"
        )
        configuration.integrate_inplace(vel, dt)

        err = ee_task.compute_error(configuration)
        # print(i, np.linalg.norm(err))
        if np.linalg.norm(err) < thresh:
            break

    Q = configuration.q
    print(f"iter: {i}, err: {np.linalg.norm(err)}")
    return Q


def convert_sim_joints(Q, gripper_artic):
    """
    Map the joints of the sim aloha used for IK
    back to the real aloha to train a policy.
    Different formats, units, orientations ...."""
    vec = torch.tensor(
        [
            Q[0],
            Q[1],
            Q[1],
            Q[2],
            Q[2],
            Q[3],
            Q[4],
            Q[5],
            0,
            Q[8],
            Q[9],
            Q[9],
            Q[10],
            Q[10],
            Q[11],
            Q[12],
            Q[13],
            0,
        ]
    )
    # Minor but important distinction here.
    # In lerobot, the gripper_artic comes from the real robot
    # doesn't need to be mapped sim2real and is thus written into `vec`
    # *after*. Here, gripper_artic comes from hand pose retargeted to sim gripper
    # and thus needs to be written into `vec` *before* mapping sim2real.
    vec[-1] = gripper_artic
    vec = torch.rad2deg(map_sim2real(vec))
    return vec


def map_sim2real(vec):
    """
    inverse of map_real2sim from r-pad/lerobot
    sim = real*sign + offset
    real = (sim - offset)*sign
    """
    sign = torch.tensor([-1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1])
    offset = torch.tensor(
        [
            pi / 2,
            0,
            0,
            -pi / 2,
            -pi / 2,
            0,
            0,
            0,
            0,
            pi / 2,
            0,
            0,
            -pi / 2,
            -pi / 2,
            0,
            0,
            0,
            0,
        ]
    )
    vec = (vec - offset) * sign

    # Inverted from real2sim
    real_shoulder_min, real_shoulder_max = 0.23, 3.59
    sim_shoulder_min, sim_shoulder_max = -1.26, 1.85

    vec[1] = (vec[1] - sim_shoulder_min) * (
        (real_shoulder_max - real_shoulder_min) / (sim_shoulder_max - sim_shoulder_min)
    ) + real_shoulder_min
    vec[2] = (vec[2] - sim_shoulder_min) * (
        (real_shoulder_max - real_shoulder_min) / (sim_shoulder_max - sim_shoulder_min)
    ) + real_shoulder_min
    vec[10] = (vec[10] - sim_shoulder_min) * (
        (real_shoulder_max - real_shoulder_min) / (sim_shoulder_max - sim_shoulder_min)
    ) + real_shoulder_min
    vec[11] = (vec[11] - sim_shoulder_min) * (
        (real_shoulder_max - real_shoulder_min) / (sim_shoulder_max - sim_shoulder_min)
    ) + real_shoulder_min

    real_gripper_min, real_gripper_max = -1.7262, 0.11
    sim_gripper_min, sim_gripper_max = -0.04, 0
    vec[8] = (vec[8] - sim_gripper_min) * (
        (real_gripper_max - real_gripper_min) / (sim_gripper_max - sim_gripper_min)
    ) + real_gripper_min
    vec[17] = (vec[17] - sim_gripper_min) * (
        (real_gripper_max - real_gripper_min) / (sim_gripper_max - sim_gripper_min)
    ) + real_gripper_min
    return vec


def render_rightArm_images(renderer, data, camera="teleoperator_pov", use_seg=False):
    """
    Render RGB, depth, and segmentation images with right arm masking from MuJoCo simulation.
    Args:
        renderer (mujoco.Renderer): MuJoCo renderer instance configured for the scene
        data (mujoco.MjData): MuJoCo data object containing current simulation state
        camera (str, optional): Name of the camera to render from.

    Returns:
        tuple: A tuple containing:
            - rgb (np.ndarray): Masked RGB image of shape (H, W, 3), dtype uint8.
                               Right arm pixels retain original colors, background pixels are black.
            - depth (np.ndarray): Masked depth image of shape (H, W), dtype float32.
                                 Right arm pixels contain depth values, background pixels are zero.
            - seg (np.ndarray): Binary segmentation mask of shape (H, W), dtype bool.
                               True for right arm pixels, False for background.
    """
    renderer.update_scene(data, camera=camera)
    rgb = renderer.render()

    # Depth rendering
    renderer.enable_depth_rendering()
    depth = renderer.render()
    renderer.disable_depth_rendering()

    # Segmentation rendering
    renderer.enable_segmentation_rendering()
    seg = renderer.render()
    renderer.disable_segmentation_rendering()

    seg = seg[:, :, 0]  # channel 1 is foreground/background
    # NOTE: Classes for the right arm excluding the camera mount. Handpicked
    target_classes = set(range(65, 91)) - {81, 82, 83}

    seg = np.isin(seg, list(target_classes)).astype(bool)
    if use_seg:
        rgb[~seg] = 0
        depth[~seg] = 0

    return rgb, depth, seg


def render_with_ik(
    model, mink_config, renderer, data, eef_pos, eef_rot, gripper_artic, n=10
):
    render_images = []
    render_seg = []
    render_depth = []
    actual_eef_pos = []
    actual_eef_rot = []
    actual_eef_artic = []
    joint_state = []

    site = "right/gripper"
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)

    # Sync mink configuration and mjdata
    qpos = data.qpos
    mink_config.update(qpos)

    n_poses = eef_pos.shape[0]
    for i in tqdm(range(n_poses)):
        Q = inverse_kinematics(model, mink_config, eef_pos[i], eef_rot[i])

        # Keep them synced ....
        data.qpos = Q
        data.qpos[8 + 6 : 8 + 8] = gripper_artic[i]
        mujoco.mj_forward(model, data)
        mink_config.update(Q)

        # Get the pose (4x4 matrix) of the site in the world frame
        actual_eef_pos.append(data.site_xpos[site_id].copy())
        actual_eef_rot.append(data.site_xmat[site_id].reshape(3, 3).copy())
        actual_eef_artic.append(gripper_artic[i])

        # Store joint angles (after mapping back to real robot)
        joint_angles = convert_sim_joints(data.qpos, gripper_artic[i])
        joint_state.append(joint_angles)

        if i % n == 0:
            rgb, depth, seg = render_rightArm_images(renderer, data)
            render_images.append(rgb)
            render_seg.append(seg)
            render_depth.append(depth)

    return (
        np.array(render_images),
        np.array(render_seg),
        np.array(render_depth),
        np.array(actual_eef_pos),
        np.array(actual_eef_rot),
        np.array(actual_eef_artic),
        np.array(joint_state),
    )


def gripper_points_to_rotation(gripper_center, palm_point, finger_point):
    # Always use palm->gripper as primary axis (more stable)
    forward = gripper_center - palm_point
    x_axis = forward / np.linalg.norm(forward, axis=1, keepdims=True)

    # Use finger relative to the forward direction for secondary axis
    finger_vec = gripper_center - finger_point

    # Project finger vector onto plane perpendicular to forward
    finger_projected = (
        finger_vec - np.sum(finger_vec * x_axis, axis=1, keepdims=True) * x_axis
    )
    y_axis = finger_projected / np.linalg.norm(finger_projected, axis=1, keepdims=True)

    # Z completes the frame
    z_axis = np.cross(x_axis, y_axis)

    return np.stack([x_axis, y_axis, z_axis], axis=-1)


def setup_camera(model, cam_id, cam_to_world, width, height, K):
    world_to_cam = np.linalg.inv(cam_to_world)
    model.cam_pos[cam_id] = cam_to_world[:3, 3]
    R_flip = np.diag([1, -1, -1])
    R_cam = R.from_matrix(cam_to_world[:3, :3] @ R_flip)
    cam_quat = R_cam.as_quat()  # [x, y, z, w]
    cam_quat = cam_quat[[3, 0, 1, 2]]  # Reorder to [w, x, y, z] for MuJoCo
    model.cam_quat[cam_id] = cam_quat
    fovy = np.degrees(2 * np.arctan((height / 2) / K[1, 1]))
    model.cam_fovy[cam_id] = fovy


def retarget_human_pose(hand_pose):
    CLOSE_THRESHOLD = 0.5
    gripper_points = hand_pose[:, HUMAN_GRIPPER_IDX]
    eef_pos = (gripper_points[:, 0, :] + gripper_points[:, 1, :]) / 2
    eef_articulation = np.linalg.norm(
        gripper_points[:, 0, :] - gripper_points[:, 1, :], axis=1
    )
    eef_articulation = (eef_articulation - eef_articulation.min()) / (
        eef_articulation.max() - eef_articulation.min()
    )
    # Force close gripper beyond threshold
    eef_articulation[eef_articulation > CLOSE_THRESHOLD] = 1
    eef_articulation = (
        eef_articulation * (ALOHA_GRIPPER_MAX - ALOHA_GRIPPER_MIN)
    ) + ALOHA_GRIPPER_MIN
    # eef pose, base, right finger
    gripper_rot = gripper_points_to_rotation(
        eef_pos, gripper_points[:, 2, :], gripper_points[:, 0, :]
    )
    return eef_pos, gripper_rot, eef_articulation


def visualize_eef(eef_pos, eef_rot, fname):
    combined_mesh = o3d.geometry.TriangleMesh()
    # Subsample frames
    indices = np.arange(0, eef_pos.shape[0])

    for i in indices:
        # Create coordinate frame
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)

        # Apply transformation
        T = np.eye(4)
        T[:3, :3] = eef_rot[i]
        T[:3, 3] = eef_pos[i]
        frame.transform(T)

        # Add to combined mesh
        combined_mesh += frame

    o3d.io.write_triangle_mesh(fname, combined_mesh)


def extract_from_robot_demo(
    model, data, robot_demo, world_to_cam=None, renderer=None, n=10
):
    """
    if world to cam is provided, transform to camera frame and return
    """
    # Get the pose of the site.
    site = "right/gripper"
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
    data.qpos[0] = -np.pi  # move the left arm out of the way for rendering
    render_images = []

    joint_angles = robot_demo["raw/follower_right/joint_states/pos"][:]
    # Compute forward kinematics for each timestep
    eef_pos, gripper_rot, eef_articulation = [], [], []
    for i in tqdm(range(joint_angles.shape[0])):
        data.qpos[8 : 8 + 6] = joint_angles[i, :6]
        data.qpos[8 + 6 : 8 + 8] = joint_angles[i, [7, 7]]
        # Compute the forward kinematics.
        mujoco.mj_forward(model, data)

        # Get the pose (4x4 matrix) of the site in the world frame
        site_pos = data.site_xpos[site_id].copy()
        site_mat = data.site_xmat[site_id].reshape(3, 3).copy()
        articulation = joint_angles[i, 7].copy()

        if renderer and i % n == 0:
            rgb, depth, seg = render_rightArm_images(renderer, data)
            render_images.append(rgb)

        eef_pos.append(site_pos)
        gripper_rot.append(site_mat)
        eef_articulation.append(articulation)
    eef_pos = np.array(eef_pos)
    gripper_rot = np.array(gripper_rot)
    eef_articulation = np.array(eef_articulation)

    if world_to_cam is not None:
        world_eef_pos_hom = np.concatenate(
            [eef_pos, np.ones((eef_pos.shape[0], 1))], axis=1
        )
        world_to_cam_rot = world_to_cam[:3, :3]

        cam_eef_pos_hom = (world_to_cam @ world_eef_pos_hom.T).T
        eef_pos = cam_eef_pos_hom[:, :3]

        cam_gripper_rot = world_to_cam_rot @ gripper_rot
        gripper_rot = cam_gripper_rot
    return eef_pos, gripper_rot, eef_articulation, render_images


def smooth_and_interpolate_pose(eef_pos, eef_rot, artic, N):
    """
    Smooth and interpolate robot pose trajectory to N frames.

    Args:
        eef_pos: (T, 3) end-effector positions
        eef_rot: (T, 3, 3) rotation matrices
        artic: (T) gripper articulation
        N: target number of frames
    """
    T = eef_pos.shape[0]
    # Smooth first
    window_len = 9
    # Smooth positions and joint angles
    eef_pos_smooth = savgol_filter(eef_pos, window_len, 3, axis=0)
    artic_smooth = savgol_filter(artic, window_len, 3, axis=0)

    # Smooth rotations
    rot_obj = R.from_matrix(eef_rot)

    # Simple rotation smoothing via quaternion averaging in windows
    quats = rot_obj.as_quat()
    quat_smooth = savgol_filter(quats, window_len, 3, axis=0)
    quat_smooth = quat_smooth / np.linalg.norm(quat_smooth, axis=1, keepdims=True)

    # Interpolate to N frames
    t_orig = np.linspace(0, 1, T)
    t_new = np.linspace(0, 1, N)

    # Interpolate positions and joints
    pos_interp = interp1d(t_orig, eef_pos_smooth, axis=0, kind="cubic")(t_new)
    artic_interp = interp1d(t_orig, artic_smooth, axis=0, kind="cubic")(t_new)

    # SLERP for rotations
    rot_smooth_obj = R.from_quat(quat_smooth)
    rot_interp = R.from_quat(interp1d(t_orig, rot_smooth_obj.as_quat(), axis=0)(t_new))
    rot_interp_normalized = R.from_quat(
        rot_interp.as_quat()
        / np.linalg.norm(rot_interp.as_quat(), axis=1, keepdims=True)
    )

    return pos_interp, rot_interp_normalized.as_matrix(), artic_interp
