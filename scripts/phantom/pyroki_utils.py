"""
Solves the basic IK problem.
"""
from typing import Sequence

import numpy as onp
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
from jax.typing import ArrayLike
from scipy.spatial.transform import Rotation

import pyroki as pk

PYROKI2MUJOCO = [0, 1, 2, 3, 4, 10, 11, 12, 5, 6, 7, 8, 9, 13, 14, 15] # joint reorder

def solve_ik_with_manipulability(
    robot: pk.Robot,
    target_link_name: str,
    target_position: onp.ndarray,
    target_wxyz: onp.ndarray,
    manipulability_weight: float = 0.0,
) -> onp.ndarray:
    """
    Solves the basic IK problem for a robot, with manipulability cost.

    Args:
        robot: PyRoKi Robot.
        target_link_name: str.
        position: onp.ndarray. Shape: (3,).
        wxyz: onp.ndarray. Shape: (4,).
        manipulability_weight: float. Weight for the manipulability cost.

    Returns:
        cfg: onp.ndarray. Shape: (robot.joint.actuated_count,).
    """
    assert target_position.shape == (3,) and target_wxyz.shape == (4,)

    assert target_position.shape == (3,) and target_wxyz.shape == (4,)
    target_link_idx = robot.links.names.index(target_link_name)

    T_world_target = jaxlie.SE3(
        jnp.concatenate([jnp.array(target_wxyz), jnp.array(target_position)], axis=-1)
    )
    cfg = _solve_ik_jax(
        robot,
        T_world_target,
        jnp.array(target_link_idx),
        jnp.array(manipulability_weight),
    )
    assert cfg.shape == (robot.joints.num_actuated_joints,)

    return onp.array(cfg)


def solve_ik(
    robot: pk.Robot,
    target_link_name: str,
    target_wxyz: onp.ndarray,
    target_position: onp.ndarray,
    timesteps: int | None = None,
    dt: float | None = None,
) -> onp.ndarray:
    """
    Solves the basic IK problem for a robot.

    Args:
        robot: PyRoKi Robot.
        target_link_name: String name of the link to be controlled.
        target_wxyz: onp.ndarray. Target orientation.
        target_position: onp.ndarray. Target position.

    Returns:
        cfg: onp.ndarray. Shape: (robot.joint.actuated_count,).
    """
    assert target_position.shape == (3,) and target_wxyz.shape == (4,)
    target_link_index = robot.links.names.index(target_link_name)
    
    if timesteps and dt:
        cfg = _solve_ik_jax_opt(
        robot,
        jnp.array(target_link_index),
        jnp.array(target_wxyz),
        jnp.array(target_position),
        timesteps, 
        dt
    )
    else:
        cfg = _solve_ik_jax(
            robot,
            jnp.array(target_link_index),
            jnp.array(target_wxyz),
            jnp.array(target_position),
        )
    assert cfg.shape == (robot.joints.num_actuated_joints,)
    return onp.array(cfg)


@jdc.jit
def _solve_ik_jax(
    robot: pk.Robot,
    target_link_index: jax.Array,
    target_wxyz: jax.Array,
    target_position: jax.Array,
) -> jax.Array:
    joint_var = robot.joint_var_cls(0)
    factors = [
        pk.costs.pose_cost_analytic_jac(
            robot,
            joint_var,
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyz), target_position
            ),
            target_link_index,
            pos_weight=50.0,
            ori_weight=10.0,
        ),
        pk.costs.limit_cost(
            robot,
            joint_var,
            weight=100.0,
        ),
    ]
    sol = (
        jaxls.LeastSquaresProblem(factors, [joint_var])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        )
    )
    return sol[joint_var]


def _solve_ik_jax_opt(
    robot: pk.Robot,
    target_link_index: jax.Array,
    target_wxyz: jax.Array,
    target_position: jax.Array,
    timesteps: int,
    dt: float,
) -> jax.Array:
    traj_vars = robot.joint_var_cls(jnp.arange(timesteps))
    target_idx = timesteps // 2 - (1 - timesteps % 2)
    target_se3 = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(target_wxyz),
        target_position,
    )

    factors = [
        pk.costs.pose_cost_analytic_jac(
            robot,
            traj_vars[target_idx],
            target_se3,
            target_link_index,
            pos_weight=50.0,
            ori_weight=10.0,
        ),
        pk.costs.limit_cost(
            robot,
            traj_vars[target_idx],
            weight=100.0,
        ),
    ]

    if timesteps >= 2:
        factors.append(
            pk.costs.smoothness_cost(
                traj_vars[1:],
                traj_vars[:-1],
                jnp.array([0.1])[None],
            )
        )

    if timesteps >= 7:
        factors.append(
            pk.costs.five_point_jerk_cost(
                traj_vars[jnp.arange(6, timesteps)],
                traj_vars[jnp.arange(5, timesteps - 1)],
                traj_vars[jnp.arange(4, timesteps - 2)],
                traj_vars[jnp.arange(2, timesteps - 4)],
                traj_vars[jnp.arange(1, timesteps - 5)],
                traj_vars[jnp.arange(0, timesteps - 6)],
                dt,
                jnp.array([0.1])[None],
            )
        )

    sol = (
        jaxls.LeastSquaresProblem(factors, [traj_vars])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        )
    )

    return sol[traj_vars][target_idx]


def solve_trajopt(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    world_coll: Sequence[pk.collision.CollGeom],
    target_link_name: str,
    start_position: ArrayLike,
    start_wxyz: ArrayLike,
    end_position: ArrayLike,
    end_wxyz: ArrayLike,
    timesteps: int,
    dt: float,
) -> ArrayLike:
    if isinstance(start_position, onp.ndarray):
        np = onp
    elif isinstance(start_position, jnp.ndarray):
        np = jnp
    else:
        raise ValueError(f"Invalid type for `ArrayLike`: {type(start_position)}")

    # 1. Solve IK for the start and end poses.
    target_link_index = robot.links.names.index(target_link_name)
    start_cfg, end_cfg = solve_iks_with_collision(
        robot=robot,
        coll=robot_coll,
        world_coll_list=world_coll,
        target_link_index=target_link_index,
        target_position_0=jnp.array(start_position),
        target_wxyz_0=jnp.array(start_wxyz),
        target_position_1=jnp.array(end_position),
        target_wxyz_1=jnp.array(end_wxyz),
    )

    # 2. Initialize the trajectory through linearly interpolating the start and end poses.
    init_traj = jnp.linspace(start_cfg, end_cfg, timesteps)

    # 3. Optimize the trajectory.
    traj_vars = robot.joint_var_cls(jnp.arange(timesteps))

    robot = jax.tree.map(lambda x: x[None], robot)  # Add batch dimension.
    robot_coll = jax.tree.map(lambda x: x[None], robot_coll)  # Add batch dimension.

    # Basic regularization / limit costs.
    factors: list[jaxls.Cost] = [
        pk.costs.rest_cost(
            traj_vars,
            traj_vars.default_factory()[None],
            jnp.array([0.01])[None],
        ),
        pk.costs.limit_cost(
            robot,
            traj_vars,
            jnp.array([100.0])[None],
        ),
    ]

    # Collision avoidance.
    def compute_world_coll_residual(
        vals: jaxls.VarValues,
        robot: pk.Robot,
        robot_coll: pk.collision.RobotCollision,
        world_coll_obj: pk.collision.CollGeom,
        prev_traj_vars: jaxls.Var[jax.Array],
        curr_traj_vars: jaxls.Var[jax.Array],
    ):
        coll = robot_coll.get_swept_capsules(
            robot, vals[prev_traj_vars], vals[curr_traj_vars]
        )
        dist = pk.collision.collide(
            coll.reshape((-1, 1)), world_coll_obj.reshape((1, -1))
        )
        colldist = pk.collision.colldist_from_sdf(dist, 0.1)
        return (colldist * 20.0).flatten()

    for world_coll_obj in world_coll:
        factors.append(
            jaxls.Cost(
                compute_world_coll_residual,
                (
                    robot,
                    robot_coll,
                    jax.tree.map(lambda x: x[None], world_coll_obj),
                    robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
                    robot.joint_var_cls(jnp.arange(1, timesteps)),
                ),
                name="World Collision (sweep)",
            )
        )

    # Start / end pose constraints.
    factors.extend(
        [
            jaxls.Cost(
                lambda vals, var: ((vals[var] - start_cfg) * 100.0).flatten(),
                (robot.joint_var_cls(jnp.arange(0, 2)),),
                name="start_pose_constraint",
            ),
            jaxls.Cost(
                lambda vals, var: ((vals[var] - end_cfg) * 100.0).flatten(),
                (robot.joint_var_cls(jnp.arange(timesteps - 2, timesteps)),),
                name="end_pose_constraint",
            ),
        ]
    )

    # Velocity / acceleration / jerk minimization.
    factors.extend(
        [
            pk.costs.smoothness_cost(
                robot.joint_var_cls(jnp.arange(1, timesteps)),
                robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
                jnp.array([0.1])[None],
            ),
            pk.costs.five_point_velocity_cost(
                robot,
                robot.joint_var_cls(jnp.arange(4, timesteps)),
                robot.joint_var_cls(jnp.arange(3, timesteps - 1)),
                robot.joint_var_cls(jnp.arange(1, timesteps - 3)),
                robot.joint_var_cls(jnp.arange(0, timesteps - 4)),
                dt,
                jnp.array([10.0])[None],
            ),
            pk.costs.five_point_acceleration_cost(
                robot.joint_var_cls(jnp.arange(2, timesteps - 2)),
                robot.joint_var_cls(jnp.arange(4, timesteps)),
                robot.joint_var_cls(jnp.arange(3, timesteps - 1)),
                robot.joint_var_cls(jnp.arange(1, timesteps - 3)),
                robot.joint_var_cls(jnp.arange(0, timesteps - 4)),
                dt,
                jnp.array([0.1])[None],
            ),
            pk.costs.five_point_jerk_cost(
                robot.joint_var_cls(jnp.arange(6, timesteps)),
                robot.joint_var_cls(jnp.arange(5, timesteps - 1)),
                robot.joint_var_cls(jnp.arange(4, timesteps - 2)),
                robot.joint_var_cls(jnp.arange(2, timesteps - 4)),
                robot.joint_var_cls(jnp.arange(1, timesteps - 5)),
                robot.joint_var_cls(jnp.arange(0, timesteps - 6)),
                dt,
                jnp.array([0.1])[None],
            ),
        ]
    )

    # 4. Solve the optimization problem.
    solution = (
        jaxls.LeastSquaresProblem(
            factors,
            [traj_vars],
        )
        .analyze()
        .solve(
            initial_vals=jaxls.VarValues.make((traj_vars.with_value(init_traj),)),
        )
    )
    return np.array(solution[traj_vars])


@jdc.jit
def solve_iks_with_collision(
    robot: pk.Robot,
    coll: pk.collision.RobotCollision,
    world_coll_list: Sequence[pk.collision.CollGeom],
    target_link_index: int,
    target_position_0: jax.Array,
    target_wxyz_0: jax.Array,
    target_position_1: jax.Array,
    target_wxyz_1: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Solves the basic IK problem with collision avoidance. Returns joint configuration."""
    joint_var_0 = robot.joint_var_cls(0)
    joint_var_1 = robot.joint_var_cls(1)
    joint_vars = robot.joint_var_cls(jnp.arange(2))
    vars = [joint_vars]

    # Weights and margins defined directly in factors.
    factors = [
        pk.costs.pose_cost(
            robot,
            joint_var_0,
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyz_0), target_position_0
            ),
            jnp.array(target_link_index),
            jnp.array([5.0] * 3),
            jnp.array([1.0] * 3),
        ),
        pk.costs.pose_cost(
            robot,
            joint_var_1,
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyz_1), target_position_1
            ),
            jnp.array(target_link_index),
            jnp.array([5.0] * 3),
            jnp.array([1.0] * 3),
        ),
    ]
    factors.extend(
        [
            pk.costs.limit_cost(
                jax.tree.map(lambda x: x[None], robot),
                joint_vars,
                jnp.array(100.0),
            ),
            pk.costs.rest_cost(
                joint_vars,
                jnp.array(joint_vars.default_factory()[None]),
                jnp.array(0.001),
            ),
            pk.costs.self_collision_cost(
                jax.tree.map(lambda x: x[None], robot),
                jax.tree.map(lambda x: x[None], coll),
                joint_vars,
                0.02,
                5.0,
            ),
        ]
    )
    factors.extend(
        [
            pk.costs.world_collision_cost(
                jax.tree.map(lambda x: x[None], robot),
                jax.tree.map(lambda x: x[None], coll),
                joint_vars,
                jax.tree.map(lambda x: x[None], world_coll),
                0.05,
                10.0,
            )
            for world_coll in world_coll_list
        ]
    )

    # Small cost to encourage the start + end configs to be close to each other.
    @jaxls.Cost.create_factory(name="JointSimilarityCost")
    def joint_similarity_cost(vals, var_0, var_1):
        return ((vals[var_0] - vals[var_1]) * 0.01).flatten()

    factors.append(joint_similarity_cost(joint_var_0, joint_var_1))

    sol = jaxls.LeastSquaresProblem(factors, vars).analyze().solve(verbose=False)
    return sol[joint_var_0], sol[joint_var_1]


def inverse_kinematics_pk(robot, target_position, gripper_rot, target_link_name = "right/aloha_vx300s/ee_gripper_link", timesteps=None, dt=None):
    trajectory_orientations = Rotation.from_matrix(gripper_rot).as_quat(scalar_first=True)

    solution = solve_ik(
            robot=robot,
            target_link_name=target_link_name,
            target_position=onp.array(target_position),
            target_wxyz=onp.array(trajectory_orientations),
            timesteps=timesteps, 
            dt=dt,
        )
    
    # Reorder joint
    return solution[PYROKI2MUJOCO]

def inverse_kinematics_pk_traj_opt(robot, robot_coll, world_coll, eef_pos, eef_rot, frame_idx, timesteps=10, dt=0.06666, target_link_name = "right/aloha_vx300s/ee_gripper_link"):
    #Frame wise retarget
    n_poses = eef_pos.shape[0]
    assert timesteps < n_poses, "num of frames need to be greater than timesteps"

    start_idx = max(0, frame_idx - timesteps//2 + (1 - timesteps%2))
    end_idx = min(n_poses - 1, frame_idx + timesteps//2)

    solution_time_step = 0
    if start_idx == 0:
        end_idx = timesteps-1
        solution_time_step = frame_idx
    elif end_idx == n_poses-1:
        start_idx = n_poses-timesteps
        solution_time_step =  frame_idx - start_idx
    else:
        solution_time_step = timesteps//2 - (1 - timesteps%2)

    start_pos, start_rot = eef_pos[start_idx], eef_rot[start_idx]
    end_pos, end_rot = eef_pos[end_idx], eef_rot[end_idx]

    start_trajectory_orientations = Rotation.from_matrix(start_rot).as_quat(scalar_first=True)
    end_trajectory_orientations = Rotation.from_matrix(end_rot).as_quat(scalar_first=True)

    solution = solve_trajopt(
        robot, robot_coll, world_coll, target_link_name,
        start_pos, start_trajectory_orientations,
        end_pos, end_trajectory_orientations,
        timesteps=timesteps , dt=dt
    )
    if timesteps % 2 == 0 and solution_time_step != timesteps-1:
        solution = (solution[solution_time_step] + solution[solution_time_step + 1 ])/2
    else:
        solution = solution[solution_time_step]
    
    return solution[PYROKI2MUJOCO]

def inverse_kinematics_pk_traj_opt_v2(robot, robot_coll, world_coll, eef_pos, eef_rot, dt=0.06666, target_link_name = "right/aloha_vx300s/ee_gripper_link"):
    timesteps = eef_pos.shape[0]
    start_pos, start_rot = eef_pos[0], eef_rot[0]
    end_pos, end_rot = eef_pos[-1], eef_rot[-1]

    start_trajectory_orientations = Rotation.from_matrix(start_rot).as_quat(scalar_first=True)
    end_trajectory_orientations = Rotation.from_matrix(end_rot).as_quat(scalar_first=True)

    solution = solve_trajopt(
        robot, robot_coll, world_coll, target_link_name,
        start_pos, start_trajectory_orientations,
        end_pos, end_trajectory_orientations,
        timesteps=timesteps , dt=dt
    )
    assert eef_pos.shape[0] == solution.shape[0], "solution length should be same as input length"
    
    return solution[: ,PYROKI2MUJOCO]

   
