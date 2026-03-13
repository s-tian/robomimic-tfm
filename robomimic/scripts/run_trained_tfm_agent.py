"""
The main script for evaluating a policy in an environment.

Args:
    agent (str): path to saved checkpoint pth file

    horizon (int): if provided, override maximum horizon of rollout from the one 
        in the checkpoint

    env (str): if provided, override name of env from the one in the checkpoint,
        and use it for rollouts

    render (bool): if flag is provided, use on-screen rendering during rollouts

    video_path (str): if provided, render trajectories to this video file path

    video_skip (int): render frames to a video every @video_skip steps

    camera_names (str or [str]): camera name(s) to use for rendering on-screen or to video

    dataset_path (str): if provided, an hdf5 file will be written at this path with the
        rollout data

    dataset_obs (bool): if flag is provided, and @dataset_path is provided, include 
        possible high-dimensional observations in output dataset hdf5 file (by default,
        observations are excluded and only simulator states are saved).

    seed (int): if provided, set seed for rollouts

Example usage:

    # Evaluate a policy with 50 rollouts of maximum horizon 400 and save the rollouts to a video.
    # Visualize the agentview and wrist cameras during the rollout.
    
    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --video_path /path/to/output.mp4 \
        --camera_names agentview robot0_eye_in_hand 

    # Write the 50 agent rollouts to a new dataset hdf5.

    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --dataset_path /path/to/output.hdf5 --dataset_obs 

    # Write the 50 agent rollouts to a new dataset hdf5, but exclude the dataset observations
    # since they might be high-dimensional (they can be extracted again using the
    # dataset_states_to_obs.py script).

    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --dataset_path /path/to/output.hdf5
"""
import argparse
import json
import h5py
import imageio
import numpy as np
from copy import deepcopy
from collections import deque
import tqdm
import omegaconf
import hydra
import torch
import random

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.envs.env_base import EnvBase
from robomimic.algo import RolloutPolicy
from robomimic.scripts.dataset_states_to_obs import get_camera_info

from tfm.models.utils.rotation_transformer import RotationTransformer
from tfm.scripts.train_utils import encode_language_instruction

rotation_converter = RotationTransformer(from_rep='axis_angle', to_rep='rotation_6d')
TASK_ENCODING = None

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


def _get_eval_data_cfg(cfg):
    data_cfg = omegaconf.OmegaConf.select(cfg, "data.data0")
    if data_cfg is not None:
        return data_cfg, "data.data0"
    return cfg.data, "data"


def _normalize_image_resolution(camera_heights, camera_widths):
    height = camera_heights[0] if isinstance(camera_heights, (list, tuple)) else camera_heights
    width = camera_widths[0] if isinstance(camera_widths, (list, tuple)) else camera_widths
    return [height, width]


def _get_obs_camera_resolution(obs, camera_name, fallback_height=None, fallback_width=None):
    image_key = f"{camera_name}_image"
    if image_key in obs:
        image = obs[image_key]
        if image.ndim >= 3 and image.shape[-1] in (1, 3):
            return np.array(image.shape[-3:-1])
        if image.ndim >= 3 and image.shape[-3] in (1, 3):
            return np.array(image.shape[-2:])
    if fallback_height is not None and fallback_width is not None:
        return np.array([fallback_height, fallback_width])
    return None


def _build_camera_info(env, obs, camera_names, camera_heights=None, camera_widths=None):
    if not EnvUtils.is_robosuite_env(env=env):
        return {}

    camera_info = get_camera_info(
        env=env,
        camera_names=camera_names,
        camera_height=camera_heights,
        camera_width=camera_widths,
    )
    for camera_name, params in camera_info.items():
        for param_key, value in params.items():
            camera_info[camera_name][param_key] = np.array(value)
        resolution = _get_obs_camera_resolution(
            obs,
            camera_name,
            fallback_height=camera_heights,
            fallback_width=camera_widths,
        )
        if resolution is not None:
            camera_info[camera_name]["resolution"] = resolution
    return camera_info


def _ensure_numpy_positive_strides(data):
    def _fix_array(arr):
        if any(stride < 0 for stride in arr.strides) or not arr.flags.c_contiguous:
            return np.ascontiguousarray(arr)
        return arr

    return TensorUtils.map_ndarray(data, _fix_array)


def _get_policy_obs_horizon(policy):
    if hasattr(policy.policy, "e2e_policy"):
        return policy.policy.e2e_policy.obs_horizon
    if hasattr(policy.policy, "tfm_policy"):
        return policy.policy.tfm_policy.obs_horizon
    if hasattr(policy.policy, "obs_horizon"):
        return policy.policy.obs_horizon
    return 1


def _stack_obs_history(obs_history):
    stacked_obs = {}
    latest_obs = obs_history[-1]
    for key, value in latest_obs.items():
        if isinstance(value, np.ndarray):
            stacked_obs[key] = np.stack([obs[key] for obs in obs_history], axis=0)
        else:
            stacked_obs[key] = value
    return stacked_obs

def rollout(policy, env, horizon, render=False, video_writer=None, video_skip=5, return_obs=False, camera_names=None, camera_heights=None, camera_widths=None):
    """
    Helper function to carry out rollouts. Supports on-screen rendering, off-screen rendering to a video, 
    and returns the rollout trajectory.

    Args:
        policy (instance of RolloutPolicy): policy loaded from a checkpoint
        env (instance of EnvBase): env loaded from a checkpoint or demonstration metadata
        horizon (int): maximum horizon for the rollout
        render (bool): whether to render rollout on-screen
        video_writer (imageio writer): if provided, use to write rollout to video
        video_skip (int): how often to write video frames
        return_obs (bool): if True, return possibly high-dimensional observations along the trajectory. 
            They are excluded by default because the low-dimensional simulation states should be a minimal 
            representation of the environment. 
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.

    Returns:
        stats (dict): some statistics for the rollout - such as return, horizon, and task success
        traj (dict): dictionary that corresponds to the rollout trajectory
    """
    # assert isinstance(env, EnvBase)
    assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))

    policy.start_episode()
    obs = env.reset()
    state_dict = env.get_state()

    # hack that is necessary for robosuite tasks for deterministic action playback
    obs = env.reset_to(state_dict)
    obs = _ensure_numpy_positive_strides(obs)
    obs_horizon = _get_policy_obs_horizon(policy)
    obs_history = deque([deepcopy(obs) for _ in range(obs_horizon)], maxlen=obs_horizon)

    results = {}
    video_count = 0  # video frame counter
    total_reward = 0.
    traj = dict(actions=[], rewards=[], dones=[], states=[], initial_state_dict=state_dict)
    if return_obs:
        # store observations too
        traj.update(dict(obs=[], next_obs=[]))
    try:
        for step_i in tqdm.tqdm(range(horizon)):
            stacked_obs = _stack_obs_history(obs_history)
            camera_info = _build_camera_info(
                env=env,
                obs=stacked_obs,
                camera_names=camera_names,
                camera_heights=camera_heights,
                camera_widths=camera_widths,
            )
            # add camera info
            stacked_obs["camera_info"] = camera_info
            stacked_obs["task_encoding"] = TASK_ENCODING
            # scale all image obs by 255 and convert to uint8 for more efficient storage and processing in the policy
            for key in stacked_obs:
                if key.endswith("_image") and stacked_obs[key].dtype == np.float32:
                    stacked_obs[key] = (stacked_obs[key] * 255).astype(np.uint8)
            # get action from policy
            act = policy.policy.get_action(obs_dict=stacked_obs)[0].detach().cpu().numpy()
            if "rot6d" in policy.policy.cfg.model and policy.policy.cfg.model.rot6d and policy.policy.cfg.model.predict_ee_actions:
                rot = act[..., 3:9]
                rot = rotation_converter.inverse(rot)
                act = np.concatenate([act[..., :3], rot, act[..., 9:]], axis=-1)
            act[-1] = (act[-1] * 2) - 1

            # play action
            next_obs, r, done, _ = env.step(act)
            next_obs = _ensure_numpy_positive_strides(next_obs)
            obs_history.append(deepcopy(next_obs))

            # compute reward
            total_reward += r
            success = env.is_success()["task"]

            # visualization
            if render:
                env.render(mode="human", camera_name=camera_names[0])
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        video_img.append(env.render(mode="rgb_array", height=256, width=256, camera_name=cam_name))
                    video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                    video_writer.append_data(video_img)
                video_count += 1

            # collect transition
            traj["actions"].append(act)
            traj["rewards"].append(r)
            traj["dones"].append(done)
            traj["states"].append(state_dict["states"])
            if return_obs:
                # Note: We need to "unprocess" the observations to prepare to write them to dataset.
                #       This includes operations like channel swapping and float to uint8 conversion
                #       for saving disk space.
                traj["obs"].append(ObsUtils.unprocess_obs_dict(stacked_obs))
                traj["next_obs"].append(ObsUtils.unprocess_obs_dict(next_obs))

            # break if done or if success
            if done or success:
                break

            # update for next iter
            obs = deepcopy(next_obs)
            state_dict = env.get_state()

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success))

    if return_obs:
        # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
        traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
        traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return stats, traj


@hydra.main(config_path="../../../tfm/tfm/configs", config_name="eval_config")
def run_trained_agent(cfg):
    # some arg checking
    write_video = (cfg.rollout.video_path is not None)
    assert not (cfg.rollout.render and write_video) # either on-screen or video but not both
    if cfg.rollout.render:
        # on-screen rendering can only support one camera
        assert len(cfg.rollout.camera_names) == 1

    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # read rollout settings
    rollout_num_episodes = cfg.rollout.n_rollouts
    rollout_horizon = cfg.rollout.horizon
    data_cfg, data_cfg_path = _get_eval_data_cfg(cfg)
    if rollout_horizon is None:
        # read horizon from config
        config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        rollout_horizon = config.experiment.rollout.horizon

    # create environment from saved checkpoint
    try:
        env_meta = FileUtils.get_env_metadata_from_dataset(data_cfg.hdf5_path)
    except:
        env_meta = FileUtils.get_env_metadata_from_dataset(data_cfg.zarr_path)
    ### Uncomment for joint position control
    # controller_config = {
    #     'type': 'JOINT_POSITION', 
    #     'input_max': np.pi, 
    #     'input_min': -np.pi, 
    #     'output_max': np.pi, 
    #     'output_min': -np.pi, 
    #     'kp': 50, 
    #     'damping_ratio': 1, 
    #     'input_type': 'absolute',
    #     'impedance_mode': 'fixed', 
    #     'kp_limits': [0, 300], 
    #     'damping_ratio_limits': [0, 10], 
    #     'qpos_limits': None, 
    #     'interpolation': None, 
    #     'ramp_ratio': 0.2,
    #     'gripper': {'type': 'GRIP'},
    # }
    # env_meta['env_kwargs']['controller_configs'] = controller_config
    # env_meta['env_kwargs']['controller_configs']['body_parts']['right'] = controller_config
    ### End uncomment for joint position control
    # env_meta['env_kwargs']['controller_configs']['body_parts']['right']['input_type'] = 'absolute'
    env_meta['env_kwargs']['controller_configs']['control_delta'] = False
    # breakpoint()

    # env_meta["env_kwargs"]["camera_names"].remove("robot0_eye_in_hand")
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        render_offscreen=True,
        use_image_obs=env_meta["env_kwargs"].get("use_camera_obs", False), 
        use_depth_obs=env_meta["env_kwargs"].get("camera_depths", False),
    ) 
    image_resolution = _normalize_image_resolution(
        env_meta["env_kwargs"]["camera_heights"],
        env_meta["env_kwargs"]["camera_widths"],
    )
    omegaconf.OmegaConf.update(cfg, f"{data_cfg_path}.image_resolution", image_resolution, force_add=True)

    # restore policy
    if cfg.target_class is not None:
        target_class = hydra.utils.get_class(cfg.target_class)
        policy = RolloutPolicy(target_class(cfg, log_wandb=cfg.use_wandb, log_verbose=cfg.log_verbose))
        from tfm.models.robomimic_algo import E2ETFMAlgo, TFMAlgo
        try:
            if isinstance(policy.policy, E2ETFMAlgo):
                robomimic_data_config = {"obs": policy.policy.e2e_policy.cfg.data.data0.obs_config}
            elif isinstance(policy.policy, TFMAlgo):
                robomimic_data_config = {"obs": policy.policy.tfm_policy.cfg.data.data0.obs_config}
            else:
                robomimic_data_config = {"obs": policy.policy.cfg.data.data0.obs_config}
        except:
            print("Defaulting on robomimic data config")
            # use default
            robomimic_data_config = {
                "obs": {
                    "low_dim": [
                        "robot0_eef_pos",
                        "robot0_eef_quat_site",
                        "robot0_joint_pos",
                        "robot0_gripper_qpos",
                    ],
                    "rgb": ["agentview_image"],
                    "depth": ["agentview_depth"],
                    "scan": []
                },
            }
        # init dataloader
        # train_dataset = get_dataset(cfg, "train")
        ObsUtils.initialize_obs_utils_with_obs_specs(robomimic_data_config)
    else:
        policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=cfg.ckpt_path, device=device, verbose=True)

    global TASK_ENCODING
    task_language_instruction = data_cfg.get("task_language_instruction", "")
    if task_language_instruction:
        TASK_ENCODING = encode_language_instruction(task_language_instruction)
    else:
        num_tasks = len(cfg.data) if omegaconf.OmegaConf.select(cfg, "data.data0") is not None else 1
        TASK_ENCODING = torch.nn.functional.one_hot(
            torch.tensor(cfg.rollout.task_id),
            num_tasks,
        ).float().numpy()

    # maybe set seed
    if cfg.rollout.seed is not None:
        np.random.seed(cfg.rollout.seed)
        random.seed(cfg.rollout.seed)
        torch.manual_seed(cfg.rollout.seed)

    # maybe create video writer
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(cfg.rollout.video_path, fps=20)

    # maybe open hdf5 to write rollouts
    write_dataset = (cfg.rollout.dataset_path is not None)
    if write_dataset:
        data_writer = h5py.File(cfg.rollout.dataset_path, "w")
        data_grp = data_writer.create_group("data")
        total_samples = 0

    if cfg.use_wandb:
        import wandb
        import datetime
        wandb.init(project="tfm-rollouts", name=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    rollout_stats = []
    for i in range(rollout_num_episodes):
        stats, traj = rollout(
            policy=policy, 
            env=env, 
            horizon=rollout_horizon, 
            render=cfg.rollout.render, 
            video_writer=video_writer, 
            video_skip=cfg.rollout.video_skip, 
            return_obs=(write_dataset and cfg.rollout.dataset_obs),
            camera_names=cfg.rollout.camera_names,
            camera_heights=env_meta['env_kwargs']['camera_heights'],
            camera_widths=env_meta['env_kwargs']['camera_widths'],
        )
        rollout_stats.append(stats)

        if write_dataset:
            # store transitions
            ep_data_grp = data_grp.create_group("demo_{}".format(i))
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            if cfg.rollout.dataset_obs:
                for k in traj["obs"]:
                    ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
                    ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

            # episode metadata
            if "model" in traj["initial_state_dict"]:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"] # model xml for this episode
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode
            total_samples += traj["actions"].shape[0]

        rollout_stats_dict = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
        avg_rollout_stats = { k : np.mean(rollout_stats_dict[k]) for k in rollout_stats_dict }
        avg_rollout_stats["Num_Success"] = np.sum(rollout_stats_dict["Success_Rate"])
        print("Average Rollout Stats")
        print(json.dumps(avg_rollout_stats, indent=4))

    if write_video:
        video_writer.close()

    if write_dataset:
        # global metadata
        data_grp.attrs["total"] = total_samples
        data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4) # environment info
        data_writer.close()
        print("Wrote dataset trajectories to {}".format(cfg.rollout.dataset_path))

    env.env.close()


if __name__ == "__main__":
    run_trained_agent()
