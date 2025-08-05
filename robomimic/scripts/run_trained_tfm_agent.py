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
import tqdm
import omegaconf
import hydra
import torch

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

rotation_converter = RotationTransformer(from_rep='axis_angle', to_rep='rotation_6d')

TASK_ID = 1

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

    results = {}
    video_count = 0  # video frame counter
    total_reward = 0.
    traj = dict(actions=[], rewards=[], dones=[], states=[], initial_state_dict=state_dict)
    if return_obs:
        # store observations too
        traj.update(dict(obs=[], next_obs=[]))
    try:
        for step_i in tqdm.tqdm(range(horizon)):
            if EnvUtils.is_robosuite_env(env=env):
                camera_info = get_camera_info(
                    env=env,
                    camera_names=camera_names, 
                    camera_height=camera_heights, 
                    camera_width=camera_widths,
                )
                for camera_name in camera_info.keys():
                    # convert the 'intrinsics' and 'extrinsics' keys to np arrays
                    for param_key in camera_info[camera_name].keys():
                        camera_info[camera_name][param_key] = np.array(camera_info[camera_name][param_key])
                    
            # add camera info
            obs["camera_info"] = camera_info
            num_tasks = len(policy.policy.cfg.data)
            obs["task_encoding"] = torch.nn.functional.one_hot(torch.tensor(TASK_ID), num_tasks).float().numpy()
            # get action from policy
            act = policy(ob=obs)
            if "rot6d" in policy.policy.cfg.model and policy.policy.cfg.model.rot6d and policy.policy.cfg.model.predict_ee_actions:
            # if True:
                rot = act[..., 3:9]
                rot = rotation_converter.inverse(rot)
                act = np.concatenate([act[..., :3], rot, act[..., 9:]], axis=-1)

            # play action
            next_obs, r, done, _ = env.step(act)

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
                traj["obs"].append(ObsUtils.unprocess_obs_dict(obs))
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
    if rollout_horizon is None:
        # read horizon from config
        config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        rollout_horizon = config.experiment.rollout.horizon

    # create environment from saved checkpoint
    env_meta = FileUtils.get_env_metadata_from_dataset(cfg.data.hdf5_path)
    ### Uncomment for joint position control
    controller_config = {
        'type': 'JOINT_POSITION', 
        'input_max': np.pi, 
        'input_min': -np.pi, 
        'output_max': np.pi, 
        'output_min': -np.pi, 
        'kp': 50, 
        'damping_ratio': 1, 
        'input_type': 'absolute',
        'impedance_mode': 'fixed', 
        'kp_limits': [0, 300], 
        'damping_ratio_limits': [0, 10], 
        'qpos_limits': None, 
        'interpolation': None, 
        'ramp_ratio': 0.2,
        'gripper': {'type': 'GRIP'},
    }
    env_meta['env_kwargs']['controller_configs']['body_parts']['right'] = controller_config
    ### End uncomment for joint position control
    # env_meta['env_kwargs']['controller_configs']['body_parts']['right']['input_type'] = 'absolute'

    # env_meta["env_kwargs"]["camera_names"].remove("robot0_eye_in_hand")

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=True, 
        render_offscreen=True,
        use_image_obs=env_meta["env_kwargs"].get("use_camera_obs", False), 
        use_depth_obs=env_meta["env_kwargs"].get("camera_depths", False),
    ) 
    omegaconf.OmegaConf.update(cfg, "data.data0.image_resolution", env_meta["env_kwargs"]["camera_heights"], force_add=True)

    from robomimic.envs.wrappers import FrameStackWrapper
    env = FrameStackWrapper(env, num_frames=2)

    # restore policy
    if cfg.target_class is not None:
        target_class = hydra.utils.get_class(cfg.target_class)
        policy = RolloutPolicy(target_class(cfg, log_wandb=cfg.use_wandb))
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

    global TASK_ID
    TASK_ID = cfg.rollout.task_id

    # maybe set seed
    if cfg.rollout.seed is not None:
        np.random.seed(cfg.rollout.seed)
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


if __name__ == "__main__":
    run_trained_agent()