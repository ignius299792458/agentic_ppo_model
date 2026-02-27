"""Environment creation and wrapping.

Creates vectorized Gymnasium environments with episode statistics tracking,
optional video recording, and optional live rendering.
"""

from typing import Callable
import gymnasium as gym


def _make_env(gym_id, seed, idx, capture_video, run_name, render, renderAll=False) -> Callable:
    """Create a closure that builds a single environment instance.

    Only env index 0 gets rendering or video recording to avoid
    redundant overhead across parallel environments.
    renderAll: If True, render all environments.

    Args:
        gym_id: Gymnasium environment ID (e.g. "CartPole-v1").
        seed: Random seed for this environment's action/observation spaces.
        idx: Index of this env within the vectorized set (0-based).
        capture_video: If True and idx==0, wrap with RecordVideo (rgb_array mode).
        render: If True and idx==0, use human render mode (live window).
        run_name: Experiment name used for the video save directory.
        renderAll: If True, render all environments.
    Returns:
        Callable that creates and returns the configured environment.
    """
    def thunk():
        if renderAll:
            render_mode = "human"
        elif render and idx == 0:
            render_mode = "human"
        elif capture_video and idx == 0:
            render_mode = "rgb_array"
            render_mode = None
        env = gym.make(gym_id, render_mode=render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def make_envs(args, run_name) -> gym.vector.SyncVectorEnv:
    """Create a synchronous vectorized environment.

    Runs num_envs copies of the environment in a single process.
    Each env gets a unique seed (base seed + index).

    Args:
        args: Parsed arguments with gym_id, seed, num_envs, capture_video, render, renderAll.
        run_name: Experiment name for video directory naming.

    Returns:
        gym.vector.SyncVectorEnv: Vectorized environment.
    """
    return gym.vector.SyncVectorEnv(
        [
            _make_env(
                args.gym_id, args.seed + i, i, args.capture_video, args.render, args.renderAll, run_name
            )
            for i in range(args.num_envs)
        ]
    )
