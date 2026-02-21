import gymnasium as gym


def _make_env(gym_id, seed, idx, capture_video, render, run_name):
    def thunk():
        if render and idx == 0:
            render_mode = "human"
        elif capture_video and idx == 0:
            render_mode = "rgb_array"
        else:
            render_mode = None
        env = gym.make(gym_id, render_mode=render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

def make_envs(args, run_name):
    return gym.vector.SyncVectorEnv(
        [
            _make_env(args.gym_id, args.seed + i, i, args.capture_video, args.render, run_name)
            for i in range(args.num_envs)
        ]
    ) 