envs_dict = {
    "cartpole": "gym.envs.classic_control:CartPoleEnv",
    "pendulum": "gym.envs.classic_control:PendulumEnv",
    # "Standard" Mujoco Envs
    "halfcheetah": "gym.envs.mujoco.half_cheetah:HalfCheetahEnv",
    "ant": "gym.envs.mujoco.ant:AntEnv",
    "hopper": "gym.envs.mujoco.hopper:HopperEnv",
    "walker": "gym.envs.mujoco.walker2d:Walker2dEnv",
    "humanoid": "gym.envs.mujoco.humanoid:HumanoidEnv",
    "swimmer": "gym.envs.mujoco.swimmer:SwimmerEnv",
    "inverteddoublependulum": "gym.envs.mujoco.inverted_double_pendulum:InvertedDoublePendulumEnv",
    "invertedpendulum": "gym.envs.mujoco.inverted_pendulum:InvertedPendulumEnv",
    "antmaze": "d4rl.locomotion.ant:make_ant_maze_env",
    "maze": "d4rl.pointmaze:MazeEnv",
    # "Standard" Mujoco Envs V4
    "halfcheetah_v4": "gym.envs.mujoco.half_cheetah_v4:HalfCheetahEnv",
    "ant_v4": "gym.envs.mujoco.ant_v4:AntEnv",
    "hopper_v4": "gym.envs.mujoco.hopper_v4:HopperEnv",
    "walker_v4": "gym.envs.mujoco.walker2d_v4:Walker2dEnv",
    "humanoid_v4": "gym.envs.mujoco.humanoid_v4:HumanoidEnv",
    "swimmer_v4": "gym.envs.mujoco.swimmer_v4:SwimmerEnv",
    "inverteddoublependulum_v4": "gym.envs.mujoco.inverted_double_pendulum_v4:InvertedDoublePendulumEnv",
    "invertedpendulum_v4": "gym.envs.mujoco.inverted_pendulum_v4:InvertedPendulumEnv",
    # normal envs
    "lunarlandercont": "gym.envs.box2d.lunar_lander:LunarLanderContinuous",
    # robotics envs
    "fetch-reach": "gym.envs.robotics.fetch.reach:FetchReachEnv",
    "fetch-push": "gym.envs.robotics.fetch.push:FetchPushEnv",
    "fetch-pick-place": "gym.envs.robotics.fetch.pick_and_place:FetchPickAndPlaceEnv",
    "fetch-slide": "gym.envs.robotics.fetch.slide:FetchSlideEnv",
}
