from rllab.envs.gym_env import GymEnv
import os
import apimock as spx
import shutil

dat_folda = "test"

if os.path.exists(dat_folda):
    shutil.rmtree(dat_folda)

env = GymEnv("LunarLander-v2", log_dir=dat_folda)

spx.init(env.observation_space.low.shape[0], env.action_space.n)

for i in range(100000):
    o = env.reset()

    spx.reset()

    while True:

        a = spx.get_action(o)
        o, r, d, env_info = env.step(a)
        spx.reward(r)

        if d:
            break

