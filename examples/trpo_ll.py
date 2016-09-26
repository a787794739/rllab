from __future__ import print_function
from __future__ import absolute_import

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy

import os
import shutil

dat_folda = "t2"

if os.path.exists(dat_folda):
    shutil.rmtree(dat_folda)

env = GymEnv("LunarLander-v2", log_dir=dat_folda)

policy = CategoricalMLPPolicy(
    env_spec=env.spec,
    hidden_sizes=(8, 8)
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=4000,
    max_path_length=env.horizon,
    n_itr=500,
    discount=0.99,
    step_size=0.01,
    # Uncomment both lines (this and the plot parameter below) to enable plotting
    # plot=True,
)

algo.train()

