from __future__ import print_function
from __future__ import absolute_import

import rllab
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.misc import tensor_utils
import numpy as np

import rllab.misc.logger as logger

class spec_stub():
    def __init__(self, Xsz, Ysz):
        self.action_space = rllab.spaces.Discrete(Ysz)
        lg = (10 ** 10)
        self.observation_space = rllab.spaces.Box(-lg, lg, (Xsz,))

    def log_diagnostics(self, paths):
        pass # lol

class SharedVars():
    def __init__(self):
        self.env = None
        self.agent = None
        self.algor = None

        # all paths ...
        self.paths = []
        self.count = 0
        self.itera = 0

        # variables for data collection ...
        self.observations = None
        self.actions = None
        self.rewards = None
        self.agent_infos = None
        self.env_infos = None

sh = SharedVars()

def init(Xsz, Ysz):
    sh.env = spec_stub(Xsz, Ysz)

    sh.agent = CategoricalMLPPolicy(
        env_spec=sh.env,
        hidden_sizes=(8, 8)
    )

    baseline = LinearFeatureBaseline(env_spec=sh.env)

    sh.algor = TRPO(
        env=sh.env,
        policy=sh.agent,
        baseline=baseline,
        batch_size=4000,
        max_path_length=4000,
        n_itr=500,
        discount=0.99,
        step_size=0.01,
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        # plot=True,
    )

    sh.algor.start_worker()
    sh.algor.init_opt()

def reset():
    sh.agent.reset()

    if not sh.observations is None:
        path = dict(
            observations=tensor_utils.stack_tensor_list(sh.observations),
            actions=tensor_utils.stack_tensor_list(sh.actions),
            rewards=tensor_utils.stack_tensor_list(sh.rewards),
            agent_infos=tensor_utils.stack_tensor_dict_list(sh.agent_infos),
            env_infos=tensor_utils.stack_tensor_dict_list(sh.env_infos),
        )

        sh.paths.append(path)
        sh.count += len(sh.observations)

        # check if it is time to update
        if sh.count > sh.algor.batch_size:
            itr = sh.itera
            with logger.prefix('itr #%d | ' % itr):
                paths = sh.paths
                samples_data = sh.algor.sampler.process_samples(itr, paths)
                sh.algor.log_diagnostics(paths)
                sh.algor.optimize_policy(itr, samples_data)
                logger.log("saving snapshot...")
                params = sh.algor.get_itr_snapshot(itr, samples_data)
                sh.algor.current_itr = itr + 1
                params["algo"] = sh.algor
                if sh.algor.store_paths:
                    params["paths"] = samples_data["paths"]
                logger.save_itr_params(itr, params)
                logger.log("saved")
                logger.dump_tabular(with_prefix=False)

                sh.paths = []

                if sh.algor.plot:
                    sh.algor.update_plot()
                    if sh.algor.pause_for_plot:
                        raw_input("Plotting evaluation run: Press Enter to "
                                  "continue...")



            sh.itera += 1
            sh.count = 0

    # reset arrays
    sh.observations, sh.actions, sh.rewards, sh.agent_infos, sh.env_infos = [], [], [], [], []

def get_action(o):
    a, agent_info = sh.agent.get_action(o)

    sh.observations.append(sh.env.observation_space.flatten(o))

    sh.actions.append(sh.env.action_space.flatten(a))
    sh.agent_infos.append(agent_info)
    sh.env_infos.append({})

    return a

def reward(r):
    sh.rewards.append(r)