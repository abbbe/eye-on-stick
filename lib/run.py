import numpy as np
import os, json

from scipy.stats import t as stats_t

os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlruns/db.sqlite"
import mlflow
mlflow_client = mlflow.tracking.MlflowClient()

from stable_baselines import SAC
from stable_baselines.common.vec_env import VecNormalize
from stable_baselines.common.cmd_util import make_vec_env

from lib.viz import showarray

def mk_env_agent(env_class, model_name, params, model_version=None, gui=False):
    if model_version is not None:
        registered_model = mlflow_client.get_model_version(model_name, model_version)
    else:
        registered_model = mlflow_client.get_latest_versions(model_name, stages=["None"])[0]
    # registered_model .source, .version
    
    model = SAC.load(registered_model.source)

    params_fname = f'{registered_model.source}.json' # FIXME
    with open(params_fname, 'r') as fp:
        loaded_params = json.load(fp)

    params = {**loaded_params, **params} # merge, overriding loaded params 
    env = make_vec_env(lambda: env_class(params['NJ'], params, gui=gui), n_envs=1)
    
    model.set_env(env)
    env.env_method('set_render_info', {'name': model_name, 'version': model_version, 'real_version': registered_model.version}) # FIXME
    
    return env, model

def run_env_nsteps(env, model, nsteps, displayfunc=showarray, trajfunc=None):
    all_alphas, all_eye_levels, all_rewards = [], [], []

    def get_metrics():
        alpha_t = stats_t.fit(all_alphas) # (tdf, mu_t, sigma_t) 
        eyelevel_t = stats_t.fit(all_eye_levels) # (tdf, mu_t, sigma_t) 
        
        return {
            "alpha_tdf": alpha_t[0], "alpha_tmu": alpha_t[1], "alpha_tsigma": alpha_t[2],
            "eyelevel_tdf": eyelevel_t[0], "eyelevel_tmu": eyelevel_t[1], "eyelevel_tsigma": eyelevel_t[2],
            "reward_total": np.sum(all_rewards),
            "reward_mean": np.mean(all_rewards), "reward_std": np.std(all_rewards)
        }

    obs = env.reset()
    for _ in range(nsteps):
        if displayfunc is not None:
            displayfunc(env.render(mode='rgb_array'))

        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(actions)

        for info in infos:
            all_alphas.append(info['alpha'])
            all_eye_levels.append(info['eye_level'])
            if trajfunc is not None:
                # if trajectory callback is set, call it XXX multiple times per vector env FIXME
                trajfunc(info['traj'])
        all_rewards.append(rewards)
        
            
    metrics = get_metrics()

    return metrics, dict(all_alphas=all_alphas, all_eye_levels=all_eye_levels, all_rewards=all_rewards)