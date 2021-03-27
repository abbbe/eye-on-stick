import numpy as np
import os, json

from scipy.stats import t as stats_t

os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlruns/db.sqlite"
import mlflow
mlflow_client = mlflow.tracking.MlflowClient()

from stable_baselines import SAC
from stable_baselines.common.vec_env import VecNormalize
from stable_baselines.common.cmd_util import make_vec_env

def find_model(model_name, model_version=None):
    if model_version is not None:
        registered_model = mlflow_client.get_model_version(model_name, model_version)
    else:
        registered_model = mlflow_client.get_latest_versions(model_name, stages=["None"])[0]
    # registered_model .source, .version
    
    return registered_model
    
def mk_env_agent(env_class, registered_model, params, gui=False):
    model = SAC.load(registered_model.source)

    params_fname = f'{registered_model.source}.json' # FIXME
    with open(params_fname, 'r') as fp:
        loaded_params = json.load(fp)

    params = {**loaded_params, **params} # merge, overriding loaded params 
    env = make_vec_env(lambda: env_class(params['NJ'], params, gui=gui), n_envs=1)
    
    model.set_env(env)
    env.env_method('set_render_info', {'name': registered_model.name, 'version': registered_model.version}) # FIXME
    
    return env, model

"""
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
               
    #metrics = get_metrics()
    #№return metrics, dict(all_alphas=all_alphas, all_eye_levels=all_eye_levels, all_rewards=all_rewards)
"""

def nsteps_env_agent(env, model, nsteps, displayfunc=None):
    all_obs = []
    all_rewards = []
    all_dones = None
    all_infos = []
    all_imgs = []
    
    def stash_imgs():
        if displayfunc is not None or all_imgs is not None:
            imgs_array = env.render(mode='rgb_array')
            #print('imgs_array.shape=', imgs_array.shape)
            if displayfunc is not None:
                displayfunc(imgs_array)
            if all_imgs is not None:
                all_imgs.append(imgs_array)

        if all_infos is not None:
            all_infos.extend(infos)
            
        #print('all_imgs.shape=', all_imgs.shape)
    
    obs = env.reset()
    #stash_imgs() FIXME we skip first image, but maybe we should drop last one instead? or let it be +1
    
    for _ in range(nsteps):
        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(actions)

        stash_imgs()

        if all_obs is not None:
            all_obs.extend(obs)

        if all_rewards is not None:
            all_rewards.extend(rewards)

        if all_dones is not None:
            all_dones.extend(dones)

        if all_infos is not None:
            all_infos.extend(infos)
            
    return all_obs, all_rewards, all_dones, all_infos, all_imgs
