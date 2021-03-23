import numpy as np
import time
from lib.viz import showarray

from scipy.stats import t as stats_t

def run_env_nsteps(env, model, nsteps, display, wait=None):
    all_alphas, all_eye_phis, all_rewards = [], [], []

    def get_metrics():
        alpha_t = stats_t.fit(all_alphas) # (tdf, mu_t, sigma_t) 
        eyeerr_t = stats_t.fit(np.array(all_eye_phis) - np.pi/2) # (tdf, mu_t, sigma_t) 
        
        return {
            "alpha_tdf": alpha_t[0], "alpha_tmu": alpha_t[1], "alpha_tsigma": alpha_t[2],
            "eyeerr_tdf": eyeerr_t[0], "eyeerr_tmu": eyeerr_t[1], "eyeerr_tsigma": eyeerr_t[2],
            #"alpha_mean": np.mean(all_alphas), "alpha_std": np.std(all_alphas),
            #"eye_phi_mean": np.mean(all_eye_phis), "eye_phi_std": np.std(all_eye_phis),
            "reward_total": np.sum(all_rewards),
            "reward_mean": np.mean(all_rewards), "reward_std": np.std(all_rewards)
        }

    obs = env.reset()
    for _ in range(nsteps):
        if display:
            display.clear_output(wait=True)
            showarray(env.render(mode='rgb_array'))

        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(actions)

        all_alphas.append([info['alpha'] for info in infos])                
        all_eye_phis.append([info['eye_phi'] for info in infos])                
        all_rewards.append(rewards)

        if wait:
            time.sleep(wait)
            
    metrics = get_metrics()

    return metrics, dict(all_alphas=all_alphas, all_eye_phis=all_eye_phis, all_rewards=all_rewards)