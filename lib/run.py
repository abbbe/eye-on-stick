import numpy as np

from lib.viz import showarray

from scipy.stats import t as stats_t

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