import numpy as np

from lib.viz import showarray

def run_env_nsteps(env, model, nsteps, display):
    all_alphas, all_eye_phis, all_rewards = [], [], []

    def get_metrics():
        return {
            "alpha_mean": np.mean(all_alphas), "alpha_std": np.std(all_alphas),
             "eye_phi_mean": np.mean(all_eye_phis), "eye_phi_std": np.std(all_eye_phis),
             "reward_total": np.sum(all_rewards), "reward_mean": np.mean(all_rewards),
             "reward_std": np.std(all_rewards)
        }

    obs = env.reset()
    for _ in range(nsteps):
        if display:
            display.clear_output(wait=True)
            showarray(env.render(mode='rgb_array'))
        #import time
        #time.sleep(.05)

        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(actions)

        all_alphas.append([info['alpha'] for info in infos])                
        all_eye_phis.append([info['eye_phi'] for info in infos])                
        all_rewards.append(rewards)

    metrics = get_metrics()

    return metrics