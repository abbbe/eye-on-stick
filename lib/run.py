import numpy as np

from lib.viz import showarray
from IPython import display

from scipy.stats import t as stats_t

def draw_text(txt, vpos=0):
    vpos += 1
    dashboard_draw.text((10, 10*vpos), txt)
    return vpos

def default_displayfunc(img_array):
    display.clear_output(wait=True)

    if False:
        dashboard_img = Image.new('RGB', (img_array.shape[1], img_array.shape[0]))
        dashboard_draw = ImageDraw.Draw(dashboard_img)

        vpos = 0
        for key, val in metrics.items():
            vpos = draw_text(f'{key:15s} {val:+.4f}', vpos=vpos)

        dashboard_img_array = np.asarray(dashboard_img)
        img_array = np.vstack((img_array, dashboard_img_array))

    showarray(img_array)

def run_env_nsteps(env, model, nsteps, displayfunc=False):
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
        # if displayfunc is set, render the environment
        if type(displayfunc) == bool:
            if displayfunc == True:
                displayfunc = default_displayfunc
            else:
                displayfunc = None
        if displayfunc is not None:
            displayfunc(env.render(mode='rgb_array'))

        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(actions)

        all_alphas.append([info['alpha'] for info in infos])                
        all_eye_phis.append([info['eye_phi'] for info in infos])                
        all_rewards.append(rewards)
            
    metrics = get_metrics()

    return metrics, dict(all_alphas=all_alphas, all_eye_phis=all_eye_phis, all_rewards=all_rewards)