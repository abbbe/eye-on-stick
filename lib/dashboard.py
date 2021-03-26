import io
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from IPython import display
from lib.viz import showarray

import lib.eos as eos

fig, ax = plt.subplots(figsize=(4, 4), dpi=224/4)
colors="rgbcmy"

def get_metrics_dashboard(metrics):
    dashboard_img = Image.new('RGB', (224, 224))
    dashboard_draw = ImageDraw.Draw(dashboard_img)

    def draw_text(txt, vpos=0):
        vpos += 1
        dashboard_draw.text((10, 10*vpos), txt)
        return vpos

    # draw metrics
    vpos = 0
    for key, val in metrics.items():
        vpos = draw_text(f'{key:15s} {val:+.4f}', vpos=vpos)

    return np.asarray(dashboard_img)

def get_episode_dashboard1(gearfuncs, trajs):
    if gearfuncs is None:
        return np.zeros((224, 224, 3))

    ax.cla()
    ax.set_xlim([-eos.PHI_AMP, eos.PHI_AMP])
    ax.set_ylim([-eos.PHI_AMP, eos.PHI_AMP])

    gfx = np.linspace(-eos.PHI_AMP, eos.PHI_AMP, 50)
    for i, gf in enumerate(gearfuncs):
        gfy = gf(gfx)
        ax.scatter(gfx, gfy, 1, colors[i])

    for i in range(trajs.shape[2]):
        ax.scatter(trajs[:,0,i], trajs[:,1,i], 1, colors[i])

    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw')
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                         newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()

    return img_arr[:,:,[0,1,2]] # take only RGB from RGBA

def get_episode_dashboard2(trajs):
    if trajs is None:
        return np.zeros((224, 224, 3))

    ax.cla()
    ax.autoscale(True)
    for i in range(trajs.shape[2]):
        ax.plot(range(trajs.shape[0]), trajs[:,0,i], colors[i])

    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw')
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                         newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()

    return img_arr[:,:,[0,1,2]] # take only RGB from RGBA

def get_episode_dashboard3(trajs):
    if trajs is None:
        return np.zeros((224, 224, 3))

    ax.cla()
    ax.autoscale(True)
    for i in range(trajs.shape[2]):
        ax.scatter(trajs[:,0,i], trajs[:,2,i], 1, colors[i]) # ... phi, dphi ...
    ax.axhline(y=0.0, color='grey', linestyle='-')

    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw')
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                         newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()

    return img_arr[:,:,[0,1,2]] # take only RGB from RGBA

def get_blank_dashboard():
    dashboard_img = Image.new('RGB', (224, 224))
    dashboard_draw = ImageDraw.Draw(dashboard_img)
    return np.asarray(dashboard_img)

def get_episode_dashboard(lastrun_gearfuncs, lastrun_trajs, lastrun_metrics):
    return np.hstack((
        get_episode_dashboard1(lastrun_gearfuncs, lastrun_trajs),
        get_episode_dashboard2(lastrun_trajs),
        get_episode_dashboard3(lastrun_trajs),
        get_metrics_dashboard(lastrun_metrics),
        get_blank_dashboard()
    ))

None