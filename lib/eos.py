import numpy as np

import gym
from gym import spaces

import mlflow
from PIL import Image, ImageDraw

import matplotlib.pyplot as plt

## ----------------------------------------------------

# configure logging
import logging
logger = logging.getLogger()
fhandler = logging.FileHandler(filename='eye-on-stick-clone.log', mode='w')
formatter = logging.Formatter('%(asctime)s %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.DEBUG)

logging.getLogger('gym').setLevel(logging.DEBUG)

# suppress trash from PIL and TF
# https://github.com/camptocamp/pytest-odoo/issues/15
logging.getLogger('PIL').setLevel(logging.ERROR)

# https://github.com/hill-a/stable-baselines/issues/298
import os
# https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import warnings
# https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
import logging
tf.get_logger().setLevel(logging.ERROR)

## --- CONST --------------------------------------------

X_LOW, X_HIGH = 2, 3
Y_LOW, Y_HIGH = -2, 2
VX_LOW, VX_HIGH = -0.01, 0.01
VY_LOW, VY_HIGH = -0.05, 0.05

SCREEN_SIZE = 500
SCREEN_SCALE = SCREEN_SIZE / 7
BG_COLOR = (0, 0, 0)
BORDER_COLOR = (0, 128, 0)

TEXT_COLOR = (0, 128, 0)
LINE_HEIGHT = 15

TARGET_CIRCLE_COLOR = (255, 0, 0)        
TARGET_CIRCLE_SIZE = 0.05

JOINT_CIRCLE_COLOR = (0, 0, 255)
JOINT_CIRCLE_SIZE = 0.05

STICK_COLOR = (0, 0, 255)
STICK_LEN = 1.0
STICK_WIDTH = 0.01

BASE_X = -2
BASE_Y = 0

PHI_AMP = (np.pi/190) * 45 # angle: up to 45 degrees in total
DPHI_AMP = (np.pi/180) * 5 # angular speed: up to 5 degrees per step

N_ERAS = 150 # eras 
N_STEPS = 1000 # steps each
N_ENVS = 32
N_LEARN_EPOCHS = 2000

class EyeOnStickEnv(gym.Env):    
    metadata = {'render.modes': ['rgb_array']}
    
    ACC_PLUS = 2
    ACC_ZERO = 1
    ACC_MINUS = 0
    
    def __init__(self, N_JOINTS, N_SEGS):
        super(EyeOnStickEnv, self).__init__()
        self.stick_len = 1.0
        
        self.N_JOINTS = N_JOINTS
        self.N_SEGS = N_SEGS

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.N_JOINTS,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3 + self.N_JOINTS,), dtype=np.float32)
        
        self.nresets = 0
        self.nsteps = 0
        
        self.reset(reset_pose=True)
    
    def reset_pose(self):
        # the stick is randomly oriented, but stationary
        self.phi = np.zeros((self.N_JOINTS)) # np.random.uniform(low=-np.pi/2, high=np.pi/2)
        self.dphi = np.zeros((self.N_JOINTS))
        
        self._recalc()
    
    def reset(self, reset_pose=False):
        self.nresets += 1
        self.nsteps = 0

        self.last_actions = []
        self.actions_log = ""
        self.info = dict(info='')

        # set random target location
        self.target_x = np.random.uniform(low=X_LOW, high=X_HIGH)
        self.target_y = np.random.uniform(low=Y_LOW, high=Y_HIGH)
        self.target_vx = np.random.uniform(low=VX_LOW, high=VX_HIGH)
        self.target_vy = np.random.uniform(low=VY_LOW, high=VY_HIGH)
        self.phi_k = 1 # np.random.uniform(low=0.75, high=1.25) * np.random.choice([-1, 1])
        
        self.joints = np.zeros((self.N_JOINTS + 1, 2))
        self.joints[0] = [BASE_X, BASE_Y]

        if reset_pose:
            self.reset_pose()
            # _recalc() is done by reset_pose()
        else:
            self._recalc()
        
        return self.get_obs()
    
    def _recalc(self):    
        angle = self.phi[0]
        for i in range(1, self.N_JOINTS + 1):
            angle += self.phi[i - 1]
            self.joints[i, 0] = self.joints[i - 1, 0] + self.stick_len * np.cos(angle)
            self.joints[i, 1] = self.joints[i - 1, 1] + self.stick_len * np.sin(angle)

        self.eye_x = self.joints[-1][0]
        self.eye_y = self.joints[-1][1]        
        self.eye_phi = angle
        
        dx = self.target_x - self.eye_x
        dy = self.target_y - self.eye_y
        self.alpha = np.arctan2(dy, dx) - self.eye_phi
              
    def get_obs(self):
        alpha = np.array([np.sin(self.alpha), np.cos(self.alpha), self.alpha / PHI_AMP])
        dphis = self.dphi / DPHI_AMP
        return np.hstack((alpha, dphis)).astype(np.float32)
    
    def step(self, actions):
        actions = np.array(actions)
        assert(actions.shape == self.dphi.shape)
        
        self.nsteps += 1
        
        # target moves
        self.target_x += self.target_vx
        self.target_y += self.target_vy
        
        # episode over if target goes out of range
        done = bool(self.target_x < X_LOW or self.target_x > X_HIGH or self.target_y < Y_LOW or self.target_y > Y_HIGH)
        
        # eos moves
        for i in range(actions.shape[0]):
            if actions[i] > 0:
                action_char = '+'
            elif actions[i] < 0:
                action_char = '-'
            else:
                action_char = 'o'
            self.actions_log += action_char
            
            self.dphi[i] += actions[i] * DPHI_AMP / 10 # ***FIXME***

            self.dphi[self.dphi > DPHI_AMP] = DPHI_AMP
            self.dphi[self.dphi < -DPHI_AMP] = -DPHI_AMP

            self.phi += self.dphi
            
            phis_above_max = self.phi > PHI_AMP
            phis_below_min = self.phi < -PHI_AMP
            self.phi[phis_above_max] = PHI_AMP
            self.phi[phis_below_min] = -PHI_AMP
            self.dphi[phis_above_max] = self.dphi[phis_below_min] = 0

        self.actions_log += ' '
        if self.nsteps % 27 == 0:
            self.actions_log += '\n'
            
        self._recalc()
                
        reward_aim = - np.log(np.abs(self.alpha)) - 1 # goes below zero somewhere between 10 and 30 degrees
        reward_action = - np.sum(np.sqrt(np.abs(actions)))

        reward = reward_aim + reward_action
            
        self.last_actions = actions
        self.info = dict(alpha=self.alpha, info=f"reward={reward:6.4f}(aim={reward_aim:6.4f}, action={reward_action:6.4f}), alpha={self.alpha:6.4f}")
        return self.get_obs(), reward, done, self.info


    def render(self, mode='rgb_array'):
        if mode != 'rgb_array':
            raise NotImplementedError()

        image = Image.new('RGB', (SCREEN_SIZE, SCREEN_SIZE), BG_COLOR)
        draw = ImageDraw.Draw(image)
        draw.polygon([
            (0, 0),
            (0, SCREEN_SIZE-1),
            (SCREEN_SIZE-1, SCREEN_SIZE-1),
            (SCREEN_SIZE-1, 0),
            (0, 0)
        ], outline=BORDER_COLOR)
            
        def draw_circle(x, y, r, fill):
            px = int(SCREEN_SIZE / 2 + x * SCREEN_SCALE)
            py = int(SCREEN_SIZE / 2 + y * SCREEN_SCALE)
            pr = int(r * SCREEN_SCALE)
            draw.ellipse((px - pr, py - pr, px + pr, py + pr), fill=fill)        

        def draw_line(x1, y1, x2, y2, fill, w):
            px1 = int(SCREEN_SIZE / 2 + x1 * SCREEN_SCALE)
            py1 = int(SCREEN_SIZE / 2 + y1 * SCREEN_SCALE)
            px2 = int(SCREEN_SIZE / 2 + x2 * SCREEN_SCALE)
            py2 = int(SCREEN_SIZE / 2 + y2 * SCREEN_SCALE)
            pw = int(w * SCREEN_SCALE)
            draw.line((px1, py1, px2, py2), fill=fill, width=pw)

        def draw_text(pos, txt):
            draw.text(pos, txt, fill=TEXT_COLOR)
            
        draw_text((10, LINE_HEIGHT), "round %5d, step %5d, last_actions %s" % (self.nresets, self.nsteps, self.last_actions))
        draw_text((10, 2*LINE_HEIGHT), "phi %s, dphi %s, alpha %.3f" %
                  (self.phi, self.dphi, self.alpha))
        draw_text((10, 3*LINE_HEIGHT), "info %s" % (str(self.info['info'])))
        draw_text((10, 4*LINE_HEIGHT), self.actions_log)

        x1 = self.joints[0, 0]
        y1 = self.joints[0, 1]
        draw_circle(x1, y1, JOINT_CIRCLE_SIZE, JOINT_CIRCLE_COLOR)
        
        for i in range(1, self.N_JOINTS+1):
            x2 = self.joints[i, 0]
            y2 = self.joints[i, 1]
        
            draw_line(x1, y1, x2, y2, STICK_COLOR, STICK_WIDTH)
            draw_circle(x1, y1, JOINT_CIRCLE_SIZE, JOINT_CIRCLE_COLOR)
            
            x1 = x2
            y1 = y2
               
        draw_circle(self.target_x, self.target_y, TARGET_CIRCLE_SIZE, TARGET_CIRCLE_COLOR)
        
        return np.asarray(image)

    def close(self):
        pass
