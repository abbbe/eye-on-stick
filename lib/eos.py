import numpy as np

import gym
from gym import spaces
from stable_baselines import PPO2, ACKTR

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

CIRCLE_SIZE = 0.05
TARGET_CIRCLE_COLOR = (255, 0, 0)        
EYE_CIRCLE_COLOR = (0, 0, 255)
BASE_CIRCLE_COLOR = (0, 0, 255)

STICK_LEN = 1.0
STICK_WIDTH = 0.01
STICK_COLOR = (0, 0, 255)

PHI_AMP = np.pi/2
DPHI_AMP = 10
DPHI = np.pi/360

N_GOALS = 1000 # FIXME
REWARD_AIM_CLIP = 10
N_STEPS = 100

N_ENVS = 32
N_LEARN_EPOCHS = 2000 * N_ENVS

#policy=ACKTR

POLICY_CLASS = PPO2
MODEL_NAME = 'MlpLnLstmPolicy'



class EyeOnStickEnv(gym.Env):    
    metadata = {'render.modes': ['rgb_array']}
    
    ACC_PLUS = 2
    ACC_ZERO = 1
    ACC_MINUS = 0
    
    def __init__(self):
        super(EyeOnStickEnv, self).__init__()
        self.base_x = 0
        self.base_y = 0
        self.stick_len = 1.0

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        
        self.nresets = 0
        self.nsteps = 0
        
        self.reset(reset_pose=True)
    
    def reset_pose(self):
        # the stick is randomly oriented, but stationary
        self.phi = 0 # np.random.uniform(low=-np.pi/2, high=np.pi/2)
        self.phi_min = self.phi - PHI_AMP
        self.phi_max = self.phi + PHI_AMP
        self.dphi = 0
        
        self._recalc()
    
    def reset(self, reset_pose=False):
        self.nresets += 1
        self.nsteps = 0
        self.actions_log = ""
        self.info = dict()
        self.n_goals = 0

        # set random target location
        self.target_x = np.random.uniform(low=X_LOW, high=X_HIGH)
        self.target_y = np.random.uniform(low=Y_LOW, high=Y_HIGH)
        self.target_vx = np.random.uniform(low=VX_LOW, high=VX_HIGH)
        self.target_vy = np.random.uniform(low=VY_LOW, high=VY_HIGH)
        self.phi_k = 1 # np.random.uniform(low=0.75, high=1.25) * np.random.choice([-1, 1])
        
        if reset_pose:
            self.reset_pose()
            # _recalc() is done by reset_pose()
        else:
            self._recalc()
        
        return self.get_obs()
    
    def _recalc(self):    
        # eye observes target as projection on retina
        self.eye_x = self.stick_len * np.cos(self.phi)
        self.eye_y = self.stick_len * np.sin(self.phi)
        self.eye_phi = self.phi
        
        dx = self.target_x - self.eye_x
        dy = self.target_y - self.eye_y
        self.alpha = np.arctan2(dy, dx) - self.eye_phi
              
    def get_obs(self):
        return np.array([np.sin(self.alpha), np.cos(self.alpha), self.alpha / PHI_AMP, self.dphi / DPHI_AMP]).astype(np.float32)
    
    def step(self, action):
        self.nsteps += 1
        
        self.target_x += self.target_vx
        self.target_y += self.target_vy
        
        alpha_before = self.alpha

        if action == self.ACC_PLUS:
            self.dphi += 1
            action_char = '+'
        elif action == self.ACC_MINUS:
            self.dphi -= 1
            action_char = '-'
        elif action == self.ACC_ZERO:
            action_char = 'o'
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        self.actions_log += action_char
        if len(self.actions_log) % 75 == 0:
            self.actions_log += '\n'

        if self.dphi > DPHI_AMP:
            self.dphi = DPHI_AMP
        elif self.phi < -DPHI_AMP:
            self.dphi = -DPHI_AMP
            
        self.phi += self.dphi * DPHI * self.phi_k
        if self.phi > self.phi_max:
            self.phi = self.phi_max
            self.dphi = 0
        elif self.phi < self.phi_min:
            self.phi = self.phi_min
            self.dphi = 0
            
        self._recalc()
                
        reward_aim = - np.log(np.abs(self.alpha)) - 1 # goes below zero somewhere between 10 and 30 degrees
        if reward_aim > REWARD_AIM_CLIP: # 3 - clip to below one degree is perfect)
            reward_aim = REWARD_AIM_CLIP
        
        if reward_aim > 2: # around 3 degrees
            self.n_goals += 1
        else:
            self.n_goals = 0
        
        reward = reward_aim
        done = bool(self.n_goals > N_GOALS)
            
        self.info = dict(n_goals=self.n_goals, reward_aim=f"{reward_aim:.2f}", alpha=self.alpha)
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
            
        draw_text((10, LINE_HEIGHT), "round %d, step %d" % (self.nresets, self.nsteps))
        draw_text((10, 2*LINE_HEIGHT), "phi_k %.3f, phi %.3f, dphi %.3f, alpha %.3f" %
                  (self.phi_k, self.phi, self.dphi, self.alpha))
        draw_text((10, 3*LINE_HEIGHT), "info %s" % (str(self.info)))
        draw_text((10, 4*LINE_HEIGHT), self.actions_log)
            
        dx = self.stick_len * np.cos(self.eye_phi)
        dy = self.stick_len * np.sin(self.eye_phi)
        draw_circle(self.base_x, self.base_y, CIRCLE_SIZE, BASE_CIRCLE_COLOR)
        draw_line(self.base_x, self.base_y, self.base_x + dx, self.base_y + dy, STICK_COLOR, STICK_WIDTH)
        draw_circle(self.eye_x, self.eye_y, CIRCLE_SIZE, EYE_CIRCLE_COLOR)
        draw_circle(self.target_x, self.target_y, CIRCLE_SIZE, TARGET_CIRCLE_COLOR)
               
        return np.asarray(image)

    def close(self):
        pass
