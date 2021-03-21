import numpy as np

import gym
from gym import spaces

import mlflow
from PIL import Image, ImageDraw

import matplotlib.pyplot as plt

from lib.fuzz import mk_monotonic_f

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

BASE_X = -2
BASE_Y = 0

X_LOW, X_HIGH = 2, 3
Y_LOW, Y_HIGH = 0, 3
VX_LOW, VX_HIGH = 0, 0 # -0.01, 0.01
VY_LOW, VY_HIGH = -0.01, 0.01

SCREEN_SIZE = (900, 500)
SCREEN_CENTER = (450, 450)
SCREEN_SCALE = 100
BG_COLOR = (0, 0, 0)
BORDER_COLOR = (0, 128, 0)

TARGET_BOX_COLOR = (0, 128, 0)

AXIS_WIDTH = 0.01
AXIS_COLOR = TARGET_BOX_COLOR

TEXT_COLOR = (0, 128, 0)
LINE_HEIGHT = 15

TARGET_CIRCLE_COLOR = (255, 0, 0)        
TARGET_CIRCLE_SIZE = 0.05

JOINT_CIRCLE_COLOR = (0, 0, 255)
JOINT_CIRCLE_SIZE = 0.05

STICK_COLOR = (0, 0, 255)
STICK_LEN = 1.0
STICK_WIDTH = 0.01

PHI_AMP = (np.pi/180) * 45 # angle: up to 45 degrees in total
DPHI_AMP = (np.pi/180) * 0.5 # angular speed: up to 1 degrees per step

# environment runs trying to catch the target in the very center of the eye view
# if catches (alpha < ALPHA_GOAL) for N_GOALS steps, it gets a reward of 10 and make target jump to another random location
# otherwise reward is proportional to the eye view angle on the target

N_GOALS = 5
ALPHA_MAXDIFF_GOAL = (np.pi/180) * 3
EYE_PHI_GOAL = np.pi/2
EYE_PHI_MAXDIFF_GOAL = ALPHA_MAXDIFF_GOAL

class EyeOnStickEnv(gym.Env):    
    metadata = {'render.modes': ['rgb_array']}
    
    ACC_PLUS = 2
    ACC_ZERO = 1
    ACC_MINUS = 0
    
    def __init__(self, N_JOINTS, params):
        super(EyeOnStickEnv, self).__init__()
        self.stick_len = 1.0
        
        self.N_JOINTS = N_JOINTS
        self.params = params

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.N_JOINTS,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3 + 2*self.N_JOINTS,), dtype=np.float32)
        
        self.nresets = 0
        self.nsteps = 0
                
        self.reset()
    
    #def reset_pose(self):
    #    # the stick is randomly oriented, but stationary
    #    self.phi = np.zeros((self.N_JOINTS)) # np.random.uniform(low=-np.pi/2, high=np.pi/2)
    #    self.dphi = np.zeros((self.N_JOINTS))
    #    self._recalc()
    
    def set_random_target(self, recalc=True):
        self.target_x = np.random.uniform(low=X_LOW, high=X_HIGH)
        self.target_y = np.random.uniform(low=Y_LOW, high=Y_HIGH)
        self.target_vx = np.random.uniform(low=VX_LOW, high=VX_HIGH)
        self.target_vy = np.random.uniform(low=VY_LOW, high=VY_HIGH)
        if recalc: self._recalc()

    def set_zero_pose(self, recalc=True):
        self.phi = np.zeros((self.N_JOINTS))
        self.dphi = np.zeros((self.N_JOINTS))
        if recalc: self._recalc()

    def set_random_pose(self, recalc=True):
        self.phi = np.random.uniform(low=-PHI_AMP, high=PHI_AMP, size=(self.N_JOINTS))
        self.dphi = np.random.uniform(low=-DPHI_AMP, high=DPHI_AMP, size=(self.N_JOINTS))        
        if recalc: self._recalc()
            
    def reset(self):
        self.nresets += 1
        self.nsteps = 0

        # initialize to something
        self.joints = np.zeros((self.N_JOINTS + 1, 2))

        self.ngoals = 0
        self.actions_log = ""
        self.info = dict(info="", last_actions=[], alpha=None, eye_phi=None)

        self.gearfuncs = []
        for i in range(self.N_JOINTS):
            f = mk_monotonic_f(
                noise=self.params.get('GEAR_FUNC_NOISE', 0),
                low=-PHI_AMP, high=PHI_AMP)
            self.gearfuncs.append(f)
        
        self.set_random_target(recalc=False)
        #self.set_random_pose(recalc=False)
        self.set_zero_pose(recalc=False)
        self._recalc()
        
        return self.get_obs()

    def _recalc(self):    
        angle = 0
        self.joints[0] = [BASE_X, BASE_Y]
        
        for i in range(1, self.N_JOINTS + 1):
            #angle += self.gearfuncs[i - 1](self.phi[i - 1])
            angle += self.phi[i - 1]
            self.joints[i, 0] = self.joints[i - 1, 0] + self.stick_len * np.sin(angle)
            self.joints[i, 1] = self.joints[i - 1, 1] + self.stick_len * np.cos(angle)

        self.eye_x = self.joints[-1][0]
        self.eye_y = self.joints[-1][1]        
        self.eye_phi = angle
        
        dx = self.target_x - self.eye_x
        dy = self.target_y - self.eye_y
        self.alpha = np.arctan2(dx, dy) - self.eye_phi
              
    def get_obs(self):
        # prepare normalized observations
        alpha = np.array([np.sin(self.alpha), np.cos(self.alpha), self.alpha / PHI_AMP])
        dphis = self.dphi / DPHI_AMP
        phis = self.phi / PHI_AMP
        return np.hstack((alpha, dphis, phis)).astype(np.float32)
    
    def step(self, actions):
        actions = np.array(actions)
        assert(actions.shape == self.dphi.shape)
        
        self.nsteps += 1
        
        # target moves
        self.target_x += self.target_vx
        self.target_y += self.target_vy
        
        # episode over if target goes out of range
        #done = bool(self.target_x < X_LOW or self.target_x > X_HIGH or self.target_y < Y_LOW or self.target_y > Y_HIGH)
        
        # eos moves
        for i in range(actions.shape[0]):
            # construct action log
            if actions[i] > 0:
                action_char = '+'
            elif actions[i] < 0:
                action_char = '-'
            else:
                action_char = 'o'
            self.actions_log += action_char
            
            # increase angular velocity according to acceleration action
            self.dphi[i] += actions[i] * DPHI_AMP / 10 # ***FIXME***
            # keep angular velocity within [-DPHI_AMP, DPHI_AMP] interval
            self.dphi[self.dphi > DPHI_AMP] = DPHI_AMP
            self.dphi[self.dphi < -DPHI_AMP] = -DPHI_AMP

            # increase angle according to the angular velocity
            self.phi += self.dphi
            # keep angle within [-PHI_AMP, PHI_AMP] interval
            phis_above_max = self.phi > PHI_AMP
            phis_below_min = self.phi < -PHI_AMP
            self.phi[phis_above_max] = PHI_AMP
            self.phi[phis_below_min] = -PHI_AMP
            # zero out angular velocity where limits were hit
            self.dphi[phis_above_max] = 0
            self.dphi[phis_below_min] = 0

        # format action log: add space between individual moves and add new line after 27 moves
        self.actions_log += ' '
        if self.nsteps % 27 == 0:
            self.actions_log += '\n'
        
        self._recalc()
        
        #reward_aim = np.square((np.pi - np.abs(self.alpha)) / np.pi) * self.params.get('REWARD_AIM_WEIGHT', 1)
        reward_aim = 0 # - np.log(np.abs(self.alpha)) # * self.params.get('REWARD_AIM_WEIGHT', 1)
        reward_level = 0 # - np.tanh(np.abs(self.eye_phi - EYE_PHI_GOAL)) # keep head aligned with x-axis #  self.params.get('REWARD_LEVEL_WEIGHT', 1) * 
        reward_action = 0 # - np.sum(np.square(self.dphi)) # self.params.get('REWARD_ACTION_WEIGHT', 1) * 

        done = False
        #if (np.abs(self.alpha) < ALPHA_MAXDIFF_GOAL) and (np.abs(self.eye_phi - EYE_PHI_GOAL) < EYE_PHI_MAXDIFF_GOAL):
        if np.abs(self.alpha) < ALPHA_MAXDIFF_GOAL:
            # position is good
            if self.ngoals > N_GOALS:
                # caught enough goals, give reward and chase another target
                done = True
                reward_aim = 10
                self.ngoals = 0
            else:
                reward_aim = 1
                self.ngoals += 1
        else:
            # position is bad
            self.ngoals = 0
            reward_aim = 0

        # reward if both aim and level are good, add penalty for any actions (which are accelerations and deccelerations)
        reward = reward_aim # * reward_level + reward_action

        # stash data for metrics and monitoring
        self.info = dict(
            alpha=self.alpha, eye_phi=self.eye_phi,
            last_actions=actions, 
            info=f"reward={reward:7.4f} (aim={reward_aim:7.4f}, level={reward_level:7.4f}, action={reward_action:7.4f})")
        return self.get_obs(), reward, done, self.info


    def render(self, mode='rgb_array'):
        if mode != 'rgb_array':
            raise NotImplementedError()

        image = Image.new('RGB', SCREEN_SIZE, BG_COLOR)
        draw = ImageDraw.Draw(image)
        draw.polygon([(0, 0), (0, SCREEN_SIZE[1]-1), (SCREEN_SIZE[0]-1, SCREEN_SIZE[1]-1), (SCREEN_SIZE[0]-1, 0), (0, 0)], outline=BORDER_COLOR) # FIXME

        def xy2pxy(x, y):
            px = int(SCREEN_CENTER[0] + x * SCREEN_SCALE)
            py = int(SCREEN_CENTER[1] - y * SCREEN_SCALE)
            return px, py
        
        def draw_rect(x1, y1, x2, y2, c):
            px1, py1 = xy2pxy(x1, y1)
            px2, py2 = xy2pxy(x2, y2)
            
            draw.polygon([
                (px1, py1),
                (px2, py1),
                (px2, py2),
                (px1, py2),
                (px1, py1),
            ], outline=c)
            
        def draw_circle(x, y, r, c):
            px, py = xy2pxy(x, y)
            pr = int(r * SCREEN_SCALE)
            draw.ellipse((px - pr, py - pr, px + pr, py + pr), fill=c)        

        def draw_line(x1, y1, x2, y2, c, w):
            px1, py1 = xy2pxy(x1, y1)
            px2, py2 = xy2pxy(x2, y2)
            pw = int(w * SCREEN_SCALE)
            draw.line((px1, py1, px2, py2), fill=c, width=pw)

        def draw_text(pos, txt, c=TEXT_COLOR):
            draw.text(pos, txt, fill=c)
            
        # draw rectangular area where target can appear
        draw_rect(X_LOW, Y_LOW, X_HIGH, Y_HIGH, c=TARGET_BOX_COLOR)
        
        # draw annotated x and y axes
        draw_text(xy2pxy(0, 0), "0", c=AXIS_COLOR)
        draw_line(0, 0, 1, 0, c=AXIS_COLOR, w=AXIS_WIDTH)
        draw_text(xy2pxy(1, 0), "x", c=AXIS_COLOR)
        draw_line(0, 0, 0, 1, c=AXIS_COLOR, w=AXIS_WIDTH)
        draw_text(xy2pxy(0, 1), "y", c=AXIS_COLOR)
        
        def r2d(r): return r / np.pi * 180

        
        with np.printoptions(precision=4, sign='+'):
            draw_text((10, LINE_HEIGHT), "round %5d, step %3d, aplha° %7.2f, eye_phi° %7.2f, last_actions %s"
                  % (self.nresets, self.nsteps, r2d(self.alpha), r2d(self.eye_phi), self.info['last_actions']))
            draw_text((10, 2*LINE_HEIGHT), "phi° %s" % (r2d(self.phi)))
            draw_text((10, 3*LINE_HEIGHT), "dphi° %s" % (r2d(self.dphi)))
        draw_text((10, 4*LINE_HEIGHT), "info %s" % (str(self.info['info'])))
        draw_text((10, 5*LINE_HEIGHT), self.actions_log)

        x1 = self.joints[0, 0]
        y1 = self.joints[0, 1]
        draw_circle(x1, y1, JOINT_CIRCLE_SIZE, JOINT_CIRCLE_COLOR)
        
        try:
            for i in range(1, self.N_JOINTS+1):
                x2 = self.joints[i, 0]
                y2 = self.joints[i, 1]

                draw_line(x1, y1, x2, y2, STICK_COLOR, STICK_WIDTH)
                draw_circle(x1, y1, JOINT_CIRCLE_SIZE, JOINT_CIRCLE_COLOR)

                x1 = x2
                y1 = y2

            # draw eye extension
            ex1 = self.eye_x + self.stick_len/2 * np.sin(self.eye_phi)
            ey1 = self.eye_y + self.stick_len/2 * np.cos(self.eye_phi)
            ex2 = self.eye_x + self.stick_len/2 * np.sin(self.eye_phi + self.alpha)
            ey2 = self.eye_y + self.stick_len/2 * np.cos(self.eye_phi + self.alpha)
            draw_line(self.eye_x, self.eye_y, ex1, ey1, AXIS_COLOR, STICK_WIDTH)
            draw_line(self.eye_x, self.eye_y, ex2, ey2, AXIS_COLOR, STICK_WIDTH)
            draw_line(ex1, ey1, ex2, ey2, AXIS_COLOR, STICK_WIDTH)            
        except ValueError as err:
            print("#ValueError joints=%s, phi=%s, last_action=%s" % (self.joints, self.phi, self.info['last_actions']))
            raise err
               
        draw_circle(self.target_x, self.target_y, TARGET_CIRCLE_SIZE, TARGET_CIRCLE_COLOR)
        
        return np.asarray(image)

    def close(self):
        pass
