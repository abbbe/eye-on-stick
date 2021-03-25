import numpy as np

import gym
from gym import spaces

from PIL import Image, ImageDraw

import matplotlib.pyplot as plt

from lib.fuzz import mk_monotonic_f

import lib.init

import logging
logger = logging.getLogger()

## --- CONST --------------------------------------------

BASE_X = -2
BASE_Y = 0

VX_LOW, VX_HIGH = 0, 0 # -0.01, 0.01
VY_LOW, VY_HIGH = 0, 0 # -0.05, 0.05

TEXT_COLOR = (0, 128, 0)
LINE_HEIGHT = 15

PHI_AMP = (np.pi/180) * 45 # angle: up to 45 degrees in total
DPHI_AMP = (np.pi/180) * 0.5 # angular speed: up to half degrees per step

def draw_text(draw, pos, txt, c=TEXT_COLOR): # FIXME - dup with eos2d
    draw.text(pos, txt, fill=c)

class EyeOnStickEnv(gym.Env):    
    metadata = {'render.modes': ['rgb_array']}
    
    def __init__(self, N_JOINTS, params):
        super(EyeOnStickEnv, self).__init__()
        self.stick_len = 1.0        
        self.N_JOINTS = N_JOINTS
        self.params = params

        # action space: positive or negative angular acceleration per joint
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.N_JOINTS,), dtype=np.float32)

        # observation space: (alpha_cm, dalpha_cm) + (((alpha, dalpha) + (phi, dpi) per joint) each is represented by sin, cos pair)
        nobs = 2 * 2 + 2 * (2 + 2 * self.N_JOINTS) 
        self.observation_space = spaces.Box(low=-1, high=1, shape=(nobs,), dtype=np.float32)
        
        self.nresets = 0
        self.nsteps = 0
                
        #logger.debug(f'{self.__class__.__name__}.__init__: NJ={N_JOINTS}, T_LOW={self.T_LOW}, T_HIGH={self.T_HIGH}')
        #self.reset()

    def set_target(self, _t):
        raise NotImplementedError()
        
    def set_random_target(self):
        #print('T_LOW=', self.T_LOW)
        #print('T_HIGH=', self.T_HIGH)
        if np.random.choice([True, False]):
            if np.random.choice([True, False]):
                t = self.T_LOW
            else:
                t = self.T_HIGH
        else:
                t = np.random.uniform(low=self.T_LOW, high=self.T_HIGH)

        #logger.debug(f"{self.__class__.__name__}.set_random_target: {t}")
        self.set_target(t)
        
    def set_zero_pose(self):
        self.phi = np.zeros((self.N_JOINTS))
        self.dphi = np.zeros((self.N_JOINTS))
        self._phi = None

    def set_random_pose(self):
        self.phi = np.random.uniform(low=-PHI_AMP, high=PHI_AMP, size=(self.N_JOINTS))
        self.dphi = np.random.uniform(low=-DPHI_AMP, high=DPHI_AMP, size=(self.N_JOINTS))        
        self._phi = None
            
    def reset(self):
        self.nresets += 1

        self.nsteps = 0
        self.actions_log = ""
        self.info = dict(info="", last_actions=[], alpha=None, eye_level=None)

        # take other set of random gearfuncs, one for each joint
        self.gearfuncs = []
        for i in range(self.N_JOINTS):
            noise = self.params.get('GEAR_FUNC_NOISE', 0)
            f = mk_monotonic_f(noise=noise, low=-PHI_AMP, high=PHI_AMP)
            #self.gearfuncs.append(lambda x: sigma2(-PHI_AMP, PHI_AMP, x, 25))
            self.gearfuncs.append(f)
        
        if 'TARGET_POS' in self.params:
            self.set_target(self.params['TARGET_POS'])
        else:
            self.set_random_target()
            
        self.set_random_pose()
        #self.set_zero_pose()
        
        self.alpha = None # to allow consistent .dalpha calculations
        self.apply_phi()
        
        return self.get_obs()

    def get_obs(self):
        # prepare normalized observations
        obs_angles = [self.alpha, self.dalpha]
        obs_angles.extend(self.phi)
        obs_angles.extend(self.dphi)
        
        # combine .alpha_cm, .dalpha_cm with sin/cos of the angles
        obss = self.alpha_cm.tolist()
        #print('obss1=', obss)
        obss.extend(self.dalpha_cm.tolist())
        #print('obss2=', obss)
        for x in obs_angles:
            obss.extend([np.cos(x), np.sin(x)])
            
        #print('obss=', obss)
        return np.array(obss).astype(np.float32)

    def step(self, actions):
        # convert to np.array and do sanity check
        actions = np.array(actions)
        assert(actions.shape == self.phi.shape)
        
        self.nsteps += 1
        
        if False:
            # the target bounces between the floor and the ceiling
            self.target_x += self.target_vx
            self.target_y += self.target_vy

            # bounce between the floor and the ceiling
            if self.target_y < self.Y_LOW:
                self.target_y = self.Y_LOW
                self.target_vy = np.abs(self.target_vy)
            elif self.target_y > self.Y_HIGH:
                self.target_y = self.Y_HIGH
                self.target_vy = - np.abs(self.target_vy)
            
        # update phis and dphis according to the given actions, impose limits        
        for i in range(actions.shape[0]):
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

            # append to the action log
            if actions[i] > 0:
                action_char = '+'
            elif actions[i] < 0:
                action_char = '-'
            else:
                action_char = 'o'
            self.actions_log += action_char

        # format action log: add space between individual moves and add new line after 27 moves
        self.actions_log += ' '
        if self.nsteps % 15 == 0:
            self.actions_log += '\n'
                
        self.apply_phi()
        # turns .phi into ._phi by applying gearfunc()
        # sets the angles of the joints to .phi
        # updates .alpha and .eye_level for rewards calculations

        # -- calculate reward based on .alpha and .eye_level
        
        # good aim is rewarded if the eye points to the target plus/minus ALPHA_MAXDIFF_GOAL
        reward_aim = bool(np.abs(self.alpha) < (np.pi/180) * self.params.get('ALPHA_MAXDIFF_GOAL'))
        
        # good eye level is rewarded if the eye level (error) is plus-minus EYE_LEVEL_MAXDIFF_GOAL
        if self.params.get('EYE_LEVEL_MAXDIFF_GOAL', None):
            # reward if eye phi is close enough to the goal value
            reward_level = bool(np.abs(self.eye_level) < (np.pi/180) * self.params.get('EYE_LEVEL_MAXDIFF_GOAL'))
        else:
            # this parameter is unset - tolerate any eye lelev
            reward_level = True
        reward_action = 0 # TODO - np.sum(np.square(self.dphi)) # self.params.get('REWARD_ACTION_WEIGHT', 1) * 

        # force end of the episode after so many steps
        done = self.nsteps > self.params.get('MAX_NSTEPS')
        # final reward - if and only if both level and aim are decent
        if reward_level and reward_aim:
            reward = 1
        else:
            reward = 0

        # stash data for metrics and monitoring
        self.info = dict(
            alpha=self.alpha, eye_level=self.eye_level,
            last_actions=actions, 
            info=f"done={done}, reward={reward:7.4f} (aim={reward_aim}, level={reward_level}, action={reward_action})",
            traj=np.vstack((self.phi, self._phi, self.dphi))) # (3, NJ)
        return self.get_obs(), reward, done, self.info

    def set_render_info(self, info):
        self.render_info = info

    def apply_phi(self):
        raise NotImplementedError()
    
    def render(self, mode='rgb_array'):
        if mode != 'rgb_array':
            raise NotImplementedError()
            
        raise NotImplementedError()

    def render_step_dashboard(self, SCREEN_SIZE=(224*2, 224), BG_COLOR=None, LINE_HEIGHT=15):    
        image = Image.new('RGB', SCREEN_SIZE, BG_COLOR)
        draw = ImageDraw.Draw(image)

        def r2d(r): return r / np.pi * 180

        with np.printoptions(precision=4, sign='+'):
            draw_text(draw, (10, LINE_HEIGHT), "nresets %5d, nsteps %3d, aplha° %7.2f, eye_level° %7.2f"
                  % (self.nresets, self.nsteps, r2d(self.alpha), r2d(self.eye_level)))
            draw_text(draw, (10, 2*LINE_HEIGHT), "alpha_cm %s" % (self.alpha_cm))
            draw_text(draw, (10, 3*LINE_HEIGHT), "last_actions %s" % (self.info['last_actions']))
            draw_text(draw, (10, 4*LINE_HEIGHT), "phi° %s" % (r2d(self.phi)))
            draw_text(draw, (10, 5*LINE_HEIGHT), "dphi° %s" % (r2d(self.dphi)))
        draw_text(draw, (10, 6*LINE_HEIGHT), "info %s" % (str(self.info['info'])))
        draw_text(draw, (10, 7*LINE_HEIGHT), (str(self.render_info)))
        draw_text(draw, (10, 8*LINE_HEIGHT), self.actions_log)

        return np.asarray(image)

    def close(self):
        pass
            