import numpy as np
from PIL import Image, ImageDraw

import lib.eos as eos
from lib.eos import EyeOnStickEnv

SCREEN_SIZE = (224, 224)
SCREEN_CENTER = (112, 220)
SCREEN_SCALE = 35
BG_COLOR = (0, 0, 0)
BORDER_COLOR = (0, 128, 0)

TARGET_BOX_COLOR = (0, 128, 0)

TARGET_CIRCLE_COLOR = (255, 0, 0)        
TARGET_CIRCLE_SIZE = 0.05

JOINT_CIRCLE_COLOR = (0, 0, 255)
JOINT_CIRCLE_SIZE = 0.05

STICK_COLOR = (0, 0, 255)
STICK_LEN = 1.0
STICK_WIDTH = 0.01

AXIS_WIDTH = 0.01
AXIS_COLOR = TARGET_BOX_COLOR

def xy2pxy(x, y):
    px = int(SCREEN_CENTER[0] + x * SCREEN_SCALE)
    py = int(SCREEN_CENTER[1] - y * SCREEN_SCALE)
    return px, py

def draw_rect(draw, x1, y1, x2, y2, c):
    px1, py1 = xy2pxy(x1, y1)
    px2, py2 = xy2pxy(x2, y2)

    draw.polygon([
        (px1, py1),
        (px2, py1),
        (px2, py2),
        (px1, py2),
        (px1, py1),
    ], outline=c)

def draw_circle(draw, x, y, r, c):
    px, py = xy2pxy(x, y)
    pr = int(r * SCREEN_SCALE)
    draw.ellipse((px - pr, py - pr, px + pr, py + pr), fill=c)        

def draw_line(draw, x1, y1, x2, y2, c, w):
    px1, py1 = xy2pxy(x1, y1)
    px2, py2 = xy2pxy(x2, y2)
    pw = int(w * SCREEN_SCALE)
    draw.line((px1, py1, px2, py2), fill=c, width=pw)

def draw_text(draw, pos, txt, c=eos.TEXT_COLOR):
    draw.text(pos, txt, fill=c)
            
class EyeOnStickEnv2D(EyeOnStickEnv):
    def __init__(self, N_JOINTS, params):
        # target range depends on number of joints and number of dimensions, for 2D case:
        self.T_LOW, self.T_HIGH = 0.7, (N_JOINTS-1) + .7
        self.target_x = xxx
        
        super(EyeOnStickEnv2D, self).__init__(N_JOINTS, params)

    def set_1dof_target(self, t):
        self.target_y = t
            
    def recalc(self):            
        # -- orientation of joints and position of the endpoints - used for reward calculation and rendering, not an observatio
        angle = 0 # real cumulative angle, not an observation
        
        self.joints = np.zeros((self.N_JOINTS + 1, 2)) # real XY coordinates of the endpoints, used by render(), not an observation
        self._phi = np.zeros((self.N_JOINTS)) # this is a real (relative) angle, but it is not an observation, only a metric
        
        self.joints[0] = [eos.BASE_X, eos.BASE_Y] # joint[0] keep coordinates of the base point
        for i in range(1, self.N_JOINTS + 1):
            self._phi[i-1] = self.gearfuncs[i - 1](self.phi[i - 1]) 
            angle += self._phi[i - 1] # this is a real absolute value
            self.joints[i, 0] = self.joints[i - 1, 0] + self.stick_len * np.sin(angle)
            self.joints[i, 1] = self.joints[i - 1, 1] + self.stick_len * np.cos(angle)
        
        # -- eye position and orientation - used for reward calculation and rendering, not an observation
        self.eye_x = self.joints[-1][0] # real XY coordinates of the eye
        self.eye_y = self.joints[-1][1]        
        self.eye_level = angle - np.pi/2 # cumulative angle of the last joint becomes eye orientation
        
        # -- alpha and dalpha - an observation, also used for reward calculation
        dx = self.target_x - self.eye_x
        dy = self.target_y - self.eye_y
        
        if self.alpha is not None:
            prev_alpha = self.alpha
        else:
            prev_alpha = None
        self.alpha = np.arctan2(dx, dy) - self.eye_phi
        if prev_alpha is not None:
            self.dalpha = self.alpha - prev_alpha
        else:
            self.dalpha = 0

    def render(self, mode='rgb_array'):
        image = Image.new('RGB', SCREEN_SIZE, BG_COLOR)
        draw = ImageDraw.Draw(image)

        # draw a line around the border of the screen
        draw.polygon([(0, 0), (0, SCREEN_SIZE[1]-1), (SCREEN_SIZE[0]-1, SCREEN_SIZE[1]-1), (SCREEN_SIZE[0]-1, 0), (0, 0)], outline=BORDER_COLOR) # FIXME
            
        if False:
            # draw annotated x and y axes
            draw_text(draw, xy2pxy(0, 0), "0", c=AXIS_COLOR)
            draw_line(draw, 0, 0, 1, 0, c=AXIS_COLOR, w=AXIS_WIDTH)
            draw_text(draw, xy2pxy(1, 0), "x", c=AXIS_COLOR)
            draw_line(draw, 0, 0, 0, 1, c=AXIS_COLOR, w=AXIS_WIDTH)
            draw_text(draw, xy2pxy(0, 1), "y", c=AXIS_COLOR)
        
        # draw rectangular area where target can appear
        draw_rect(draw, eos.X_LOW, self.Y_LOW, eos.X_HIGH, self.Y_HIGH, c=TARGET_BOX_COLOR)
                
        # draw the sticks-and-circles robot, based on joint coordinates stashed by recalc()
        x1 = self.joints[0, 0]
        y1 = self.joints[0, 1]
        draw_circle(draw, x1, y1, JOINT_CIRCLE_SIZE, JOINT_CIRCLE_COLOR)
        
        try:
            for i in range(1, self.N_JOINTS+1):
                x2 = self.joints[i, 0]
                y2 = self.joints[i, 1]

                draw_line(draw, x1, y1, x2, y2, STICK_COLOR, STICK_WIDTH)
                draw_circle(draw, x1, y1, JOINT_CIRCLE_SIZE, JOINT_CIRCLE_COLOR)

                x1 = x2
                y1 = y2

            # draw eye extension
            ex1 = self.eye_x + self.stick_len/2 * np.sin(self.eye_phi)
            ey1 = self.eye_y + self.stick_len/2 * np.cos(self.eye_phi)
            ex2 = self.eye_x + self.stick_len/2 * np.sin(self.eye_phi + self.alpha)
            ey2 = self.eye_y + self.stick_len/2 * np.cos(self.eye_phi + self.alpha)
            draw_line(draw, self.eye_x, self.eye_y, ex1, ey1, AXIS_COLOR, STICK_WIDTH)
            draw_line(draw, self.eye_x, self.eye_y, ex2, ey2, AXIS_COLOR, STICK_WIDTH)
            draw_line(draw, ex1, ey1, ex2, ey2, AXIS_COLOR, STICK_WIDTH)            
        except ValueError as err:
            print("#ValueError joints=%s, phi=%s, last_action=%s" % (self.joints, self.phi, self.info['last_actions']))
            raise err

        # draw the target as a red ball
        draw_circle(draw, self.target_x, self.target_y, TARGET_CIRCLE_SIZE, TARGET_CIRCLE_COLOR)
        
        img_array = np.asarray(image)
        return img_array

