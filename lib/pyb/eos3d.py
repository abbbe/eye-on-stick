import numpy as np
from scipy import ndimage

import logging
logger = logging.getLogger()

from lib.eos import EyeOnStickEnv
from lib.pyb.pybullet_robot import World, Manipulator, FixedCamera, LinkedCamera

# https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
def angle_between(v1, v2):
    def unit_vector(vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

class EyeOnStickEnv3D(EyeOnStickEnv):
    def __init__(self, N_JOINTS, params, gui=False):
        assert N_JOINTS % 2 == 0
        self.NS = int(N_JOINTS/2)
        self.NP = params.get('NP', 1)
        
        sq22 = np.sqrt(2) / 2
        Z_LOW  = 0.5 + sq22 # half meter elevation from the base + one 45 degrees + others horizontal
        Z_HIGH = 0.5 + (self.NS - 2) + sq22 # same plus all links upright but one
        
        self.T_LOW = np.array([3, -1, Z_LOW])
        self.T_HIGH = np.array([3, +1, Z_HIGH])
        
        self.T_CENTER_LOW = np.array([3, 0, Z_LOW])
        self.T_CENTER_HIGH = np.array([3, 0, Z_HIGH])

        self.w = World(gui)
        #self.side_cam = FixedCamera(self.w, np.array(((5,0,0.5), (-1,0,0), (0,0,1))))
        self.side_cam = FixedCamera(self.w, np.array(((1.5, -4, 1.5), (0, 1, 0), (0, 0, 1))))
        self.back_cam = FixedCamera(self.w, np.array(((-3, -0.1, 1.5), (1, 0, 0), (0, 0, 1))))
        self.m = Manipulator(self.w, self.NS, self.NP, style=Manipulator.STYLES[0])
        self.eye_cam = LinkedCamera(self.w, self.m.body_id, self.m.eye_link_id)

        super(EyeOnStickEnv3D, self).__init__(N_JOINTS, params)

    def set_target(self, t): # shape of t will match shapes of .T_LOW/.T_HIGH
        # invoked from reset
        super(EyeOnStickEnv3D, self).set_target(t)
        
        logger.debug(f"{self.__class__.__name__}.set_target: {t}")
        self.w.setTarget(self.target_pos)
    
    def step(self, actions):
        obs = super(EyeOnStickEnv3D, self).step(actions)
        return obs
        
    def apply_phi(self):            
        # --- FIXME REFACTOR AWAY
        if self.alpha is not None:
            prev_alpha_cm = self.alpha_cm
        else:
            prev_alpha_cm = None

        if self.gearfunc:
            self._phi = self.gearfunc(self.phi)
        else:
            self._phi = self.phi
        #----
        
        # --- move the motors
        _phi = self._phi.reshape(self.NS, 2)
        #_phi[:, 1] = 0 ## dirty hack to glue the robot to XZ plane
        self.m.step(_phi)
        
        # --- calculate the eye level
        p, v, _u = self.eye_cam.getPVU()
        
        # get v_xy - (not normalized) projection of vector v on xy plane
        if v[0] == 0.0 and v[1] == 0.0:
            # corner case - projection is a single point, but any vector on xy plane will do in this case
            v_xy = [1, 0, 0]
        else:
            # projection of v on the horizontal plane
            v_xy = [v[0], v[1], 0]
        self.eye_level = angle_between(v, v_xy)

        # --- calculate angle between the camera view vector and direction to the target, for reward calculations only
        tvec = self.target_pos - p        
        self.alpha = angle_between(v, tvec)
        #print('p=', p, 'v=', v, 'v_xy=', v_xy, 'target=', self.target_pos, 'tvec=', tvec)
        
        
        # --- calculate center mass of the target .alpha_cm and .alpha_cm_value (=1 if the target is in view, 0 otherwise)
        target_mask = self.eye_cam.getBodyMask(self.w.targetId)
        #print('target_mask', target_mask)
        if np.any(target_mask):
            target_cm = ndimage.measurements.center_of_mass(target_mask)
            #print('target_cm', target_cm)
            #logger.debug("target_cm=%s" % str(target_cm))
            self.alpha_cm = np.array([
                2 * target_cm[0] / target_mask.shape[0] - 1,
                2 * target_cm[1] / target_mask.shape[1] - 1
            ])
            self.alpha_cm_value = 1
        else:
            self.alpha_cm = np.array([0, 0])
            self.alpha_cm_value = 0
            
        #logger.debug("alpha_cm=%s" % str(self.alpha_cm))            
        #self.alpha = angle_between(v, tvec)
        
        # --- FIXME REFACTOR AWAY
        if prev_alpha_cm is not None:
            self.dalpha_cm = self.alpha_cm - prev_alpha_cm
        else:
            self.dalpha_cm = np.array([0, 0])
        #----

    def render(self, mode='rgb_array'):
        side = self.side_cam.getRGBAImage()
        eye = self.eye_cam.getRGBAImage()
        back = self.back_cam.getRGBAImage()
        #debug = self.w.getDebugVisualizerCameraRGBAImage()
        img = np.hstack((side, eye, back))[...,:-1]
        #print(f'render(): img.shape={img.shape}')
        return img

    def close(self):
        self.m.close()
        self.side_cam.close()
        self.w.close()
        