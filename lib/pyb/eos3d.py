import numpy as np

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
        
        sq22 = np.sqrt(2) / 2
        Z_LOW  = 0.5 + sq22 # half meter elevation from the base + one 45 degrees + others horizontal
        Z_HIGH = 0.5 + (self.NS - 2) + sq22 # same plus all links upright but one
        
        self.T_LOW = np.array([3, -1, Z_LOW])
        self.T_HIGH = np.array([3, +1, Z_HIGH])

        self.w = World(gui)
        #self.side_cam = FixedCamera(self.w, np.array(((5,0,0.5), (-1,0,0), (0,0,1))))
        self.side_cam = FixedCamera(self.w, np.array(((1.5, -4, 1.5), (0, 1, 0), (0, 0, 1))))
        self.m = Manipulator(self.w, self.NS, 1, 1, style=Manipulator.STYLES[0])
        self.eye_cam = LinkedCamera(self.w, self.m.body_id, self.m.eye_link_id)

        super(EyeOnStickEnv3D, self).__init__(N_JOINTS, params)
            
    def set_1dof_target(self, t):
        # invoked from reset
        self.target_pos = t
        logger.debug(f"{self.__class__.__name__}.set_1dof_target: {t}")
        self.w.setTarget(self.target_pos)
        
    def step(self, actions):
        obs = super(EyeOnStickEnv3D, self).step(actions)
        return obs
        
    def apply_phi(self):            
        # --- FIXME REFACTOR AWAY
        if self.alpha is not None:
            prev_alpha = self.alpha
        else:
            prev_alpha = None

        self._phi = np.zeros((self.N_JOINTS)) # this is a real (relative) angle, but it is not an observation, only a metric
        for i in range(self.N_JOINTS):
            self._phi[i] = self.gearfuncs[i](self.phi[i])
        #----
        
        # move the motors
        _phi = self._phi.reshape(self.NS, 2)
        #_phi[:, 1] = 0 ## dirty hack to glue the robot to XZ plane
        self.m.step(_phi)
        
        # calculate angle of view towards the target and eye level
        p, v, _u = self.eye_cam.getPVU()        
        # eye_level is an angle with the horizontal plane
        if v[0] == 0.0 and v[1] == 0.0:
            v_xy = [1, 0, 0] # any vector on xy plane will do
        else:
            v_xy = [v[0], v[1], 0] # projection of v on the horizontal plane
        tvec = self.target_pos - p
        
        #print('p=', p, 'v=', v, 'v_xy=', v_xy, 'target=', self.target_pos, 'tvec=', tvec)
        
        self.eye_level = angle_between(v, v_xy)
        self.alpha = angle_between(v, tvec)
        
        # --- FIXME REFACTOR AWAY
        if prev_alpha is not None:
            self.dalpha = self.alpha - prev_alpha
        else:
            self.dalpha = 0
        #----

    def render(self, mode='rgb_array'):
        side = self.side_cam.getRGBAImage()[...,:-1]
        eye = self.eye_cam.getRGBAImage()[...,:-1]
        return np.hstack((side, eye))


    def close(self):
        self.m.close()
        self.side_cam.close()
        self.w.close()
        