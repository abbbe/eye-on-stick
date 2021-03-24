
class EyeOnStickEnv3D(EyeOnStickEnv):
    def __init__(self, N_JOINTS, params):
        assert self.N_JOINTS % 2 == 0
        
        # Y limits depend on number of joints
        self.Y_LOW, self.Y_HIGH = 0.7, (self.N_JOINTS/2-2) + .7
        
#    def reset(self):
#        super(EyeOnStickEnv2D, self).reset()
        
