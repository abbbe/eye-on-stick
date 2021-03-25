#!/usr/bin/env python3

import pybullet as p
import pybullet_data

import os
import time
import numpy as np
import cv2

import tempfile

#BASEDIR = os.path.dirname(__file__)

from lib.pyb.urdf_printer import URDFPrinter

# W, H, FOVX
#CAMERA_SPECS = (1920, 1080, 64) # https://www.chiefdelphi.com/t/horizontal-fov-of-microsoft-lifecam-cinema/156204/7
CAMERA_SPECS = (224, 224, 45) # https://www.chiefdelphi.com/t/horizontal-fov-of-microsoft-lifecam-cinema/156204/7

class Manipulator:    
    STYLES = [
        {'name': 'wire0',    'plate_radius': 0.0025, 'plate_length': 0.05, 'plate0_color': 'Red', 'plate_color': 'Black', 'block1_color': 'Transparent', 'block2_color': 'Transparent', 'camera_color': 'Transparent'},
        {'name': 'wire',     'plate_radius': 0.01, 'plate_length': 0.056, 'plate0_color': 'Black', 'plate_color': 'Black', 'block1_color': 'Transparent', 'block2_color': 'Transparent', 'camera_color': 'Transparent'},
        {'name': 'original', 'plate_radius': 0.1, 'plate_length': 0.01,   'plate0_color': 'Black', 'plate_color': 'Black', 'block1_color': 'Transparent', 'block2_color': 'Transparent', 'camera_color': 'Black' },
        {'name': 'fat',      'plate_radius': 0.05 , 'plate_length': 2*0.028, 'plate0_color': 'Black', 'plate_color': 'Black', 'block1_color': 'Transparent', 'block2_color': 'Transparent', 'camera_color': 'Transparent'}
    ]
    
    def __init__(self, w, NS, NP, NA, style):
        self.w = w
        self.NS = NS
        self.NP = NP
        self.NA = NP
        self.style = style
        
        urdf_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
        URDFPrinter().print_manipulator(urdf_file, self.NS, self.NP, self.style, scale=1.0/0.028/2)
        urdf_file.close()
        
        self.body_id = self.w.loadBody(urdf_file.name)
        os.remove(urdf_file.name)
        
        assert(p.getNumJoints(self.body_id) == self.NS * self.NP * 3 + 1)
        self.eye_link_id = p.getNumJoints(self.body_id) - 1

    def _setJointMotorPosition(self, joint, pos):
        p.resetJointState(self.body_id, joint, pos)
        #print(joint, pos)

    def _setJointPosition(self, section, pos0, pos1):
        j = (section * self.NP) * 3
        
        pos0 /= self.NA # spread along several axes
        pos1 /= self.NA
        
        assert self.NP % self.NA == 0
        k = int(self.NP / self.NA)
        
        for _ in range(self.NA):
            self._setJointMotorPosition(j, pos0)
            self._setJointMotorPosition(j + 1, pos1)
            j += 2 * k

    def step(self, phis):
        for i in range(self.NS):
            self._setJointPosition(i, phis[i, 0], phis[i, 1])

    def _print_joints_pos(self, body_id=None):
        if body_id is None:
            body_id = self.body_id
            
        for i in range(p.getNumJoints(body_id)):
            js = p.getJointState(body_id, i)
            pos, orn, _, _, _, _ = p.getLinkState(body_id, i)

            rot_matrix = p.getMatrixFromQuaternion(orn)
            rot_matrix = np.array(rot_matrix).reshape(3, 3)
            v = rot_matrix.dot((0, 0, 1))

            print("#J%d %f" % (i, js[0]))
            print("#B%d %s %s" % (i, pos, v))

    def close(self):
        if self.body_id is not None:
            p.removeBody(self.body_id)
            self.body_id = None

# --------------------------------------------------------------------

class Camera(object):
    def __init__(self, w, specs=CAMERA_SPECS):            
        self.w = w
        self.W, self.H, self.FOVX = specs
        
        aspect = self.W / self.H
        self.projection_matrix = p.computeProjectionMatrixFOV(self.FOVX/aspect, aspect, 0.1, 15)
    
    def getImages(self, pvu):
        (cam_p, camera_vector, up_vector) = pvu

        view_matrix = p.computeViewMatrix(cam_p, cam_p + 0.1 * camera_vector, up_vector)
        imgs = p.getCameraImage(self.W, self.H, view_matrix, self.projection_matrix)
        assert((self.W, self.H) == (imgs[0], imgs[1]))

        return imgs

    def getRGBAImagePVU(self, pvu):
        imgs = self.getImages(pvu)
        rgba = np.reshape(imgs[2], (self.H, self.W, 4)).astype(np.float32)
        return rgba

#    def getBGRImage(self, pvu):
#        imgs = self.getCameraImages(pvu)
#        rgba = np.reshape(imgs[2], (self.H, self.W, 4)).astype(np.uint8)
#        bgr = cv2.merge((rgba[:,:,2], rgba[:,:,1], rgba[:,:,0])) # take BGR from RBGA
#        return bgr

    def getPO(self):
        raise NotImplemented
        
    def getPVU(self):
        cam_p, cam_o = self.getPO()
        cam_v, cam_u = self.w.orn2vu(cam_o)
        return [cam_p, cam_v, cam_u]

    def getRGBAImage(self):
        pvu = self.getPVU()
        img = self.getRGBAImagePVU(pvu)
        return img
    
    def close(self):
        pass
    
class FixedCamera(Camera):
    def __init__(self, w, pvu):        
        super(FixedCamera, self).__init__(w)
        self.pvu = pvu

    def getPVU(self):
        return self.pvu

    
class LinkedCamera(Camera):
    def __init__(self, w, body_id, link_id):
        super(LinkedCamera, self).__init__(w)
        self.body_id = body_id
        self.link_id = link_id
    
    def getPO(self):
        cam_p, cam_o, _, _, _, _ = p.getLinkState(self.body_id, self.link_id)
        return list(cam_p), list(cam_o)
    
# --------------------------------------------------------------------

class World(object):
    def __init__(self, gui=False):
        try:
            if gui:
                p.connect(p.GUI)
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
                #p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
                #p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
                #p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)

                #p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=-90, cameraPitch=-10, cameraTargetPosition=[0, 0, 1])
            else:
                p.connect(p.DIRECT) # don't render

            # load urdf file path (to load 'plane.urdf' from)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())

            self.reset()
        except:
            p.disconnect()
            raise

    def loadBody(self, file_name, startPos=[0, 0, 0], startOrientationEuler=[0, 0, 0]):
        startOrientation = p.getQuaternionFromEuler(startOrientationEuler)
        bodyId = p.loadURDF(file_name, startPos, startOrientation, useFixedBase=1)
        return bodyId

    def reset(self):
        p.resetSimulation()
        self.loadBody("plane.urdf", [0, 0, 0], [0, 0, 0])
        
        #self._loadBody("chessboard-%s.urdf" % self.chessboard,
        #    #[1, 0, 1], [np.pi/2, -np.pi/2, -np.pi/2])
        #    [0, 0, 3], [np.pi, 0, 0])
        #self._loadBody("urdfs/plane.urdf", [0, 0, 3], [0, np.pi, 0])
        #self._loadBody("urdfs/green-line.urdf", [1.5, 0, 0.5], [np.pi/2, 0, 0])

        self.targetId = None

    def setTarget(self, pos):
        if self.targetId is not None:
            p.removeBody(self.targetId)
            self.targetId = None

        self.targetId = self.loadBody("lib/pyb/urdfs/target.urdf", pos)

    def addHeadposMarker(self, pos):
        self.loadBody("urdfs/marker.urdf", pos)

# --------------------------------------------------------------------
    #def getDebugVisualizerCameraRGBAImage(self):
    #    width, height, viewMat, projMat, cameraUp, camForward, horizon, vertical, _, _, dist, camTarget = p.getDebugVisualizerCamera()
    #    width, height = 224, 224
    #    imgs = p.getCameraImage(width, height, viewMat, projMat)
    #    rgba = np.reshape(imgs[2], (height, width, 4)).astype(np.float32)
    #    return rgba

    def orn2vu(self, cam_o):
        rot_matrix = p.getMatrixFromQuaternion(cam_o)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)

        # Initial vectors
        init_camera_vector = (0, 0, 1)  # z-axis
        init_up_vector = (0, -1, 0)  # x-axis

        # Rotated vectors
        camera_vector = rot_matrix.dot(init_camera_vector)
        up_vector = rot_matrix.dot(init_up_vector)

        return camera_vector, up_vector

    def euler2orn(self, alpha, beta, gamma):
        return list(p.getQuaternionFromEuler([alpha, beta, gamma]))

    def step(self):
        p.stepSimulation()

    def close(self):
        p.disconnect()
