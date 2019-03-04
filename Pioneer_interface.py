import vrep
import sys
import time
import matplotlib.pyplot as plt
from math import atan2, pi, sqrt
import numpy as np
from utility import Dist


def imshow(res, img):
    img = np.uint8(img)
    img = img.reshape((res[0], res[1], 3))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.axis("off")
    plt.imshow(img)
    plt.show()


class PioneerP3DX_interface:

    def __init__(self):
        self.left_motor = "Pioneer_p3dx_leftMotor"
        self.right_motor = "Pioneer_p3dx_rightMotor"
        self.ultrasonic = "Pioneer_p3dx_ultrasonicSensor"
        self.robot = "Pioneer_p3dx"

        self.clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)  # Connect to V-REP

        if self.clientID != -1:
            print('Connected to remote API server')

        else:
            print('Remote API function call returned with error code: ')
            sys.exit("couldn't connect")

    def getOrientation(self):
        code, handler = vrep.simxGetObjectHandle(self.clientID, self.robot, vrep.simx_opmode_blocking)
        orient = vrep.simxGetObjectOrientation(self.clientID, handler, -1, vrep.simx_opmode_blocking)
        return orient[1]

    def get_right_motor(self):
        code, handler = vrep.simxGetObjectHandle(self.clientID, self.right_motor, vrep.simx_opmode_blocking)
        return code, handler

    def get_left_motor(self):
        code, handler = vrep.simxGetObjectHandle(self.clientID, self.left_motor, vrep.simx_opmode_blocking)
        return code, handler

    def get_i_ultrasonic_sensor(self, id):
        code, handler = vrep.simxGetObjectHandle(self.clientID, self.ultrasonic + str(id), vrep.simx_opmode_blocking)
        return code, handler

    def get_all_ultrasonic_sensor(self):
        sensor_handlers = []
        for i in range(16):
            sensor_handlers.append(self.get_i_ultrasonic_sensor(i + 1)[1])
        return sensor_handlers

    def read_sensor_info(self, handler, mode):
        # returnCode, detectionState, detectedPoint, detectedObjectHandle, detectedSurfaceNormalVector
        return vrep.simxReadProximitySensor(self.clientID, handler, mode)

    def getPosition(self):
        code, handler = vrep.simxGetObjectHandle(self.clientID, "Pioneer_p3dx", vrep.simx_opmode_blocking)
        return vrep.simxGetObjectPosition(self.clientID, handler, -1, vrep.simx_opmode_blocking)[1]

    def setPosition(self, pos):
        code, handler = vrep.simxGetObjectHandle(self.clientID, self.robot, vrep.simx_opmode_blocking)
        # vrep.simxSetObjectPosition(self.clientID, handler, -1, pos, vrep.simx_opmode_oneshot)
        # c, joints = vrep.simxGetObjects(self.clientID, vrep.sim_object_joint_type, vrep.simx_opmode_oneshot_wait)
        #
        # for child in joints:
        #     vrep.simxSetObjectPosition(self.clientID, child, -1, pos, vrep.simx_opmode_streaming)

        pos1 = vrep.simxGetObjectPosition(self.clientID, self.get_left_motor()[1], -1, vrep.simx_opmode_oneshot)
        pos2 = vrep.simxGetObjectPosition(self.clientID, self.get_right_motor()[1], -1, vrep.simx_opmode_oneshot)
        if pos1[0]:
            print("True")
        print(pos1, pos2)
        # vrep.simxSetObjectPosition(self.clientID, lw, -1, pos,vrep.simx_opmode_oneshot)
        # vrep.simxSetObjectPosition(self.clientID, rw, -1, pos,vrep.simx_opmode_oneshot)
        # print(vrep.simxSetJointPosition(self.clientID, self.get_left_motor()[1], pos, vrep.simx_opmode_oneshot_wait))
        # print(vrep.simxSetJointPosition(self.clientID, self.get_right_motor()[1], pos, vrep.simx_opmode_oneshot_wait))
        # vrep.simResetDynamicObject(self.clientID)

    def check_position(self, position):
        new = position
        change = False
        if position[0] > 2.1:
            new[0] = 1.9
            change = True
        if position[0] < -2.1:
            new[0] = -1.9
            change = True
        if position[1] > 2.1:
            new[1] = 1.9
            change = True

        if position[1] < -2.1:
            new[1] = -1.9
            change = True
        if change:
            print("check position")
            angle = atan2(new[1] - position[1], new[0] - position[0])  # * 180 / pi

            self.rotate_to(angle)
            self.move_strait(3)

            time.sleep(1)
            self.stop()
        else:
            return

    def getCameraHandle(self):
        return vrep.simxGetObjectHandle(self.clientID, "Vision_sensor", vrep.simx_opmode_blocking)

    def getFirstCameraImage(self):
        code, self.camera = self.getCameraHandle()
        return vrep.simxGetVisionSensorImage(self.clientID, self.camera, 0, vrep.simx_opmode_streaming)

    def streamCameraImage(self):
        self.getFirstCameraImage()

        while vrep.simxGetConnectionId(self.clientID) != -1:
            _, res, img = vrep.simxGetVisionSensorImage(self.clientID, self.camera, 0, vrep.simx_opmode_oneshot)

    def getCameraImage(self):
        self.getFirstCameraImage()
        code, camera = vrep.simxGetObjectHandle(self.clientID, "Vision_sensor", vrep.simx_opmode_blocking)
        code, res, img = vrep.simxGetVisionSensorImage(self.clientID, camera, 0, vrep.simx_opmode_streaming)
        img = np.uint8(img)
        img = img / 255.0
        img = img.reshape((res[0], res[1], 3))
        img = np.expand_dims(img, 0)
        return code, res, img

    def move_right(self, velocity=0.8):
        err_code, left_motor_handler = self.get_left_motor()
        vrep.simxSetJointTargetVelocity(self.clientID, left_motor_handler, velocity, vrep.simx_opmode_streaming)

    def move_left(self, velocity=0.8):
        err_code, right_motor_handler = self.get_right_motor()
        vrep.simxSetJointTargetVelocity(self.clientID, right_motor_handler, velocity, vrep.simx_opmode_streaming)

    def move_strait(self, velocity=1.3):
        err_code, right_motor_handler = self.get_right_motor()
        err_code, left_motor_handler = self.get_left_motor()
        vrep.simxSetJointTargetVelocity(self.clientID, right_motor_handler, velocity, vrep.simx_opmode_streaming)
        vrep.simxSetJointTargetVelocity(self.clientID, left_motor_handler, velocity, vrep.simx_opmode_streaming)

    def move(self, left_joint_velocity, right_joint_velocity):
        err_code, right_motor_handler = self.get_right_motor()
        err_code, left_motor_handler = self.get_left_motor()
        vrep.simxSetJointTargetVelocity(self.clientID, right_motor_handler, right_joint_velocity,
                                        vrep.simx_opmode_blocking)
        vrep.simxSetJointTargetVelocity(self.clientID, left_motor_handler, left_joint_velocity,
                                        vrep.simx_opmode_blocking)
        time.sleep(1)
        self.stop()

    def stop(self):
        err_code, right_motor_handler = self.get_right_motor()
        err_code, left_motor_handler = self.get_left_motor()
        vrep.simxSetJointTargetVelocity(self.clientID, right_motor_handler, 0, vrep.simx_opmode_streaming)
        vrep.simxSetJointTargetVelocity(self.clientID, left_motor_handler, 0, vrep.simx_opmode_streaming)

    def move_to(self, position):
        cur_pos = self.getPosition()
        angle = atan2(position[1] - cur_pos[1], position[0] - cur_pos[0])  # * 180 / pi

        initial_dist = Dist(cur_pos[0], position[0], cur_pos[1], position[1])
        if initial_dist <= 0.08:
            return
        self.getFirstCameraImage()

        self.rotate_to(angle)

        images = []
        cur_dist = initial_dist
        while cur_dist > 0.1:

            self.check_position(cur_pos)
            _, res, img = self.getCameraImage()
            images.append(img)
            self.move_strait(3)
            cur_pos = self.getPosition()
            cur_dist = Dist(cur_pos[0], position[0], cur_pos[1], position[1])
            if cur_dist > initial_dist:
                angle = atan2(position[1] - cur_pos[1], position[0] - cur_pos[0])  # * 180 / pi
                self.rotate_to(angle)
            initial_dist = cur_dist
            time.sleep(0.05)

        self.stop()
        print(np.array(images).shape)

    def rotate_to(self, angle):

        orient = self.getOrientation()
        while abs(orient[2] - angle) > 0.05:
            if (orient[2] - angle) > 0:
                self.move_right(0.4)
                self.move_left(velocity=-0.4)
            else:
                self.move_left(0.4)
                self.move_right(velocity=-0.4)
            orient = self.getOrientation()
        self.stop()


def rubbish():
    inter = PioneerP3DX_interface()
    print(inter.getPosition())
    pos0 = [0.05, -0.005, 0.05]
    pos = [-1.5904, -0.6783, 0.3535]

    pos = [-1.5904, -0.6783, -0.3535]
    pos2 = [0, -0.6783, 0.3535]
    pos1 = [-1.5904, 0, 0.3535]

    target_pos = [-0.025, -1.975, 0.0]
    x = [-2.1, 2.1]
    y = [-2.1, 2.1]

    print("start simulation")

    inter = PioneerP3DX_interface()
    print(vrep.simxStartSimulation(inter.clientID, vrep.simx_opmode_oneshot))

    target_pos = [-0.025, 0.0, 0.0]

    target = [-0.4652961790561676, 1.9605382680892944, 0.13865897059440613]
    # inter.move_to(target_pos)

    inter.check_position(inter.getPosition())
    # inter.streamCameraImage()
    # inter.stop()
    # inter.setPosition(target)

    print(inter.getPosition())
