import sys
import vrep

print('Program started')
vrep.simxFinish(-1)  # just in case, close all opened connections
clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)  # Connect to V-REP

if clientID != -1:
    print('Connected to remote API server')

else:
    print('Remote API function call returned with error code: ')
    sys.exit("couldn't connect")

err_code, left_motor_handler = vrep.simxGetObjectHandle(0, "Pioneer_p3dx_leftMotor", vrep.simx_opmode_blocking)
err_code, right_motor_handler = vrep.simxGetObjectHandle(0, "Pioneer_p3dx_rightMotor", vrep.simx_opmode_blocking)
print("left", left_motor_handler, "right", right_motor_handler)
err_code, sensor1 = vrep.simxGetObjectHandle(0, "Pioneer_p3dx_ultrasonicSensor1", vrep.simx_opmode_streaming)

print(err_code, " sronsro")
print(sensor1, " sronsro")
vrep.simxSetJointTargetVelocity(clientID, left_motor_handler, 1, vrep.simx_opmode_blocking)
vrep.simxSetJointTargetVelocity(clientID, right_motor_handler, -0.2, vrep.simx_opmode_blocking)

returnCode, detectionState, detectedPoint, detectedObjectHandle, detectedSurfaceNormalVector \
    = vrep.simxReadProximitySensor(clientID, sensor1, vrep.simx_opmode_streaming)

print(returnCode, detectionState, detectedPoint)
