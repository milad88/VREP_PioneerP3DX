import numpy as np
e = 0.9
decay = 0.995
for i in range(500):
#     print(np.random.randn() < -2)
    e = e* decay
    print(e)
# print(int(0.75 *15))
# from Exploration_Noise import ExplorationNoise
# OU_THETA = 0.15
# OU_MU = 0.
# OU_SIGMA = 0.3
# MAX_STEPS_EPISODE = 1000
# EXPLORATION_TIME = 300
# noise = ExplorationNoise.ou_noise(OU_THETA, OU_MU, OU_SIGMA, MAX_STEPS_EPISODE)
# print(noise)
# print("noise agter")
# noise = ExplorationNoise.exp_decay(noise, EXPLORATION_TIME)
# print(noise[:305])
#
# print(1-noise[298])