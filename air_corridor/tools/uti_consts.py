import numpy as np

# draw parameters
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
WORLD_WIDTH = (20 + 2) * 2 * 2
SCALE = SCREEN_WIDTH / WORLD_WIDTH * 1.5
OFFSET_x = SCREEN_WIDTH / 2
OFFSET_y = SCREEN_HEIGHT / 2
FLYOBJECT_SIZE = 5

# colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (120, 120, 120)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
PURPLE = (129, 132, 203)

# # env reward

# env reward
'''
each step penalty / 0.01 < abs(breach, collision)
'''

REWARD_REACH = 160.0
REWARD_INTERMEDIA = 40
REACH_ALIGNMENT = 0
LIABILITY_PENALITY = -10  # for vicinity






# LIABILITY_PENALITY = -10  # for vicinity
# PENALTY_TIME = -0.2
PENALTY_COLLISION = -120
PENALTY_BREACH = -100
## change made by May 6
LIABILITY_PENALITY = 0  # for vicinity
PENALTY_TIME = 0
# PENALTY_COLLISION = -80
# PENALTY_BREACH = -140



REWARD_POSITIVE_STEP = PENALTY_TIME * 0.1  # it must be smaller than abs(PENALTY_TIME)
Max_collision_vigilant = 1
REWARD_BOID = PENALTY_TIME * 0.2
BREACH_TOLERANCE = 1

# counter limit
OUTSIDE_TOLORENCE = 2

DISCRET_ACTION_SPACE = 8
NUM_ITERS = 1000

Z_UNIT = np.array([.0, .0, 1.0])
X_UNIT = np.array([1.0, .0, .0])
Y_UNIT = np.array([0.0, 1.0, .0])
TRIVIAL_TOLERANCE = 1e-05
CORRIDOR_OVERLAP = 1e-2

# assert abs(PENALTY_TIME) > REWARD_POSITIVE_STEP
