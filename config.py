import os
from typing import Final

# Training settings
NUMBER_GEN = 50
MAX_NUMBER_EPISODE = 5
NUMBER_NETS_TRAINING = 10
# During an episode, if the evolution of the mean fitness is less than this %, it ends the generation
THRESHOLD_EVOL_RANKING = 50

# Game settings
MONSTER_LIFE = 3000
SPEED: Final = 0.25

# Display settings
IMAGE_MONSTER_PATH: Final = os.path.join(os.getcwd(), "assets", "monster.png")
IMAGE_BACKGROUND_PATH: Final = os.path.join(os.getcwd(), "assets", "bg.jpg")
WINDOW_STATS_HEIGHT: Final = 100
WINDOW_WIDTH: Final = 1080
WINDOW_HEIGHT: Final = 720
IMAGE_SIZE: Final = (50, 50)
