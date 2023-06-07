import os
from typing import Final

# Training settings
NUMBER_GEN_PER_TRAINING = 50
NUMBER_TRAININGS = 10
MAX_NUMBER_EPISODE = 5
NUMBER_NETS_TRAINING = 10
# During an episode, if the evolution of the mean fitness is less than this %, it ends the generation
THRESHOLD_EVOL_RANKING = 50

# Game settings
PLAYER_LIFE = 1000
MONSTER_LIFE = 3000
SPEED: Final = 0.25

# Display settings
IMAGE_PLAYER_PATH: Final = os.path.join(os.getcwd(), "assets", "player.png")
IMAGE_MONSTER_PATH: Final = os.path.join(os.getcwd(), "assets", "monster.png")
IMAGE_BACKGROUND_PATH: Final = os.path.join(os.getcwd(), "assets", "bg.jpg")
WINDOW_STATS_HEIGHT: Final = 100
WINDOW_WIDTH: Final = 200
WINDOW_HEIGHT: Final = 300
IMAGE_SIZE: Final = (40, 40)
