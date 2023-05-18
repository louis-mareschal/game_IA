import os
from typing import Final

# Training settings
NUMBER_GEN_PER_TRAINING = 10
NUMBER_TRAININGS = 10
MAX_NUMBER_EPISODE = 10
NUMBER_NETS_TRAINING = 10
# Number of the best genomes fitnesses to compute the evolution of the mean fitness
NUMBER_BEST_FITNESS_EVOL_MEAN = 30
# During an episode, if the evolution of the mean fitness is less than this %, it ends the generation
THRESHOLD_EVOL_MEAN_FITNESS = 15

# Game settings
PLAYER_LIFE = 1000
MONSTER_LIFE = 300
SPEED: Final = 1

# Display settings
IMAGE_PLAYER_PATH: Final = os.path.join(os.getcwd(), "assets", "player.png")
IMAGE_MONSTER_PATH: Final = os.path.join(os.getcwd(), "assets", "monster.png")
IMAGE_BACKGROUND_PATH: Final = os.path.join(os.getcwd(), "assets", "bg.jpg")
WINDOW_WIDTH: Final = 1080
WINDOW_HEIGHT: Final = 720
IMAGE_SIZE: Final = (50, 50)
