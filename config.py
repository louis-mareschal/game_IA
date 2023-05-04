import os
from typing import Final

NUMBER_TRAININGS = 10
NUMBER_GEN_PER_TRAINING = 5
NUMBER_NETS_TRAINING = 1
PLAYER_LIFE = 300
WINDOW_WIDTH: Final = 1080
WINDOW_HEIGHT: Final = 720
IMAGE_PLAYER_PATH: Final = os.path.join(os.getcwd(), "assets", "player.png")
IMAGE_MONSTER_PATH: Final = os.path.join(os.getcwd(), "assets", "monster.png")
IMAGE_BACKGROUND_PATH: Final = os.path.join(os.getcwd(), "assets", "bg.jpg")
IMAGE_SIZE: Final = (50, 50)
SPEED: Final = 1
