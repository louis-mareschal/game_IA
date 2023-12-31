import pygame
import random
import typing
import config
import numpy as np


class Player(pygame.sprite.Sprite):
    def __init__(self, x, y, nb_monsters):
        super().__init__()
        self.image = pygame.transform.scale(
            pygame.image.load(config.IMAGE_PLAYER_PATH), config.IMAGE_SIZE
        )
        self.pos = pygame.math.Vector2(x, y)
        self.rect = self.image.get_rect(topleft=self.pos)
        self.speed = config.SPEED
        self.diagonal_speed = self.speed / 2**0.5
        self.net_player = None
        self.life = [config.PLAYER_LIFE] * nb_monsters

        self.vx = random.uniform(-1, 1)
        self.vy = random.uniform(-1, 1)
        norm = (self.vx**2 + self.vy**2) ** 0.5
        self.vx *= self.speed / norm
        self.vy *= self.speed / norm
        self.current_speed = self.speed

    def move_random(self):
        self.current_speed = min(self.speed, self.current_speed+self.speed/500)
        self.pos.y += self.vy
        self.pos.x += self.vx
        vx_var = random.uniform(-0.05 * self.current_speed, 0.05 * self.current_speed)
        vy_var = random.uniform(-0.05 * self.current_speed, 0.05 * self.current_speed)

        self.vx += vx_var
        self.vy += vy_var
        norm = (self.vx**2 + self.vy**2) ** 0.5
        self.vx *= self.current_speed / norm
        self.vy *= self.current_speed / norm

        # Check if player has hit the edge of the screen
        if self.rect.left < 0:
            self.pos.x = 0
            self.vx *= -1
            self.current_speed = 0
        elif self.rect.right > config.WINDOW_WIDTH:
            self.pos.x = config.WINDOW_WIDTH - config.IMAGE_SIZE[0]
            self.vx *= -1
            self.current_speed = 0

        if self.rect.top < config.WINDOW_STATS_HEIGHT:
            self.pos.y = config.WINDOW_STATS_HEIGHT
            self.vy *= -1
            self.current_speed = 0
        elif self.rect.bottom > config.WINDOW_HEIGHT:
            self.pos.y = config.WINDOW_HEIGHT - config.IMAGE_SIZE[0]
            self.vy *= -1
            self.current_speed = 0

        self.rect.topleft = round(self.pos.x), round(self.pos.y)

    def set_net(self, net):
        self.net_player = net

    def get_next_move(self, monster_x, monster_y):
        if self.net_player:
            return self.net_player.activate(
                [
                    monster_x - self.rect.x,
                    monster_y - self.rect.y,
                    self.rect.x,
                    self.rect.y,
                    config.WINDOW_WIDTH - self.rect.x,
                    config.WINDOW_HEIGHT - self.rect.y,
                ]
            )
        return random.choice(
            [[1, 0.5], [0, 0.5], [0.5, 0], [0.5, 1], [1, 0], [1, 1], [0, 0], [0, 1]]
        )

    def move(self, next_move_player: typing.List[float]):
        up = next_move_player[0] >= 6 / 10
        down = next_move_player[0] <= 4 / 10
        right = next_move_player[1] >= 6 / 10
        left = next_move_player[1] <= 4 / 10
        if up and not (right or left):
            if (
                self.rect.top > config.WINDOW_STATS_HEIGHT
            ):  # check if the monster is within the top screen boundary
                self.move_up()
        elif down and not (right or left):
            if (
                self.rect.bottom < config.WINDOW_HEIGHT
            ):  # check if the monster is within the bottom screen boundary
                self.move_down()
        elif left and not (up or down):
            if (
                self.rect.left > 0
            ):  # check if the monster is within the left screen boundary
                self.move_left()
        elif right and not (up or down):
            if (
                self.rect.right < config.WINDOW_WIDTH
            ):  # check if the monster is within the right screen boundary
                self.move_right()
        elif up and left:
            # check if the monster is within the top and left screen boundaries
            if self.rect.top > config.WINDOW_STATS_HEIGHT and self.rect.left > 0:
                self.move_up_left()
        elif up and right:
            # check if the monster is within the top and right screen boundaries
            if self.rect.top > config.WINDOW_STATS_HEIGHT and self.rect.right < config.WINDOW_WIDTH:
                self.move_up_right()
        elif down and left:
            # check if the monster is within the bottom and left screen boundaries
            if self.rect.bottom < config.WINDOW_HEIGHT and self.rect.left > 0:
                self.move_down_left()
        elif down and right:
            # check if the monster is within the bottom and right screen boundaries
            if (
                self.rect.bottom < config.WINDOW_HEIGHT
                and self.rect.right < config.WINDOW_WIDTH
            ):
                self.move_down_right()

    def move_up(self):
        self.pos.y -= self.speed
        self.rect.topleft = round(self.pos.x), round(self.pos.y)

    def move_down(self):
        self.pos.y += self.speed
        self.rect.topleft = round(self.pos.x), round(self.pos.y)

    def move_left(self):
        self.pos.x -= self.speed
        self.rect.topleft = round(self.pos.x), round(self.pos.y)

    def move_right(self):
        self.pos.x += self.speed
        self.rect.topleft = round(self.pos.x), round(self.pos.y)

    def move_up_left(self):
        self.pos.x -= self.diagonal_speed
        self.pos.y -= self.diagonal_speed
        self.rect.topleft = round(self.pos.x), round(self.pos.y)

    def move_up_right(self):
        self.pos.x += self.diagonal_speed
        self.pos.y -= self.diagonal_speed
        self.rect.topleft = round(self.pos.x), round(self.pos.y)

    def move_down_left(self):
        self.pos.x -= self.diagonal_speed
        self.pos.y += self.diagonal_speed
        self.rect.topleft = round(self.pos.x), round(self.pos.y)

    def move_down_right(self):
        self.pos.x += self.diagonal_speed
        self.pos.y += self.diagonal_speed
        self.rect.topleft = round(self.pos.x), round(self.pos.y)
