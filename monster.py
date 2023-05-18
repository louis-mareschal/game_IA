import pygame
import typing
import config


class Monster(pygame.sprite.Sprite):
    def __init__(self, x, y, id):
        super().__init__()
        self.id = id
        self.image = pygame.transform.scale(
            pygame.image.load(config.IMAGE_MONSTER_PATH), config.IMAGE_SIZE
        )
        self.pos = pygame.math.Vector2(x, y)
        self.rect = self.image.get_rect(topleft=self.pos)
        self.speed = config.SPEED
        self.diagonal_speed = self.speed / 2**0.5
        self.net_monster = None
        self.life = config.MONSTER_LIFE

    def set_net(self, net):
        self.net_monster = net

    def check_hit_wall(self):
        if (
            self.rect.top <= 0
            or self.rect.bottom >= config.WINDOW_HEIGHT
            or self.rect.left <= 0
            or self.rect.right >= config.WINDOW_WIDTH
        ):
            self.life -= 1

    def get_next_move(self, player_x, player_y):
        return self.net_monster.activate(
            [
                player_y / config.WINDOW_HEIGHT,
                self.rect.y / config.WINDOW_HEIGHT,
                player_x / config.WINDOW_WIDTH,
                self.rect.x / config.WINDOW_WIDTH,
            ]
        )

    def move(self, next_move_monster: typing.List[float]):
        down = next_move_monster[0] > 5 / 10
        up = next_move_monster[0] < 5 / 10
        right = next_move_monster[1] > 5 / 10
        left = next_move_monster[1] < 5 / 10

        if up and not (right or left):
            if (
                self.rect.top > 0
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
            if self.rect.top > 0 and self.rect.left > 0:
                self.move_up_left()
        elif up and right:
            # check if the monster is within the top and right screen boundaries
            if self.rect.top > 0 and self.rect.right < config.WINDOW_WIDTH:
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
