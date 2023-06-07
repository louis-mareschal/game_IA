import numpy as np
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
        self.diagonal_speed = self.speed / 2 ** 0.5
        self.net_monster = None
        self.life = config.MONSTER_LIFE

    def set_net(self, net):
        self.net_monster = net

    def check_hit_wall(self):
        if (
                self.rect.top <= config.WINDOW_STATS_HEIGHT
                or self.rect.bottom >= config.WINDOW_HEIGHT
                or self.rect.left <= 0
                or self.rect.right >= config.WINDOW_WIDTH
        ):
            self.life -= 1

    def get_next_move(self, player, grid):
        return self.net_monster.activate(self.get_local_view(player, grid).flatten())

    def move(self, next_move_monster: typing.List[float]):
        match np.argmax(next_move_monster):
            case 0:
                self.move_up()
            case 1:
                self.move_down()
            case 2:
                self.move_left()
            case 3:
                self.move_right()
            case 4:
                self.move_up_left()
            case 5:
                self.move_up_right()
            case 6:
                self.move_down_left()
            case 7:
                self.move_down_right()

    def move_up(self):
        if self.rect.top > config.WINDOW_STATS_HEIGHT:
            self.pos.y -= self.speed
            self.rect.topleft = round(self.pos.x), round(self.pos.y)

    def move_down(self):
        if self.rect.bottom < config.WINDOW_HEIGHT:
            self.pos.y += self.speed
            self.rect.topleft = round(self.pos.x), round(self.pos.y)

    def move_left(self):
        if self.rect.left > 0:
            self.pos.x -= self.speed
            self.rect.topleft = round(self.pos.x), round(self.pos.y)

    def move_right(self):
        if self.rect.right < config.WINDOW_WIDTH:
            self.pos.x += self.speed
            self.rect.topleft = round(self.pos.x), round(self.pos.y)

    def move_up_left(self):
        if self.rect.top > config.WINDOW_STATS_HEIGHT and self.rect.left > 0:
            self.pos.x -= self.diagonal_speed
            self.pos.y -= self.diagonal_speed
            self.rect.topleft = round(self.pos.x), round(self.pos.y)
        elif self.rect.top > config.WINDOW_STATS_HEIGHT:
            self.move_up()
        elif self.rect.left > 0:
            self.move_left()

    def move_up_right(self):
        if self.rect.top > config.WINDOW_STATS_HEIGHT and self.rect.right < config.WINDOW_WIDTH:
            self.pos.x += self.diagonal_speed
            self.pos.y -= self.diagonal_speed
            self.rect.topleft = round(self.pos.x), round(self.pos.y)
        elif self.rect.top > config.WINDOW_STATS_HEIGHT:
            self.move_up()
        elif self.rect.right < config.WINDOW_WIDTH:
            self.move_right()

    def move_down_left(self):
        if self.rect.bottom < config.WINDOW_HEIGHT and self.rect.left > 0:
            self.pos.x -= self.diagonal_speed
            self.pos.y += self.diagonal_speed
            self.rect.topleft = round(self.pos.x), round(self.pos.y)
        elif self.rect.bottom < config.WINDOW_HEIGHT:
            self.move_down()
        elif self.rect.left > 0:
            self.move_left()

    def move_down_right(self):
        if self.rect.bottom < config.WINDOW_HEIGHT and self.rect.right < config.WINDOW_WIDTH:
            self.pos.x += self.diagonal_speed
            self.pos.y += self.diagonal_speed
            self.rect.topleft = round(self.pos.x), round(self.pos.y)
        elif self.rect.bottom < config.WINDOW_HEIGHT:
            self.move_down()
        elif self.rect.right < config.WINDOW_WIDTH:
            self.move_right()

    def get_local_view_optimized(self, player, empty_grid):
        grid = np.array(empty_grid)
        size_grid = (grid.shape[0] + 2) // 3
        center_y = (player.rect.centery - config.WINDOW_STATS_HEIGHT) // config.IMAGE_SIZE[0] + size_grid - 1
        center_x = player.rect.centerx // config.IMAGE_SIZE[0] + size_grid - 1
        grid[center_y, center_x] = 1

        local_view_size = size_grid*2
        half_local_view = size_grid

        line_monster = (self.rect.centery - config.WINDOW_STATS_HEIGHT) // config.IMAGE_SIZE[0] + size_grid - 1
        column_monster = self.rect.centerx // config.IMAGE_SIZE[0] + size_grid - 1

        start_line, end_line = line_monster - half_local_view, line_monster + half_local_view + 1
        start_column, end_column = column_monster - half_local_view, column_monster + half_local_view + 1

        local_view = grid[start_line:end_line, start_column:end_column]

        return local_view

