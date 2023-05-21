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

    def get_local_view(self, player, empty_grid):
        grid = np.array(empty_grid)
        grid[(player.rect.centery - config.WINDOW_STATS_HEIGHT) // config.IMAGE_SIZE[0] + 1, player.rect.centerx //
             config.IMAGE_SIZE[0] + 1] = 1
        grid_size = grid.shape[0]
        local_view_size = 7
        half_local_view = local_view_size // 2

        local_view = np.zeros((local_view_size, local_view_size), dtype=int)

        line_monster = (self.rect.centery - config.WINDOW_STATS_HEIGHT) // config.IMAGE_SIZE[0] + 1
        column_monster = self.rect.centerx // config.IMAGE_SIZE[0] + 1
        for line in range(-half_local_view, half_local_view + 1):
            for column in range(-half_local_view, half_local_view + 1):
                column_grid = column_monster + column
                line_grid = line_monster + line

                if column_grid >= 0 and column_grid < grid_size and line_grid >= 0 and line_grid < grid_size:
                    local_view[line + half_local_view, column + half_local_view] = grid[column_grid, line_grid]

        return local_view
