import numpy as np
import pygame
import config
import random
import time


class Game:
    def __init__(self, generation: int, episode: int):
        self.generation = generation
        self.episode = episode
        self.current_step = 0
        self.monster_removed = 0

        self.all_monsters = pygame.sprite.Group()
        self.monster_alive = pygame.sprite.Group()
        self.background = pygame.image.load(config.IMAGE_BACKGROUND_PATH)
        self.font = pygame.font.Font(pygame.font.get_default_font(), 30)

    def add_monster(self, monster):
        self.all_monsters.add(monster)
        self.monster_alive.add(monster)

    def run_episode(self, nets, ge, population, screen):
        running = True
        paused = False

        max_step = 6000
        self.current_step = 0

        while running and self.current_step < max_step and len(self.monster_alive) > 0:
            population.genome_reporter.set_start_time_step()
            self.current_step += 1
            ge, nets = self.update(nets, ge)
            population.genome_reporter.set_update_time_step()

            if self.current_step % 100 == 0:
                self.display_game(screen)

            looping = True
            while looping:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            pygame.quit()
                        elif event.key == pygame.K_SPACE:
                            paused = not paused
                    elif event.type == pygame.QUIT:
                        running = False
                        pygame.quit()

                if paused:
                    time.sleep(0.5)
                else:
                    looping = False

            population.genome_reporter.set_display_time_step()
        return ge, nets, population

    def update(self, nets, ge):
        distances_monsters = self.update_distances_monsters()
        next_move_monsters = [nets[monster.id].activate(distances_monsters[i]) for i, monster in
                              enumerate(self.monster_alive)]
        for i, monster in enumerate(self.monster_alive):
            monster.move(next_move_monsters[i])
            ge[monster.id].fitness += 0.1

        monster_to_remove = []
        for monster in self.monster_alive:
            if monster.check_hit_wall():
                monster_to_remove.append(monster)
                continue
            colliding_monsters = pygame.sprite.spritecollide(monster, self.monster_alive, False)
            if len(colliding_monsters) > 1:
                monster_to_remove.extend(colliding_monsters)

        for monster in list(set(monster_to_remove)):
            self.monster_alive.remove(monster)

        return ge, nets

    def update_distances_monsters(self):
        distances_monsters = []

        # Iterate over each monster
        for monster in self.monster_alive:

            # Initialize distances for the current monster with the side of the screen
            dist_top_wall = monster.rect.top - config.WINDOW_STATS_HEIGHT
            dist_bottom_wall = config.WINDOW_HEIGHT - monster.rect.bottom
            dist_left_wall = monster.rect.left
            dist_right_wall = config.WINDOW_WIDTH - monster.rect.right
            distances_monster = [dist_top_wall,
                                 dist_bottom_wall,
                                 dist_left_wall,
                                 dist_right_wall,
                                 min(2 * dist_left_wall * 2 ** -0.5, 2 * dist_top_wall * 2 ** -0.5),
                                 min(2 * dist_right_wall * 2 ** -0.5, 2 * dist_top_wall * 2 ** -0.5),
                                 min(2 * dist_left_wall * 2 ** -0.5, 2 * dist_bottom_wall * 2 ** -0.5),
                                 min(2 * dist_right_wall * 2 ** -0.5, 2 * dist_bottom_wall * 2 ** -0.5)]

            # Check for the closest monster in each direction
            for other_monster in self.monster_alive:
                if other_monster == monster:
                    continue
                dist_y_top = monster.rect.top - other_monster.rect.bottom
                dist_y_bottom = monster.rect.bottom - other_monster.rect.top
                dist_x_left = monster.rect.left - other_monster.rect.right
                dist_x_right = monster.rect.right - other_monster.rect.left

                if abs(other_monster.rect.centerx - monster.rect.centerx) <= config.IMAGE_SIZE[0] // 2:

                    if dist_y_top >= 0:  # top
                        distances_monster[0] = min(distances_monster[0], dist_y_top)
                    else:  # bottom
                        distances_monster[1] = min(distances_monster[1], abs(dist_y_bottom))

                elif abs(other_monster.rect.centery - monster.rect.centery) <= config.IMAGE_SIZE[0] // 2:
                    if dist_x_left >= 0:  # left
                        distances_monster[2] = min(distances_monster[2], dist_x_left)
                    else:  # right
                        distances_monster[3] = min(distances_monster[3], abs(dist_x_right))

                if dist_x_left >= -config.IMAGE_SIZE[0]:  # left
                    if dist_y_top >= -config.IMAGE_SIZE[0]:  # top
                        # Right side collision
                        if  other_monster.rect.top <= monster.rect.top - dist_x_left <= other_monster.rect.bottom:
                            distances_monster[4] = min(distances_monster[4], 2 * dist_x_left * 2 ** -0.5)
                        # Bottom side collision
                        elif other_monster.rect.left <= monster.rect.left - dist_y_top <= other_monster.rect.right:
                            distances_monster[4] = min(distances_monster[4], 2 * dist_y_top * 2 ** -0.5)
                    else:  # bottom
                        # Right side collision
                        if other_monster.rect.top <= monster.rect.bottom + dist_x_left <= other_monster.rect.bottom:
                            distances_monster[6] = min(distances_monster[6], 2 * dist_x_left * 2 ** -0.5)
                        # Top side collision
                        elif other_monster.rect.left <= monster.rect.left - abs(
                                dist_y_bottom) <= other_monster.rect.right:
                            distances_monster[6] = min(distances_monster[6], 2 * abs(dist_y_bottom) * 2 ** -0.5)
                else:  # right
                    if dist_y_top >= -config.IMAGE_SIZE[0]:  # top
                        # Left side collision
                        if other_monster.rect.top <= monster.rect.top - abs(dist_x_right) <= other_monster.rect.bottom:
                            distances_monster[5] = min(distances_monster[5], 2 * abs(dist_x_right) * 2 ** -0.5)
                        # Bottom side collision
                        elif other_monster.rect.left <= monster.rect.right + dist_y_top <= other_monster.rect.right:
                            distances_monster[5] = min(distances_monster[5], 2 * dist_y_top * 2 ** -0.5)
                    else:  # bottom
                        # Left side collision
                        if other_monster.rect.top <= monster.rect.bottom + abs(dist_x_right) <= other_monster.rect.bottom:
                            distances_monster[7] = min(distances_monster[7], 2 * abs(dist_x_right) * 2 ** -0.5)
                        # Top side collision
                        elif other_monster.rect.left <= monster.rect.right + abs(
                                dist_y_bottom) <= other_monster.rect.right:
                            distances_monster[7] = min(distances_monster[7], 2 * abs(dist_y_bottom) * 2 ** -0.5)

            distances_monsters.append(distances_monster)

        return distances_monsters

    def display_game(self, screen):
        screen.blit(self.background, (0, 0))

        generation_and_episode_text = self.font.render(
            f"Generation n°{self.generation} Episode n°{self.episode}", True, (40, 0, 0)
        )
        screen.blit(generation_and_episode_text, (100, 10))

        population_text = self.font.render(
            f"Population : {len(self.monster_alive)}",
            True, (50, 0, 0)
        )
        screen.blit(population_text, (100, 50))

        # Separation line for the stats
        pygame.draw.line(screen, (50, 0, 0), (0, config.WINDOW_STATS_HEIGHT),
                         (config.WINDOW_WIDTH, config.WINDOW_STATS_HEIGHT), 4)

        # Draw the monsters on the sreen
        self.monster_alive.draw(screen)

        pygame.display.flip()
