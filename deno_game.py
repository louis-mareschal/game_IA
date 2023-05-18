import os
import neat
from random import randint
import pygame
import time

from monster import Monster
from player import Player
import config


class DemoGame:

    def __init__(self, genome_monster, neat_config_monster, generation):
        self.generation = generation
        self.genome_monster = genome_monster
        self.neat_config_monster = neat_config_monster
        self.net_monster = neat.nn.FeedForwardNetwork.create(self.genome_monster, neat_config_monster)
        self.monster = None
        self.player = None
        self.fitness_demo = None

        # Pygame initialization
        self.background = pygame.image.load(config.IMAGE_BACKGROUND_PATH)
        self.font = pygame.font.Font(pygame.font.get_default_font(), 50)
        self.backup_caption = pygame.display.get_caption()[0]
        pygame.display.set_caption(
            "DEMO : AI alternative reinforcement training using genetic algorithm"
        )
        self.screen = pygame.display.get_surface()

    def show_demo(self, number_episodes: int):
        for episode in range(1, number_episodes + 1):
            self.fitness_demo = 0
            x_player, y_player = randint(100, config.WINDOW_WIDTH - 100), randint(100, config.WINDOW_HEIGHT - 100)
            x_monster, y_monster = randint(100, config.WINDOW_WIDTH - 100), randint(100, config.WINDOW_HEIGHT - 100)
            self.monster = Monster(x_monster, y_monster, -1)
            self.player = Player(x_player, y_player, -1)

            running = True
            max_step = 6000
            current_step = 0

            while running and current_step < max_step and self.player.life > 0:
                current_step += 1

                self.display_demo()
                self.update_demo()

                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            pygame.quit()
                    elif event.type == pygame.QUIT:
                        running = False
                        pygame.quit()
        pygame.display.set_caption(self.backup_caption)

    def display_demo(self):
        self.screen.blit(self.background, (0, 0))

        generation_and_fitness_text = self.font.render(
            f"Best monster generation n°{self.generation}", True, (50, 0, 0)
        )
        self.screen.blit(generation_and_fitness_text, (50, 50))

        mean_fitness_text = self.font.render(
            f"Mean fitness : {round(self.genome_monster.fitness, 2)}", True, (20, 0, 0)
        )
        self.screen.blit(mean_fitness_text, (50, 100))

        fitness_demo_text = self.font.render(
            f"Fitness demo : {round(self.fitness_demo, 2)}", True, (20, 0, 0)
        )
        self.screen.blit(fitness_demo_text, (50, 150))

        # Draw the players on the screen
        self.screen.blit(self.player.image, self.player.rect)
        # Draw the monsters on the screen
        self.screen.blit(self.monster.image, self.monster.rect)

        pygame.display.flip()

    def update_demo(self):

        next_move_monster = self.net_monster.activate(
            [
                self.player.rect.y / config.WINDOW_HEIGHT,
                self.monster.rect.y / config.WINDOW_HEIGHT,
                self.player.rect.x / config.WINDOW_WIDTH,
                self.monster.rect.x / config.WINDOW_WIDTH,
            ]
        )

        self.monster.move(next_move_monster)
        self.player.move_random()

        if self.monster.rect.colliderect(self.player.rect):
            self.player.life -= 1
            self.fitness_demo += 0.1
        min_dist_wall = min(
            self.monster.rect.left,
            self.monster.rect.top,
            config.WINDOW_WIDTH - self.monster.rect.right,
            config.WINDOW_HEIGHT - self.monster.rect.bottom,
        )
        self.fitness_demo -= 0.01 if min_dist_wall == 0 else 0

