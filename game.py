import numpy as np
import pygame
import config
import random
import time


class Game:
    def __init__(self, generation: int, training_number: int, episode: int, pop_size: int):
        self.generation = generation
        self.training_number = training_number
        self.episode = episode
        self.current_step = 0
        self.player_killed = 0
        self.monster_removed = 0

        n = 5
        self.grid = np.zeros((n + 2 * (n-1), n + 2 * (n-1)), dtype=int)
        self.grid[n-2: -n+2, n-2: -n+2] = -1
        self.grid[n-1: -n+1, n-1: -n+1] = 0

        self.all_monsters = pygame.sprite.Group()
        self.pop_size = pop_size
        self.player = None
        self.all_players = pygame.sprite.Group()
        self.all_players_alive = pygame.sprite.Group()
        self.background = pygame.image.load(config.IMAGE_BACKGROUND_PATH)
        self.font = pygame.font.Font(pygame.font.get_default_font(), 10)

    def add_monster(self, monster):
        self.all_monsters.add(monster)

    def add_player(self, player):
        self.player = player
        self.all_players.add(player)
        self.all_players_alive.add(player)

    def run_episode(self, nets, ge, population, screen):
        running = True
        paused = False

        max_step = 6000
        self.current_step = 0

        while running and self.current_step < max_step and self.player_killed + self.monster_removed < self.pop_size:
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
        if self.training_number % 2:
            return self.update_monsters(nets, ge)

        return self.update_players(nets, ge)

    def update_monsters(self, nets, ge):
        self.player.move_random()
        for monster in self.all_monsters:
            next_move_monster = nets[monster.id].activate(
                monster.get_local_view_optimized(self.player, self.grid).flatten()
            )

            # next_move_monster = nets[monster.id].activate(
            #     [player.rect.y / config.WINDOW_HEIGHT,
            #      monster.rect.y / config.WINDOW_HEIGHT,
            #      player.rect.x / config.WINDOW_WIDTH,
            #      monster.rect.x / config.WINDOW_WIDTH])

            monster.move(next_move_monster)
            monster.check_hit_wall()

            if monster.rect.colliderect(self.player.rect):
                self.player.life[monster.id] -= 1
                ge[monster.id].fitness += 0.1

            min_dist_wall = min(
                monster.rect.left,
                monster.rect.top,
                config.WINDOW_WIDTH - monster.rect.right,
                config.WINDOW_HEIGHT - monster.rect.bottom,
            )

            ge[monster.id].fitness -= 0.01 if min_dist_wall == 0 else 0

            if monster.life == 0:
                self.monster_removed += 1
                self.all_monsters.remove(monster)
            if self.player.life[monster.id] == 0:
                ge[monster.id].fitness += (6000 - self.current_step) / 25  # len(self.all_players) - self.player_killed
                self.player_killed += 1
                self.all_monsters.remove(monster)

        return ge, nets


    def update_players(self, nets, ge):
        for monster in self.all_monsters:
            player = self.all_players.sprites()[monster.id]
            next_move_player = nets[player.id].activate(
                [
                    monster.rect.x - player.rect.x,
                    monster.rect.y - player.rect.y,
                    player.rect.x,
                    player.rect.y,
                    config.WINDOW_WIDTH - player.rect.x,
                    config.WINDOW_HEIGHT - player.rect.y,
                ]
            )

            player.move(next_move_player)

            monster.move(monster.get_next_move(player.rect.x, player.rect.y))

            if player.rect.colliderect(monster.rect):
                player.life -= 1
                ge[player.id].fitness += 0.1

            ge[player.id].fitness += 0.1

            if player.life == 0:
                self.all_monsters.remove(monster)
                self.all_players_alive.remove(player)

        return ge, nets


    def display_game(self, screen):
        screen.blit(self.background, (0, 0))
        # Printing the generation number and the type of training as well as the training number
        if self.training_number % 2:
            training_name = f"Training Monsters n°{(self.training_number + 1) // 2}"
        else:
            training_name = f"Training Players n°{(self.training_number + 1) // 2}"
        training_name_text = self.font.render(training_name, True, (30, 0, 0))
        screen.blit(training_name_text, (30, 10))

        generation_and_episode_text = self.font.render(
            f"Generation n°{self.generation} Episode n°{self.episode}", True, (40, 0, 0)
        )
        screen.blit(generation_and_episode_text, (30, 40))

        population_text = self.font.render(
            f"Population : {len(self.all_players_alive)} ({self.player_killed} kills {self.monster_removed} suicides)",
            True, (50, 0, 0)
        )
        screen.blit(population_text, (30, 70))

        # Separation line for the stats
        pygame.draw.line(screen, (50, 0, 0), (0, config.WINDOW_STATS_HEIGHT),
                         (config.WINDOW_WIDTH, config.WINDOW_STATS_HEIGHT), 4)

        # Draw the players on the sreen
        self.all_players_alive.draw(screen)
        # Draw the monsters on the sreen
        self.all_monsters.draw(screen)

        pygame.display.flip()

# def display_network(self):
#     network_display = pygame.Surface((200, 200))
#     network_display.fill((255, 255, 255))
#     generate_visualized_network(genome, nodes)
#     render_visualized_network(genome, nodes, network_display)
#
# def generate_visualized_network(self, genome, nodes):
#     """Generate the positions/colors of the neural network nodes"""
#     for i in genome.get_nodes():
#         if genome.is_input(i):
#             color = (0, 0, 255)
#             x = 50
#             y = 140 + i * 60
#         elif genome.is_output(i):
#             color = (255, 0, 0)
#             x = NETWORK_WIDTH - 50
#             y = HEIGHT / 2
#         else:
#             color = (0, 0, 0)
#             x = random.randint(NETWORK_WIDTH / 3, int(NETWORK_WIDTH * (2.0 / 3)))
#             y = random.randint(20, HEIGHT - 20)
#         nodes[i] = [(int(x), int(y)), color]
#
# def render_visualized_network(self, genome, nodes, display):
#     """Render the visualized neural network"""
#     genes = genome.get_edges()
#     for edge in genes:
#         if genes[edge].enabled:  # Enabled or disabled edge
#             color = (0, 255, 0)
#         else:
#             color = (255, 0, 0)
#
#         pygame.draw.line(display, color, nodes[edge[0]][0], nodes[edge[1]][0], 3)
#
#     for n in nodes:
#         pygame.draw.circle(display, nodes[n][1], nodes[n][0], 7)
