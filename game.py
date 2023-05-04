import pygame
import sys
import config


class Game:
    def __init__(self, generation: int, training_number: int, episode: int):
        self.generation = generation
        self.training_number = training_number
        self.episode = episode
        self.all_monsters = pygame.sprite.Group()
        self.all_players = pygame.sprite.Group()
        self.all_players_alive = pygame.sprite.Group()
        self.background = pygame.image.load(config.IMAGE_BACKGROUND_PATH)
        self.font = pygame.font.Font(pygame.font.get_default_font(), 50)

    def add_monster(self, monster):
        self.all_monsters.add(monster)

    def add_player(self, player):
        self.all_players.add(player)
        self.all_players_alive.add(player)

    def update(self, nets, ge):
        if self.training_number % 2:
            return self.update_monsters(nets, ge)

        return self.update_players(nets, ge)

    def update_monsters(self, nets, ge):
        for monster in self.all_monsters:
            player = self.all_players.sprites()[monster.id]
            next_move_monster = nets[monster.id].activate(
                [player.rect.x - monster.rect.x,
                 player.rect.y - monster.rect.y,
                 monster.rect.x, monster.rect.y,
                 config.WINDOW_WIDTH - monster.rect.x,
                 config.WINDOW_HEIGHT - monster.rect.y])

            monster.move(next_move_monster)
            player.move(player.get_next_move(monster.rect.x, monster.rect.y))

            if monster.rect.colliderect(player.rect):
                player.life -= 1
                ge[monster.id].fitness += 0.1
            ge[monster.id].fitness -= 0.1


            if player.life == 0:
                self.all_monsters.remove(monster)
                self.all_players_alive.remove(player)

        return ge, nets

    def update_players(self, nets, ge):
        for monster in self.all_monsters:
            player = self.all_players.sprites()[monster.id]
            next_move_player = nets[player.id].activate(
                [monster.rect.x - player.rect.x,
                 monster.rect.y - player.rect.y,
                 player.rect.x, player.rect.y,
                 config.WINDOW_WIDTH - player.rect.x,
                 config.WINDOW_HEIGHT - player.rect.y])

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

    def update_demo(self, net):
        player = self.all_players.sprites()[0]
        monster = self.all_monsters.sprites()[0]
        if self.training_number % 2:
            next_move_monster = net.activate(
                [player.rect.x - monster.rect.x,
                 player.rect.y - monster.rect.y,
                 monster.rect.x, monster.rect.y,
                 config.WINDOW_WIDTH - monster.rect.x,
                 config.WINDOW_HEIGHT - monster.rect.y])

            next_move_player = player.get_next_move(monster.rect.x, monster.rect.y)

        else:
            next_move_player = net.activate(
                [monster.rect.x - player.rect.x,
                 monster.rect.y - player.rect.y,
                 player.rect.x, player.rect.y,
                 config.WINDOW_WIDTH - player.rect.x,
                 config.WINDOW_HEIGHT - player.rect.y])
            next_move_monster = monster.get_next_move(player.rect.x, player.rect.y)

        monster.move(next_move_monster)
        player.move(next_move_player)

        if monster.rect.colliderect(player.rect):
            player.life -= 1

        if player.life == 0:
            self.all_monsters.remove(monster)
            self.all_players_alive.remove(player)

    def purge(self, ge, nets):
        ge_with_id = [[ge[i].fitness, i] for i in range(len(ge))]
        ge_with_id_sorted = sorted(ge_with_id, key=lambda x: x[0])
        print(f"Sorted fitness list : {ge_with_id_sorted}")
        ids_to_remove = sorted([genome_with_id[1] for genome_with_id in ge_with_id_sorted[:len(ge) // 4]], reverse=True)
        print(ids_to_remove)
        for id_to_remove in ids_to_remove:
            ge[id_to_remove] = -100000
            ge.pop(id_to_remove)
            nets.pop(id_to_remove)
        return ge, nets

    def display_game(self, screen):
        screen.blit(self.background, (0, 0))
        # Printing the generation number and the type of training as well as the training number
        if self.training_number % 2:
            training_name = f"Training Monsters n°{(self.training_number + 1) // 2}"
        else:
            training_name = f"Training Players n°{(self.training_number + 1) // 2}"
        training_name_text = self.font.render(training_name, True, (50, 0, 0))
        screen.blit(training_name_text, (50, 50))

        generation_and_episode_text = self.font.render(f"Generation n°{self.generation} Episode n°{self.episode}", True,
                                                       (20, 0, 0))
        screen.blit(generation_and_episode_text, (50, 100))

        population_text = self.font.render(f"Population : {len(self.all_players_alive)}", True, (20, 0, 0))
        screen.blit(population_text, (50, 150))

        # Draw the players on the sreen
        self.all_players_alive.draw(screen)
        # Draw the monsters on the sreen
        self.all_monsters.draw(screen)

        pygame.display.flip()

    def display_one_pair(self, screen):
        screen.blit(self.background, (0, 0))
        # Printing the generation number and the type of training as well as the training number
        if self.training_number % 2:
            training_name = f"Training Monsters n°{(self.training_number + 1) // 2}"
        else:
            training_name = f"Training Players n°{(self.training_number + 1) // 2}"
        training_name_text = self.font.render(training_name, True, (50, 0, 0))
        screen.blit(training_name_text, (50, 50))

        generation_and_episode_text = self.font.render(f"Generation n°{self.generation} Episode n°{self.episode}", True,
                                                       (30, 0, 0))
        screen.blit(generation_and_episode_text, (50, 100))

        # Draw the players on the sreen
        player = self.all_players_alive.sprites()[0]
        screen.blit(player.image, player.rect)
        # Draw the monsters on the sreen
        monster = self.all_monsters.sprites()[0]
        screen.blit(monster.image, monster.rect)

        pygame.display.flip()
