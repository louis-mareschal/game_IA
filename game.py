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
        self.background = pygame.image.load(config.IMAGE_BACKGROUND_PATH)
        self.font = pygame.font.Font(pygame.font.get_default_font(), 50)



    def add_monster(self, monster):
        self.all_monsters.add(monster)

    def add_player(self, player):
        self.all_players.add(player)

    def update(self, nets, ge):
        if self.training_number % 2:
            return self.update_monsters(nets, ge)

        return self.update_players(nets, ge)

    def update_monsters(self, nets, ge):
        for i, monster in enumerate(self.all_monsters):
            player = self.all_players.sprites()[i]
            next_move_monster = nets[i].activate(
                [monster.rect.x - player.rect.x, monster.rect.y - player.rect.y])

            monster.move(next_move_monster)
            player.move(player.get_next_move(monster.rect.x, monster.rect.y))
            if monster.rect.colliderect(player.rect):
                ge[i].fitness += 1
        return ge, nets

    def update_players(self, nets, ge):
        for i, player in enumerate(self.all_players):
            monster = self.all_monsters.sprites()[i]
            next_move_player = nets[i].activate(
                [monster.rect.x - player.rect.x, monster.rect.y - player.rect.y,
                 player.rect.x - config.WINDOW_WIDTH//2, player.rect.y - config.WINDOW_HEIGHT//2])

            player.move(next_move_player)

            monster.move(monster.get_next_move(player.rect.x, player.rect.y))

            if player.rect.colliderect(monster.rect):
                ge[i].fitness -= 1

        return ge, nets

    def purge(self, ge, nets):
        ge_with_id = [[ge[i].fitness, i] for i in range(len(ge))]
        ge_with_id_sorted = sorted(ge_with_id, key=lambda x: x[0])
        print(f"Sorted fitness list : {ge_with_id_sorted}")
        ids_to_remove = sorted([genome_with_id[1] for genome_with_id in ge_with_id_sorted[:len(ge)//4]], reverse=True)
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

        generation_and_episode_text = self.font.render(f"Generation n°{self.generation} Episode n°{self.episode}", True, (30, 0, 0))
        screen.blit(generation_and_episode_text, (50, 100))

        # Draw the players on the sreen
        self.all_players.draw(screen)
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
        player = self.all_players.sprites()[0]
        screen.blit(player.image, player.rect)
        # Draw the monsters on the sreen
        monster = self.all_monsters.sprites()[0]
        screen.blit(monster.image, monster.rect)

        pygame.display.flip()

