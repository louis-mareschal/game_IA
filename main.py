import os
import pygame
import neat
from neat_modified.population import Population
from neat_modified.checkpoint_reporter import Checkpointer
import random
from random import randint
from game import Game
from monster import Monster
from player import Player
import time

import config

pygame.init()
pygame.display.set_caption("Jeu")
screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"
pygame.mouse.set_visible(False)


def eval_genomes(population, training_number: int, training_nets):
    """
    Runs the simulation of the current population of
    monsters or players and sets their fitness based on how many times they collide with their opponent.
    The training_number is to know if we are currently training the monsters or the players and how many trainings we
    did : if training_number is odd, it is training the monsters and if it's even, the players.
    """
    genomes, neat_config = list(population.population.items()), population.config
    # Updating the generation number
    nets = []
    ge = []
    display_one_pair = False
    display_best_only = False

    # TIME TESTING
    time_episodes = []
    time_init = []
    time_updates = []
    time_displays = []

    for genome_id, genome in genomes:
        genome.fitness = round(random.random(), 2)
        net = neat.nn.FeedForwardNetwork.create(genome, neat_config)
        nets.append(net)
        ge.append(genome)

    purged_episode = [50]
    for episode in range(1, 2):  # config.NUMBER_NETS_TRAINING + 1

        time_start_episode = time.time()

        game = Game(population.generation, training_number, episode)
        if episode in purged_episode:
            print(f"PURGING AT EPISODE {episode}")
            ge, nets = game.purge(ge, nets)
            if len(ge) != len(nets):
                print(f"ERROR AFTER PURGE : len(ge)={len(ge)} len(nets)={len(nets)} ")
                raise ValueError

        x_players, y_players = randint(100, config.WINDOW_WIDTH - 100), randint(100, config.WINDOW_HEIGHT - 100)
        x_monsters, y_monsters = randint(100, config.WINDOW_WIDTH - 100), randint(100, config.WINDOW_HEIGHT - 100)
        for index_entity in range(len(ge)):
            # Creating the monsters
            monster = Monster(x_monsters, y_monsters, index_entity)
            game.add_monster(monster)
            # Creating the players
            player = Player(x_players, y_players, index_entity)
            game.add_player(player)
            if training_nets:
                if training_number % 2:
                    player.set_net(training_nets[episode % config.NUMBER_NETS_TRAINING])
                else:

                    monster.set_net(training_nets[episode % config.NUMBER_NETS_TRAINING])

        running = True
        paused = False

        im_max = 6000
        im = 0

        time_init.append(time.time() - time_start_episode)
        while running and im < im_max:
            im += 1
            time_before_update = time.time()
            ge, nets = game.update(nets, ge)
            time_updates.append(time.time() - time_before_update)

            time_before_display = time.time()
            if not display_best_only and im % 100 == 0:
                if display_one_pair:
                    game.display_one_pair(screen)
                else:
                    game.display_game(screen)

            looping = True
            while looping:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            pygame.quit()
                        elif event.key == pygame.K_SPACE:
                            paused = not paused
                        elif event.key == pygame.K_m:
                            display_one_pair = not display_one_pair
                        elif event.key == pygame.K_b:
                            display_best_only = not display_best_only
                    elif event.type == pygame.QUIT:
                        running = False
                        pygame.quit()

                if paused:
                    time.sleep(0.5)
                else:
                    looping = False
            time_displays.append(time.time() - time_before_display)

        time_episodes.append(time.time() - time_start_episode)

    best_fitness = ge[0].fitness
    best_id = 0
    for i, (genome_id, genome) in enumerate(genomes):
        genome.fitness /= neat_config.pop_size
        if genome.fitness > best_fitness:
            best_fitness = genome.fitness
            best_id = i

    # Display the best genome only against all the training nets genomes
    if display_best_only:
        for episode in range(1, 2):  # config.NUMBER_NETS_TRAINING+1
            game = Game(population.generation, training_number, episode)
            x_player, y_player = 200, 400
            x_monster, y_monster = 800, 200
            monster = Monster(x_monster, y_monster, best_id)
            game.add_monster(monster)
            player = Player(x_player, y_player, best_id)
            game.add_player(player)
            if training_nets:
                if training_number % 2:
                    player.set_net(training_nets[episode % config.NUMBER_NETS_TRAINING])
                else:
                    monster.set_net(training_nets[episode % config.NUMBER_NETS_TRAINING])

            running = True
            im_max = 3000
            im = 0

            while running and im < im_max and player.life > 0:
                im += 1

                game.display_one_pair(screen)

                game.update_demo(nets[best_id])

                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            pygame.quit()
                        elif event.key == pygame.K_b:
                            display_best_only = not display_best_only
                    elif event.type == pygame.QUIT:
                        running = False
                        pygame.quit()

    # mean_time_episode = sum(time_episodes) / config.NUMBER_EPISODES
    # mean_time_init = sum(time_init) / config.NUMBER_EPISODES
    # mean_time_update = sum(time_updates) / config.NUMBER_EPISODES
    # mean_time_display = sum(time_displays) / config.NUMBER_EPISODES
    #
    # print(f"TIME REPORT TRAINING {type_training} N°{training_number} GENERATION N°{GENERATION}\n")
    # print(f"Mean time episodes : {round(mean_time_episode, 2)}")
    # print(
    #     f"Mean time init section during episodes : {round(mean_time_init, 2)} {int(100 * mean_time_init / mean_time_episode)}% ")
    # print(
    #     f"Mean time update section during episodes : {round(mean_time_update, 2)} {int(100 * mean_time_update / mean_time_episode)}%")
    # print(
    #     f"Mean time display during episodes : {round(mean_time_display, 2)} {int(100 * mean_time_display / mean_time_episode)}%")
    # print("\n")


def run(config_player_path, config_monster_path):
    # Load configuration.
    training_nets = []
    neat_config_player = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                     config_player_path)
    neat_config_monster = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                      neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                      config_monster_path)

    for training_number in range(1, 2):  # config.NUMBER_TRAININGS * 2 + 1
        if training_number % 2:
            type_training = "monster"
            number_generation = config.NUMBER_GEN_PER_TRAINING
            neat_config = neat_config_monster

        else:
            type_training = "player"
            number_generation = config.NUMBER_GEN_PER_TRAINING + 10
            neat_config = neat_config_player

        print(f"#### Training {type_training.capitalize()}s n°{(training_number + 1) // 2} ####")

        # Create the population or load the last checkpoint
        checkpoint_dir_path = os.path.join(os.getcwd(), f"checkpoint_{type_training}")
        if len(os.listdir(checkpoint_dir_path)):
            checkpoint_name = sorted(os.listdir(checkpoint_dir_path), key=lambda name: int(name[11:]))[-1]
            checkpoint_path = os.path.join(checkpoint_dir_path, checkpoint_name)
            p = Checkpointer.restore_checkpoint(checkpoint_path)
        else:
            print("\n WARNING : CREATING A NEW POPULATION FROM SCRATCH \n")
            p = Population(neat_config, checkpoint_dir_path)

        # Use of a lambda function to be able to give another argument
        p.run(lambda population: eval_genomes(population, training_number, training_nets),
              number_generation)

        p.genome_reporter.print_best_fitnesses(10)

        training_genomes = p.genome_reporter.best_genomes(config.NUMBER_NETS_TRAINING)
        training_nets = [neat.nn.FeedForwardNetwork.create(genome, neat_config) for genome in training_genomes]

        for i, genome in enumerate(training_genomes):
            p.genome_reporter.draw_net(neat_config, genome, f"network_genome_{i + 1}")


if __name__ == '__main__':
    os.makedirs("checkpoint_monster", exist_ok=True)
    os.makedirs("checkpoint_player", exist_ok=True)

    config_player_path = os.path.join(os.getcwd(), 'config_player.txt')
    config_monster_path = os.path.join(os.getcwd(), 'config_monster.txt')
    run(config_player_path, config_monster_path)
