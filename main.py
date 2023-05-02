from numpy.random import randint, choice
import os
import pygame
import neat
import pickle
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

GENERATION = 0


def eval_genomes(genomes, neat_config, training_number: int, training_nets):
    """
    Runs the simulation of the current population of
    monsters or players and sets their fitness based on how many times they collide with their opponent.
    The training_number is to know if we are currently training the monsters or the players and how many trainings we
    did : if training_number is odd, it is training the monsters and if it's even, the players.
    """
    # Updating the generation number
    global GENERATION
    GENERATION += 1
    current_pop_size = neat_config.pop_size
    nets = []
    ge = []
    display_one_pair = False

    # TIME TESTING
    time_episodes = []
    time_init = []
    time_updates = []
    time_displays = []

    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, neat_config)
        nets.append(net)
        ge.append(genome)

    purged_episode = [5, 10, 15, 20]
    for episode in range(1, 30):

        time_start_episode = time.time()

        game = Game(GENERATION, training_number, episode)

        if episode in purged_episode:
            print(f"PURGING AT EPISODE {episode}")
            ge, nets = game.purge(ge, nets)
            current_pop_size = len(ge)

        x_players, y_players = randint(100, config.WINDOW_WIDTH - 100), randint(100, config.WINDOW_HEIGHT - 100)
        x_monsters, y_monsters = randint(100, config.WINDOW_WIDTH - 100), randint(100, config.WINDOW_HEIGHT - 100)
        for index_entity in range(current_pop_size):
            # Creating the monsters
            monster = Monster(x_monsters, y_monsters)
            game.add_monster(monster)
            # Creating the players
            player = Player(x_players, y_players)
            game.add_player(player)
            if training_nets:
                if training_number % 2:
                    player.set_net(choice(training_nets))
                else:
                    monster.set_net(choice(training_nets))

        running = True
        paused = False

        im_max = 1500 + (episode * 100)
        im = 0

        time_init.append(time.time() - time_start_episode)
        while running and im < im_max:
            im += 1
            time_before_update = time.time()
            ge, nets = game.update(nets, ge)
            time_updates.append(time.time() - time_before_update)

            time_before_display = time.time()
            if GENERATION % 1 == 0 and im % 40 == 0:
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
                    elif event.type == pygame.QUIT:
                        running = False
                        pygame.quit()

                if paused:
                    time.sleep(0.5)
                else:
                    looping = False
            time_displays.append(time.time() - time_before_display)

        time_episodes.append(time.time() - time_start_episode)

    if training_number % 2:
        type_training = "monster"
    else:
        type_training = "player"

    for genome_id, genome in genomes:
        genome.fitness /= neat_config.pop_size

    mean_time_episode = sum(time_episodes) / config.NUMBER_EPISODES
    mean_time_init = sum(time_init) / config.NUMBER_EPISODES
    mean_time_update = sum(time_updates) / config.NUMBER_EPISODES
    mean_time_display = sum(time_displays) / config.NUMBER_EPISODES

    print(f"TIME REPORT TRAINING {type_training} N°{training_number} GENERATION N°{GENERATION}\n")
    print(f"Mean time episodes : {round(mean_time_episode, 2)}")
    print(
        f"Mean time init section during episodes : {round(mean_time_init, 2)} {int(100 * mean_time_init / mean_time_episode)}% ")
    print(
        f"Mean time update section during episodes : {round(mean_time_update, 2)} {int(100 * mean_time_update / mean_time_episode)}%")
    print(
        f"Mean time display during episodes : {round(mean_time_display, 2)} {int(100 * mean_time_display / mean_time_episode)}%")
    print("\n")


def run(config_player_path, config_monster_path):
    # Load configuration.
    global GENERATION
    training_nets = []
    neat_config_player = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                     config_player_path)
    neat_config_monster = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                      neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                      config_monster_path)

    for training_number in range(1, config.NUMBER_TRAININGS * 2 + 1):
        if training_number % 2:
            type_training = "monster"
            number_generation = config.NUMBER_GEN_PER_TRAINING
            neat_config = neat_config_monster

        else:
            type_training = "player"
            number_generation = config.NUMBER_GEN_PER_TRAINING + 6
            neat_config = neat_config_player

        print(f"#### Training {type_training.capitalize()}s n°{(training_number + 1) // 2} ####")
        GENERATION = 0

        # Create the population or load the last checkpoint
        checkpoint_dir_path = os.path.join(os.getcwd(), f"checkpoint_{type_training}")
        if len(os.listdir(checkpoint_dir_path)):
            checkpoint_path = os.path.join(checkpoint_dir_path, os.listdir(checkpoint_dir_path)[-1])
            p = neat.Checkpointer.restore_checkpoint(checkpoint_path)
        else:
            print("\n\n WARNING : CREATING A NEW POPULATION FROM SCRATCH \n\n")
            p = neat.Population(neat_config)

        # Add a stdout reporter to show progress in the terminal.
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(neat.Checkpointer(number_generation, filename_prefix=os.path.join(checkpoint_dir_path, "checkpoint-")))

        # Use of a lambda function to be able to give another argument
        winner = p.run(lambda genomes, conf: eval_genomes(genomes, conf, training_number, training_nets), number_generation)

        training_genomes = stats.best_unique_genomes(10)
        training_nets = [neat.nn.FeedForwardNetwork.create(genome, neat_config) for genome in training_genomes]


        # Display the winning genome.
        print(f"\nBest genome training {type_training.capitalize()}s n°{(training_number + 1) // 2}: \n {winner}")

        # Saving the winner net for the monster
        winner_net = neat.nn.FeedForwardNetwork.create(winner, neat_config)
        with open(f"winner_net_{type_training}.pickle", 'wb') as net_file:
            pickle.dump(winner_net, net_file)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_player_path = os.path.join(local_dir, 'config_player.txt')
    config_monster_path = os.path.join(local_dir, 'config_monster.txt')
    run(config_player_path, config_monster_path)
