import os
import pygame
import random
from random import randint
import typing
import numpy as np

import neat
from neat_modified.population import Population
from neat_modified.checkpoint_reporter import Checkpointer

from game import Game
from monster import Monster
from player import Player
from demo_game import DemoGame
import config


def eval_genomes(population: Population, training_number: int, training_nets: typing.List[neat.nn.FeedForwardNetwork]):
    """
    The function runs a simulation with the current population of monsters or players and evaluates their fitness based
    on the number of collisions they have with their opponents.

    This represents one generation. One simulation is called "episode" and there are multiple episodes in one generation
    to make sure to have a representative ranking of the genomes considering the variability of each episode (different
    starting position, and different behaviors considering their position on the screen).

    To determine whether the simulation is training the monsters or the players, the function uses the training_number
    variable. If training_number is an odd value, it indicates that the monsters are being trained. Conversely,
    if training_number is an even value, it signifies that the players are being trained.

    The training nets are the best nets of the last training of the opponents, which are used for this training. If the
    list is empty, we use "smart random" agents. (only implemented for the heroes)
    """

    # Initialization : training
    genomes, neat_config = list(population.population.items()), population.config
    nets = []
    ge = []

    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, neat_config)
        nets.append(net)
        ge.append(genome)

    # Initialization : display
    print(f"\nGENERATION {population.generation}\n")

    # Initialization : stats
    population.genome_reporter.start_generation()

    # Running the episodes

    for episode in range(1, config.MAX_NUMBER_EPISODE + 1):
        population.genome_reporter.start_episode()

        game = Game(population.generation, training_number, episode, len(ge))

        x_player = config.IMAGE_SIZE[0] * 3 #randint(0, config.WINDOW_WIDTH - config.IMAGE_SIZE[0])
        y_player = config.WINDOW_STATS_HEIGHT + config.IMAGE_SIZE[0] * 3 #randint(config.WINDOW_STATS_HEIGHT, config.WINDOW_HEIGHT - config.IMAGE_SIZE[0])
        x_monsters = config.IMAGE_SIZE[0] #randint(0, config.WINDOW_WIDTH - config.IMAGE_SIZE[0])
        y_monsters = config.WINDOW_STATS_HEIGHT + config.IMAGE_SIZE[0] #randint(config.WINDOW_STATS_HEIGHT, config.WINDOW_HEIGHT - config.IMAGE_SIZE[0])
        player = Player(x_player, y_player, len(ge))
        game.add_player(player)

        for index_entity in range(len(ge)):
            # Creating the monsters
            monster = Monster(x_monsters, y_monsters, index_entity)
            game.add_monster(monster)

            # Adding the training nets from the last training (if any)
            if training_nets:
                if training_number % 2:
                    player.set_net(training_nets[episode % config.NUMBER_NETS_TRAINING])
                else:
                    monster.set_net(
                        training_nets[episode % config.NUMBER_NETS_TRAINING]
                    )
        population.genome_reporter.set_init_time_episode()
        ge, nets, population = game.run_episode(nets, ge, population, pygame.display.get_surface())

        population.genome_reporter.end_episode(genomes)

        if episode > 1 and population.genome_reporter.compute_evolution_ranking(
                genomes) < config.THRESHOLD_EVOL_RANKING:
            break
        population.genome_reporter.ranking_id_last_episode = np.array(sorted(genomes, key=lambda g: g[1].fitness,
                                                                             reverse=True)[:20])[:, 0]

    # Taking the mean of the fitnesses
    for _, genome in genomes:
        genome.fitness /= population.genome_reporter.current_episode

    # End of the generation
    population.genome_reporter.end_generation(population.population.values())

    # Display a demo of the best genome of this generation
    # DemoGame(population.genome_reporter.best_genomes(1, False)[0], neat_config, population.generation).show_demo(1)

    # Print stats
    population.genome_reporter.get_generation_stats()
    population.genome_reporter.print_species_stats(list(population.species.species.values()))
    # population.genome_reporter.plot_fitness_repartition()
    # population.genome_reporter.print_time_stats()


def run(_config_player_path: str, _config_monster_path: str):
    """
    Run the alternative training considering a config for the players and one for the monsters
    Note : If there are any checkpoints, the configuration of the checkpoints will be used, so any changes to the config
    will not have any effect.
    """
    # Load configuration
    neat_config_player = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        _config_player_path,
    )
    neat_config_monster = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        _config_monster_path,
    )

    # Starting alternative training (training the monster on odd training numbers and the players on even)
    for training_number in range(1, 2):  # config.NUMBER_TRAININGS * 2 + 1
        if training_number % 2:
            type_training = "monster"
            number_generation = config.NUMBER_GEN_PER_TRAINING
            neat_config = neat_config_monster

        else:
            type_training = "player"
            number_generation = config.NUMBER_GEN_PER_TRAINING
            neat_config = neat_config_player

        print(
            f"#### Training {type_training.capitalize()}s n°{(training_number + 1) // 2} ####"
        )

        # Create the population or load the last checkpoint
        training_nets = []
        checkpoint_dir_path = os.path.join(os.getcwd(), f"checkpoint_{type_training}")
        if len(os.listdir(checkpoint_dir_path)):
            # There are maximum 3 checkpoints in the checkpoint directory
            checkpoint_name = sorted(
                os.listdir(checkpoint_dir_path), key=lambda name: int(name[11:])
            )[-1]
            checkpoint_path = os.path.join(checkpoint_dir_path, checkpoint_name)
            p = Checkpointer.restore_checkpoint(checkpoint_path)
        else:
            print("\n WARNING : CREATING A NEW POPULATION FROM SCRATCH \n")
            p = Population(neat_config, checkpoint_dir_path)

        # RUN THE TRAINING
        # Use of a lambda function to be able to give additional arguments
        # p.run(
        #     lambda population: eval_genomes(population, training_number, training_nets),
        #     number_generation,
        # )

        # Stats
        p.genome_reporter.print_best_fitnesses(20)
        training_genomes = p.genome_reporter.best_genomes(config.NUMBER_NETS_TRAINING)

        for i, genome in enumerate(training_genomes):
            DemoGame(genome, neat_config,
                     p.generation).show_demo(3)
            p.genome_reporter.draw_net(neat_config, genome, f"network_genome_{i + 1}")

        # Saving the best nets for the alternative training
        training_nets = [
            neat.nn.FeedForwardNetwork.create(genome, neat_config)
            for genome in training_genomes
        ]


if __name__ == "__main__":
    # Initialization
    os.makedirs("checkpoint_monster", exist_ok=True)
    os.makedirs("checkpoint_player", exist_ok=True)
    random.seed(10)
    config_player_path = os.path.join(os.getcwd(), "config_player.txt")
    config_monster_path = os.path.join(os.getcwd(), "config_monster.txt")

    # Pygame
    pygame.init()
    pygame.display.set_caption(
        "AI alternative reinforcement training using genetic algorithm"
    )
    pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
    os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"
    pygame.mouse.set_visible(False)

    # Running
    run(config_player_path, config_monster_path)
