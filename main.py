import os
import pygame
import random
from random import randint
import typing
import numpy as np

import neat
from neat_modified.population import Population
from neat_modified.checkpoint_reporter import Checkpointer
from neat_modified.feed_forward import FeedForwardNetwork

from game import Game
from monster import Monster
import config


def eval_genomes(population: Population):
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
        net = FeedForwardNetwork.create(genome, neat_config)
        nets.append(net)
        ge.append(genome)

    # Initialization : display
    print(f"\nGENERATION {population.generation}\n")

    # Initialization : stats
    population.genome_reporter.start_generation()

    # Running the episodes

    for episode in range(1, config.MAX_NUMBER_EPISODE + 1):
        population.genome_reporter.start_episode()

        game = Game(population.generation, episode)

        index_entity = 0
        while len(game.all_monsters) < len(ge):
            # Creating the monsters
            x_monster = randint(0, config.WINDOW_WIDTH - config.IMAGE_SIZE[0])
            y_monster = randint(config.WINDOW_STATS_HEIGHT, config.WINDOW_HEIGHT - config.IMAGE_SIZE[0])
            monster = Monster(x_monster, y_monster, index_entity)
            if pygame.sprite.spritecollide(monster, game.all_monsters, False):
                continue
            game.add_monster(monster)
            index_entity += 1

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


def run(_config_monster_path: str):
    """
    Run the alternative training considering a config for the players and one for the monsters
    Note : If there are any checkpoints, the configuration of the checkpoints will be used, so any changes to the config
    will not have any effect.
    """
    # Load configuration
    neat_config_monster = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        _config_monster_path,
    )
    # Create the population or load the last checkpoint
    checkpoint_dir_path = os.path.join(os.getcwd(), f"checkpoint_monster")
    if len(os.listdir(checkpoint_dir_path)):
        # There are maximum 3 checkpoints in the checkpoint directory
        checkpoint_name = sorted(
            os.listdir(checkpoint_dir_path), key=lambda name: int(name[11:])
        )[-1]
        checkpoint_path = os.path.join(checkpoint_dir_path, checkpoint_name)
        p = Checkpointer.restore_checkpoint(checkpoint_path)
    else:
        print("\n WARNING : CREATING A NEW POPULATION FROM SCRATCH \n")
        p = Population(neat_config_monster, checkpoint_dir_path)

        # RUN THE TRAINING
        # Use of a lambda function to be able to give additional arguments
        p.run(eval_genomes, config.NUMBER_GEN)

        # Stats
        p.genome_reporter.print_best_fitnesses(20)
        training_genomes = p.genome_reporter.best_genomes(config.NUMBER_NETS_TRAINING)

        for i, genome in enumerate(training_genomes):
            p.genome_reporter.draw_net(neat_config_monster, genome, f"network_genome_{i + 1}")


if __name__ == "__main__":
    # Initialization
    os.makedirs("checkpoint_monster", exist_ok=True)
    random.seed(10)
    config_monster_path = os.path.join(os.getcwd(), "config_monster.txt")

    # Pygame
    pygame.init()
    pygame.display.set_caption(
        "AI reinforcement training using genetic algorithm"
    )
    pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
    os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"
    pygame.mouse.set_visible(False)

    # Running
    run(config_monster_path)
