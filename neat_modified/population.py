"""Implements the core evolution algorithm."""

from neat.math_util import mean
from .checkpoint_reporter import Checkpointer
from .genome_reporter import GenomeReporter
from .reporting import ReporterSet
import random


class CompleteExtinctionException(Exception):
    pass


class Population(object):
    """
    This class implements the core evolution algorithm:
        1. Evaluate fitness of all genomes.
        2. Check to see if the termination criterion is satisfied; exit if it is.
        3. Generate the next generation from the current population.
        4. Partition the new generation into species based on genetic similarity.
        5. Go to 1.
    """

    def __init__(self, config, checkpoint_dir_path):
        self.reporters = (
            ReporterSet()
        )  # Unnecessary but to avoid modifying every neat file
        self.checkpoint_reporter = Checkpointer(checkpoint_dir_path)
        self.genome_reporter = GenomeReporter()
        self.config = config
        stagnation = config.stagnation_type(config.stagnation_config, self.reporters)
        self.reproduction = config.reproduction_type(
            config.reproduction_config, self.reporters, stagnation
        )
        if config.fitness_criterion == "max":
            self.fitness_criterion = max
        elif config.fitness_criterion == "min":
            self.fitness_criterion = min
        elif config.fitness_criterion == "mean":
            self.fitness_criterion = mean
        elif not config.no_fitness_termination:
            raise RuntimeError(
                "Unexpected fitness_criterion: {0!r}".format(config.fitness_criterion)
            )

        # Create a population from scratch, then partition into species.
        self.population = self.reproduction.create_new(
            config.genome_type, config.genome_config, config.pop_size
        )
        self.species = config.species_set_type(
            config.species_set_config, self.reporters
        )
        self.generation = 1
        self.species.speciate(config, self.population, self.generation)
        random.seed(10)

    def run(self, fitness_function, number_generation):
        """
        Runs NEAT's genetic algorithm for at most n generations.  Runs for max_generation.

        The user-provided fitness_function must take only two arguments:
            1. The population as a list of (genome id, genome) tuples.
            2. The current configuration object.

        The return value of the fitness function is ignored, but it must assign
        a Python float to the `fitness` member of each genome.

        The fitness function is free to maintain external state, perform
        evaluations in parallel, etc.

        It is assumed that fitness_function does not modify the list of genomes,
        the genomes themselves (apart from updating the fitness member),
        or the configuration object.
        """

        current_generation = 1
        while current_generation < number_generation + 1:
            # Evaluate all genomes using the user-provided function.
            fitness_function(self)

            for g in self.population.values():
                if g.fitness is None:
                    raise RuntimeError(
                        "Fitness not assigned to genome {}".format(g.key)
                    )

            # Create the next generation from the current generation.
            self.population = self.reproduction.reproduce(
                self.config, self.species, self.config.pop_size, self.generation
            )

            # Check for complete extinction.
            if not self.species.species:
                raise CompleteExtinctionException()

            # Divide the new population into species.
            self.species.speciate(self.config, self.population, self.generation)

            current_generation += 1
            self.generation += 1

            if current_generation == number_generation + 1:
                self.checkpoint_reporter.save_checkpoint(self)
            else:
                self.checkpoint_reporter.end_generation(self)


