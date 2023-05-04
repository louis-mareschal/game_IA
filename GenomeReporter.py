"""
Better Reporter ever created
"""

from neat.math_util import mean, stdev
from neat.reporting import BaseReporter


class GenomeReporter(BaseReporter):
    """
    Gathers (via the reporting interface) and provides (to callers and/or a file)
    the most-fit genomes and information on genomes fitness.
    """

    def __init__(self):
        BaseReporter.__init__(self)
        self.genomes = []

    def post_evaluate(self, config, population, species, best_genome):

        self.genomes.append(sorted(population.values(), key=lambda genome: genome.fitness, reverse=True))

    def print_best_fitnesses(self, nb_genomes):
        print(f"\nTop {nb_genomes} fitness per generation:")
        score_board = "  MEAN  | " + " | ".join([f"Genome_{i+1}" for i in range(nb_genomes)])
        print(score_board)
        for generation, genomes_generation in enumerate(self.genomes):
            fitness_genomes = [round(genome.fitness, 2) for genome in genomes_generation][:nb_genomes]
            mean_fitness_genomes = round(mean(fitness_genomes), 2)

            len_mean = len(str(mean_fitness_genomes))
            printing_line = " " + str(mean_fitness_genomes) + " " * (7-len_mean) + "|  "
            for fitness_genome in fitness_genomes:
                printing_line += str(fitness_genome) + " " * (8-len(str(fitness_genome))) + "|  "
            print(printing_line)
        print("\n")

    def get_fitness_stat(self, f, generation):
        if generation is not None:
            return f([genome.fitness for genome in self.genomes[generation]])
        return [f([genome.fitness for genome in genomes_generation]) for genomes_generation in self.genomes]

    def get_fitness_mean(self, generation=None):
        """Get the per-generation mean fitness."""
        return self.get_fitness_stat(mean, generation)

    def get_fitness_stdev(self, generation=None):
        """Get the per-generation standard deviation of the fitness."""
        return self.get_fitness_stat(stdev, generation)

    def best_genomes(self, n, all_generations=False):
        """Returns the n most fit genomes ever seen."""
        if all_generations:
            all_genomes = []
            for genomes_generation in self.genomes:
                all_genomes.append(genomes_generation)

            return sorted(all_genomes,
                          key=lambda genome: genome.fitness, reverse=True)[:n]

        return self.genomes[-1][:n]
