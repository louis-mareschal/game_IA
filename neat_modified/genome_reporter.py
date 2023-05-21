"""
Better Reporter ever created
"""
import time
import graphviz
from tabulate import tabulate
from dataclasses import dataclass, field
from neat.math_util import mean, stdev
import numpy as np
from typing import List
import matplotlib.pyplot as plt

import config


@dataclass
class StatsTimeEpisode:
    start_time_episode: float = 0
    end_time_init: float = 0
    start_time_step: float = 0
    update_time_steps: List[float] = field(default_factory=lambda: [])
    display_time_steps: List[float] = field(default_factory=lambda: [])
    nb_steps_episode: int = 0


class GenomeReporter:
    """
    Most important Reporter of the population, it can :
    - Return the most fit genomes of any generations
    - Print some statistics about genomes fitness's or training times
    - Draw the net of a given genome
    """

    def __init__(self):
        self.genomes = []
        self.all_fitnesses = []
        self.time_generations = []
        self.stats_time_episode = StatsTimeEpisode()
        self.current_generation = 0
        self.current_episode = 0

        # Parameters to know when to end a generation
        self.ranking_id_last_episode = None

    def start_generation(self):
        self.current_generation += 1
        self.all_fitnesses.append([])
        self.time_generations.append([[], [], []])

    def end_generation(self, genomes_generation):
        sorted_genomes = sorted(
            genomes_generation, key=lambda genome: genome.fitness, reverse=True
        )
        self.genomes.append(sorted_genomes)

        self.current_episode = 0

    def start_episode(self):
        self.current_episode += 1
        # Time stats
        self.stats_time_episode.start_time_episode = time.time()

    def end_episode(self, genomes):
        # Time stats
        self.time_generations[-1][0].append(self.stats_time_episode.end_time_init)
        self.time_generations[-1][1].append(sum(self.stats_time_episode.update_time_steps))
        self.time_generations[-1][2].append(sum(self.stats_time_episode.display_time_steps))
        self.stats_time_episode = StatsTimeEpisode()
        # Fitness stats
        sum_fitness_episode = np.sum(self.all_fitnesses[-1], axis=0)
        self.all_fitnesses[-1].append([])
        for i, (genome_id, genome) in enumerate(genomes):
            if self.current_episode > 1:
                self.all_fitnesses[-1][-1].append(genome.fitness - sum_fitness_episode[i])
            else:
                self.all_fitnesses[-1][-1].append(genome.fitness)

    def set_init_time_episode(self):
        self.stats_time_episode.end_time_init = time.time() - self.stats_time_episode.start_time_episode

    def set_start_time_step(self):
        self.stats_time_episode.start_time_step = time.time()

    def set_update_time_step(self):
        self.stats_time_episode.update_time_steps.append(time.time() - self.stats_time_episode.start_time_step)

    def set_display_time_step(self):
        self.stats_time_episode.display_time_steps.append((time.time() - self.stats_time_episode.start_time_step) -
                                                          self.stats_time_episode.update_time_steps[-1])

    def print_time_stats(self):
        np_time_generations_per_episode = np.sum(np.transpose(np.array(self.time_generations), (0, 2, 1)), axis=2)
        mean_time_episode_per_generation = np.mean(np_time_generations_per_episode, axis=1)
        mean_time_init_per_generation = np.mean(np.array(self.time_generations)[:, 0], axis=1)
        mean_time_update_per_generation = np.mean(np.array(self.time_generations)[:, 1], axis=1)
        mean_time_display_per_generation = np.mean(np.array(self.time_generations)[:, 2], axis=1)

        print(f"TIME REPORT TRAINING PER GENERATION\n")
        print(f"Mean time episodes: {np.round(mean_time_episode_per_generation, 2)}")
        print(
            f"Mean time init section during episodes : {np.round(mean_time_init_per_generation, 2)} "
            f"{(100 * mean_time_init_per_generation / mean_time_episode_per_generation).astype(int)}% ")
        print(
            f"Mean time update section during episodes : {np.round(mean_time_update_per_generation, 2)} "
            f"{(100 * mean_time_update_per_generation / mean_time_episode_per_generation).astype(int)}%")
        print(
            f"Mean time display during episodes : {np.round(mean_time_display_per_generation, 2)} "
            f"{(100 * mean_time_display_per_generation / mean_time_episode_per_generation).astype(int)}%")
        print("\n")

    def print_best_fitnesses(self, nb_genomes):
        table = [["GEN", "MEAN"] + [f"Genome_{i + 1}" for i in range(nb_genomes)]]
        for generation, genomes_generation in enumerate(self.genomes):
            fitness_genomes = [
                                  round(genome.fitness, 2) for genome in genomes_generation
                              ][:nb_genomes]
            mean_fitness_genomes = round(mean(fitness_genomes), 2)
            table.append(
                [str(generation + 1), str(mean_fitness_genomes)]
                + [str(fitness) for fitness in fitness_genomes]
            )
        # Specify the table format (e.g., "plain", "simple", "grid", "fancy_grid", etc.)
        table_format = "orgtbl"

        # Generate the table
        table = tabulate(
            table, headers="firstrow", tablefmt=table_format, numalign="center"
        )

        # Print the table
        print(table)

    def compute_evolution_ranking(self, genomes):
        ranking_id_current_episode = np.array(sorted(genomes, key=lambda g: g[1].fitness, reverse=True))[:, 0]
        total_rank_changes = 0
        total_rank_changes_top = 0
        for i, genome_id in enumerate(self.ranking_id_last_episode):
            if genome_id not in ranking_id_current_episode[:20]:
                total_rank_changes_top += 1
            for j, current_genome_id in enumerate(ranking_id_current_episode):
                if genome_id == current_genome_id:
                    total_rank_changes += abs(i - j)
                    break
        print(
            f"EPISODE {self.current_episode}: Total rank changes of the 20 best genomes= {total_rank_changes} {total_rank_changes_top}")
        return total_rank_changes

    def print_species_stats(self, species_values):
        all_fitness = np.mean(self.all_fitnesses[-1], axis=0)
        fitness_range = max(1.0, max(all_fitness) - min(all_fitness))

        for species in species_values:
            mean_fitness_species = sum(species.get_fitnesses()) / len(species.members)
            adjusted_fitness = (mean_fitness_species - min(all_fitness)) / fitness_range
            print(f"Species {species.key} :")
            print(f"\tSize : {len(species.members)}")
            print(f"\tMean fitness: {sum(species.get_fitnesses()) / len(species.members)}")
            print(f"\tAdjusted Fitness : {adjusted_fitness} ")
            print(f"\tFitnesses : {sorted(species.get_fitnesses(), reverse=True)}")
            print("\n")

    def plot_fitness_repartition(self):
        nb_plots = self.current_generation + 1

        fig, ax = plt.subplots(figsize=(12, 8))

        colors = ['b', 'g', 'c', 'm', 'y', 'orange', 'purple', 'brown', 'pink', 'olive']

        for i in range(nb_plots - 1):
            ax.hist(self.all_fitnesses[-1][i], bins=30, color=colors[i], alpha=0.5, label=f"Episode {i + 1}")

        ax.hist(np.mean(self.all_fitnesses[-1], axis=0), bins=30, color='red', alpha=0.7, label='MEAN')

        ax.set_xlabel('Fitness')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Fitness Repartition Generation {len(self.genomes)}')
        ax.legend()

        plt.show()

    def get_generation_stats(self):

        best_mean_fitness = sorted(np.mean(self.all_fitnesses[-1], axis=0), reverse=True)[:10]
        mean_fitness = round(mean(best_mean_fitness), 2)
        absolute_deviation = [abs(mean_fitness_genome - mean_fitness) for mean_fitness_genome in best_mean_fitness]
        mean_absolute_deviation = round(mean(absolute_deviation), 2)
        standard_deviation = [(mean_fitness_genome - mean_fitness) ** 2 for mean_fitness_genome in best_mean_fitness]
        mean_standard_deviation = round((sum(standard_deviation) / 10) ** 0.5, 2)

        print(f"GENERATION {self.current_generation} :")
        print(f"Mean 10 best fitnesses = {mean_fitness}")
        print(f"Mean abs dev = {mean_absolute_deviation} | Mean std dev = {mean_standard_deviation}")

    def get_fitness_stat(self, f, generation):
        if generation is not None:
            return f([genome.fitness for genome in self.genomes[generation]])
        return [
            f([genome.fitness for genome in genomes_generation])
            for genomes_generation in self.genomes
        ]

    def get_fitness_mean(self, generation=None):
        """Get the per-generation mean fitness."""
        return self.get_fitness_stat(mean, generation)

    def get_fitness_stdev(self, generation=None):
        """Get the per-generation standard deviation of the fitness."""
        return self.get_fitness_stat(stdev, generation)

    def best_genomes(self, n, all_generations=True):
        """Returns the n most fit genomes ever seen."""
        if all_generations:
            all_genomes = []
            for genomes_generation in self.genomes:
                all_genomes.extend(genomes_generation)
            best_n_genomes = sorted(
                all_genomes, key=lambda genome: genome.fitness, reverse=True
            )[:n]
            return best_n_genomes
        return self.genomes[-1][:n]

    @staticmethod
    def draw_net(
            config,
            genome,
            filename,
            view=False,
            node_names=None,
            show_disabled=False,
            node_colors=None,
            fmt="png",
    ):
        """Receives a genome and draws a neural network with arbitrary topology."""

        if node_names is None:
            # node_names = {-1: "delta_y", -2: "delta_x", 0: "up_down",
            #              1: "left_right"}
            node_names = {
                -1: "player_y",
                -2: "monster_y",
                -3: "player_x",
                -4: "monster_x",
                0: "up",
                1: "down",
                2: "left",
                3: "right",
                4: "up_left",
                5: "up_right",
                6: "down_left",
                7: "down_right",

            }

        assert type(node_names) is dict

        if node_colors is None:
            node_colors = {}

        assert type(node_colors) is dict

        node_attrs = {
            "shape": "circle",
            "fontsize": "9",
            "height": "0.2",
            "width": "0.2",
        }

        dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

        inputs = set()
        for k in config.genome_config.input_keys:
            inputs.add(k)
            name = node_names.get(k, str(k))
            input_attrs = {"style": "filled", "fillcolor": "lightgreen"}
            dot.node(name, _attributes=input_attrs)

        outputs = set()
        for k in config.genome_config.output_keys:
            outputs.add(k)
            name = node_names.get(k, str(k))
            node_attrs = {"style": "filled", "fillcolor": "lightblue"}

            dot.node(name, _attributes=node_attrs)

        used_nodes = set(genome.nodes.keys())
        for n in used_nodes:
            if n in inputs or n in outputs:
                continue

            attrs = {"style": "filled", "fillcolor": "lightgray"}
            dot.node(str(n), _attributes=attrs)

        for cg in genome.connections.values():
            if cg.enabled or show_disabled:
                # if cg.input not in used_nodes or cg.output not in used_nodes:
                #    continue
                input, output = cg.key
                a = node_names.get(input, str(input))
                b = node_names.get(output, str(output))
                style = "solid" if cg.enabled else "dotted"
                color = "green" if cg.weight > 0 else "red"
                width = str(0.1 + abs(cg.weight / 5.0))
                dot.edge(
                    a,
                    b,
                    _attributes={
                        "style": style,
                        "color": color,
                        "penwidth": width,
                        "label": f"{round(cg.weight, 2)}",
                    },
                )

        dot.render(filename=filename, directory="networks", view=view, cleanup=True)
