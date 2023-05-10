"""
Better Reporter ever created
"""
import graphviz
from neat.math_util import mean, stdev
from neat.reporting import BaseReporter


class GenomeReporter():
    """
    Gathers (via the reporting interface) and provides (to callers and/or a file)
    the most-fit genomes and information on genomes fitness.
    """

    def __init__(self):
        self.genomes = []

    def end_generation(self, genomes_generation):
        sorted_genomes = sorted(genomes_generation, key=lambda genome: genome.fitness, reverse=True)
        self.genomes.append(sorted_genomes)

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

    def best_genomes(self, n, all_generations=True):
        """Returns the n most fit genomes ever seen."""
        if all_generations:
            all_genomes = []
            for genomes_generation in self.genomes:
                all_genomes.extend(genomes_generation)
            best_n_genomes = sorted(all_genomes, key=lambda genome: genome.fitness, reverse=True)[:n]
            return best_n_genomes

        return self.genomes[-1][:n]

    def draw_net(self, config, genome, filename, view=True, node_names=None, show_disabled=False,
                 node_colors=None, fmt='png'):
        """ Receives a genome and draws a neural network with arbitrary topology. """

        if node_names is None:
            node_names = {-1: "d_x", -2: "d_y", -3: "x", -4: "y", -5: "d_x_wall", -6: "d_y_wall", 0: "up_down",
                          1: "left_right"}

        assert type(node_names) is dict

        if node_colors is None:
            node_colors = {}

        assert type(node_colors) is dict

        node_attrs = {
            'shape': 'circle',
            'fontsize': '9',
            'height': '0.2',
            'width': '0.2'}

        dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

        inputs = set()
        for k in config.genome_config.input_keys:
            inputs.add(k)
            name = node_names.get(k, str(k))
            input_attrs = {'style': 'filled', 'fillcolor': 'lightgreen'}
            dot.node(name, _attributes=input_attrs)

        outputs = set()
        for k in config.genome_config.output_keys:
            outputs.add(k)
            name = node_names.get(k, str(k))
            node_attrs = {'style': 'filled', 'fillcolor': 'lightblue'}

            dot.node(name, _attributes=node_attrs)

        used_nodes = set(genome.nodes.keys())
        for n in used_nodes:
            if n in inputs or n in outputs:
                continue

            attrs = {'style': 'filled', 'fillcolor': 'lightgray'}
            dot.node(str(n), _attributes=attrs)

        for cg in genome.connections.values():
            if cg.enabled or show_disabled:
                # if cg.input not in used_nodes or cg.output not in used_nodes:
                #    continue
                input, output = cg.key
                a = node_names.get(input, str(input))
                b = node_names.get(output, str(output))
                style = 'solid' if cg.enabled else 'dotted'
                color = 'green' if cg.weight > 0 else 'red'
                width = str(0.1 + abs(cg.weight / 5.0))
                dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width,
                                            'label': f"{round(cg.weight,2)}"})

        dot.render(filename=filename, directory="networks", view=view, cleanup=True)

        #return dot
