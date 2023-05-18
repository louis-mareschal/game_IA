"""Uses `pickle` to save and restore populations (and other aspects of the simulation state)."""

import os
import gzip
import pickle
import random


class Checkpointer:
    """
    A reporter class that performs checkpointing using `pickle`
    to save and restore populations (and other aspects of the simulation state).
    """

    def __init__(self, checkpoint_dir_path):
        """
        Saves the current state (at the end of a generation) every ``generation_interval``

        :param generation_interval: If not None, maximum number of generations between save intervals
        :type generation_interval: int or None
        :param str filename_prefix: Prefix for the filename (the end will be the generation number)
        """
        self.generation_interval = 5
        self.last_generation_checkpoint = 0
        self.checkpoint_dir_path = checkpoint_dir_path

    def end_generation(self, population):
        if (
            population.generation - self.last_generation_checkpoint
        ) >= self.generation_interval:
            self.save_checkpoint(population)
            self.last_generation_checkpoint = population.generation

    def save_checkpoint(self, population):
        """Save the current simulation state."""
        if len(os.listdir(self.checkpoint_dir_path)) > 2:
            removed_checkpoint_name = sorted(
                os.listdir(self.checkpoint_dir_path), key=lambda name: int(name[11:])
            )[0]
            os.remove(os.path.join(self.checkpoint_dir_path, removed_checkpoint_name))

        output_file_name = f"checkpoint-{population.generation}"
        output_file_path = os.path.join(self.checkpoint_dir_path, output_file_name)
        print(f"Saving checkpoint to {output_file_path}")

        with gzip.open(output_file_path, "w", compresslevel=5) as f:
            data = (population, random.getstate())
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def restore_checkpoint(filename):
        """Resumes the simulation from a previous saved point."""
        with gzip.open(filename) as f:
            population, rndstate = pickle.load(f)
            random.setstate(rndstate)
            return population
