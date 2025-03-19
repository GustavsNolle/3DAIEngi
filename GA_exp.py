import numpy as np
import copy
import matplotlib.pyplot as plt
from Sim import PhysicsEnv2DAdvanced
from tqdm import tqdm

class PEnv(PhysicsEnv2DAdvanced):

    def reset(self, design=None):
        """
        Resets the environment and, if provided, injects a candidate metal design.
        The candidate 'design' should be a Boolean numpy array of shape (height, width).
        """
        obs = super().reset()
        if design is not None:
            if design.shape != (self.height, self.width):
                raise ValueError("Design shape must match environment dimensions.")
            self.metal_mask = design.copy()
            # Initialize metal temperature to current gas temperature in cells with metal.
            self.T_m = np.where(design, self.T_g, self.T_m)
            obs = self._get_obs()
        return obs

    def step(self, action):
        """
        Overrides step to allow a no-operation step when the action tuple is (-1, -1, -1).
        """
        if action == (-1, -1, -1):
            return super().step((-1, -1, -1))
        else:
            return super().step(action)

    def evaluate_design(self, num_steps=50, target_temp=300.0, target_pressure=300.0):
        """
        Runs the simulation for a fixed number of no-op steps and returns a fitness score.
        Fitness is computed as the negative sum of deviations from the target gas temperature,
        gas pressure, and (where metal is present) metal temperature.
        """
        for _ in range(num_steps):
            self.step((-1, -1, -1))
        mean_gas_temp = np.mean(self.T_g)
        deviation_temp = abs(mean_gas_temp - target_temp)
        mean_pressure = np.mean(self.P)
        deviation_pressure = abs(mean_pressure - target_pressure)
        if np.any(self.metal_mask):
            mean_metal_temp = np.mean(self.T_m[self.metal_mask])
            deviation_metal = abs(mean_metal_temp - target_temp)
        else:
            deviation_metal = 0
        error = deviation_temp + deviation_pressure + deviation_metal
        fitness = -error  # Lower error implies higher fitness.
        return fitness


# --- Genetic Algorithm to Evolve Metal-Structure Designs ---
def initialize_population(pop_size, height, width, metal_prob=0.1):
    population = []
    for _ in range(pop_size):
        candidate = np.random.rand(height, width) < metal_prob
        population.append(candidate)
    return population

def tournament_selection(population, fitnesses, tournament_size=3):
    selected_indices = np.random.choice(len(population), tournament_size, replace=False)
    best_index = selected_indices[0]
    best_fitness = fitnesses[best_index]
    for idx in selected_indices:
        if fitnesses[idx] > best_fitness:
            best_index = idx
            best_fitness = fitnesses[idx]
    return population[best_index]

def crossover(parent1, parent2, crossover_rate=0.5):
    mask = np.random.rand(*parent1.shape) < crossover_rate
    child = np.where(mask, parent1, parent2)
    return child

def mutate(candidate, mutation_rate=0.01):
    mutation_mask = np.random.rand(*candidate.shape) < mutation_rate
    candidate[mutation_mask] = ~candidate[mutation_mask]
    return candidate

def genetic_algorithm(pop_size=20, generations=2, height=200, width=100,
                      metal_prob=0.1, tournament_size=3,
                      crossover_rate=0.5, mutation_rate=0.01, num_steps=50):
    population = initialize_population(pop_size, height, width, metal_prob)
    best_candidate = None
    best_fitness = -np.inf

    for gen in tqdm(range(generations)):
        fitnesses = []
        for candidate in tqdm(population):
            env = PEnv(width=width, height=height)
            env.reset(design=candidate)
            fitness = env.evaluate_design(num_steps=num_steps)
            fitnesses.append(fitness)
            if fitness > best_fitness:
                best_fitness = fitness
                best_candidate = candidate.copy()
        print(f"Generation {gen}: Best Fitness = {best_fitness:.4f}")
        new_population = []
        for _ in range(pop_size):
            parent1 = tournament_selection(population, fitnesses, tournament_size)
            parent2 = tournament_selection(population, fitnesses, tournament_size)
            child = crossover(parent1, parent2, crossover_rate)
            child = mutate(child, mutation_rate)
            new_population.append(child)
        population = new_population

    return best_candidate, best_fitness

if __name__ == "__main__":
    best_design, best_fit = genetic_algorithm()
    print("Best design fitness:", best_fit)
    plt.figure(figsize=(6, 8))
    plt.imshow(best_design, cmap='gray', origin='lower')
    plt.title('Best Metal Structure Design')
    plt.colorbar(label='Metal Presence (True/False)')
    plt.show()
