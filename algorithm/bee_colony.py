import random
import numpy as np
import importlib as imp
from copy import copy
from utils import tools
from random import randint
from datetime import datetime
from itertools import combinations
from tqdm import tqdm, tqdm_notebook

from utils import tools, common
from algorithm import neighbor_operator, local_search
from algorithm.base import Algorithm

class BeeColony(Algorithm):

    def __init__(self, problem):
        self.problem = problem
        # Memoria histórica para evitar rutas duplicadas
        self.tabu_memory = set() 

    @property
    def name(self):
        return 'ArtificalBeeColony_Mejorado'

    def set_params(self,
                   n_epoch=1000,
                   n_initials=20,
                   n_onlookers=10,
                   search_limit=50):

        self.history = []
        self.history_alpha = []
        self.history_betta = []
        self.n_epoch = n_epoch
        self.n_initials = n_initials
        self.n_onlookers = n_onlookers
        self.search_limit = search_limit

    @staticmethod

    def fitness(problem, solution, alpha=0.5, betta=0.5):
        c = common.compute_solution(problem, solution)
        demands = common.get_routes_demand(problem, solution)
        
        # MEJORA: Sumar TODOS los excesos de capacidad, no solo el máximo
        excesos = [max(0, d - problem['capacity']) for d in demands]
        q = sum(excesos)
        
        t = 0 
        
        # Si hay exceso, aplicamos un alpha gigante dinámicamente
        # o usamos el alpha pasado por parámetro si está escalado correctamente
        penalty = alpha * q 
        
        return 1 / (c + penalty + betta * t)

    def solve(self, alpha=0.2, betta=0.2, delta=0.01, gen_alpha=1.00, gen_betta=0.5):
        
        self.tabu_memory.clear()

        solutions = [common.generate_solution(self.problem,
                                              alpha=gen_alpha,
                                              betta=gen_betta,
                                              patience=100)
                     for _ in range(self.n_initials)]

        for sol in solutions:
            self.tabu_memory.add(tuple(sol))

        fitnesses = [self.fitness(self.problem, solution, alpha=alpha, betta=betta)
                     for solution in solutions]
        counters  = np.zeros(self.n_initials, dtype=np.int32)

        alg = local_search.LocalSearch(self.problem)
        operator  = neighbor_operator.NeighborOperator()
        
        # Usamos tqdm estándar si tqdm_notebook falla en consola
        for _ in tqdm(range(self.n_epoch), total=self.n_epoch, desc="Optimizando Rutas"):

            # FASE 1: ABEJAS EMPLEADAS
            for i, solution in enumerate(solutions):
                alg.set_params(solution, n_iter=12)
                neighbor, _ = alg.solve(only_feasible=True)
                
                sol_tuple = tuple(neighbor)
                if sol_tuple in self.tabu_memory:
                    counters[i] += 1
                    continue 
                
                self.tabu_memory.add(sol_tuple) 
                
                nfitness = self.fitness(self.problem, neighbor, alpha=alpha, betta=betta)
                if nfitness > fitnesses[i]:
                    solutions[i] = neighbor
                    fitnesses[i] = nfitness
                    counters[i]  = 0
                else:
                    counters[i] += 1

            # FASE 2: ABEJAS OBSERVADORAS
            neighborhood = [[] for _ in range(self.n_initials)]
            nn_operator  = neighbor_operator.NeighborOperator()
            f_sum  = sum(fitnesses)
            probs  = [f / f_sum  for f in fitnesses]
            
            for _ in range(self.n_onlookers):
                roulette = np.random.choice(range(len(probs)), p=probs)
                solution = solutions[roulette]
                
                for _intentos in range(5):
                    neighbor = nn_operator.random_operator(solution, patience=20)
                    if tuple(neighbor) not in self.tabu_memory:
                        break 
                        
                self.tabu_memory.add(tuple(neighbor))
                neighborhood[roulette].append(neighbor)
                
                for i, neighs in enumerate(neighborhood):
                    if common.check_capacity_criteria(self.problem, neighbor):
                        neighs.append(neighbor)

            for i, neighbors in enumerate(neighborhood):
                if neighbors:
                    fits = [self.fitness(self.problem, neighbor, alpha=alpha, betta=betta)
                            for neighbor in neighbors]
                    if max(fits) > fitnesses[i]:
                        solutions[i] = neighbors[np.argmax(fits)]
                        fitnesses[i] = max(fits)
                        counters[i]  = 0
                    else:
                        counters[i] += 1

            # FASE 3: ABEJAS EXPLORADORAS
            for i, solution in enumerate(solutions):
                if counters[i] >= self.search_limit:
                    for _intentos in range(10):
                        new_sol = nn_operator.random_operator(solution, patience=10)
                        if tuple(new_sol) not in self.tabu_memory:
                            solutions[i] = new_sol
                            self.tabu_memory.add(tuple(new_sol))
                            break
                    counters[i] = 0

            # Ajuste de penalización
            criteria = [common.check_capacity_criteria(self.problem, solution)
                        for solution in solutions]
            if sum(criteria) > (len(criteria) / 2):
                alpha -= delta
            else:
                alpha += delta

            self.history.append(1 / np.mean(fitnesses))
            self.history_alpha.append(alpha)

        for i in range(self.n_initials):
            solution = solutions[np.argmax(fitnesses)]
            if not common.check_solution(self.problem, solution):
                del(solutions[np.argmax(fitnesses)])
                del(fitnesses[np.argmax(fitnesses)])
            else:
                break
                
        return solution