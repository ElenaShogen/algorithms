import numpy as np

class Jaya():
    def __init__(self, domain, cost_f, population, obj): 
    # self.swarm = np.random.uniform(self.domain_a, self.domain_b, (self.swarmsize, self.dimension))
        self.population = population
        self.dimension = obj.dimension
        self.populationsize = obj.swarmsize
        self.domain = domain
        # self.population = np.random.uniform(-self.domain, self.domain, (self.populationsize, self.dimension))
        self.population_updt  = self.population.copy() 
        self.cost_f = cost_f
        self.best_value = self.cost_f(self.population).min()
        self.best = np.zeros([self.dimension]).reshape(-1, 1) 
        self.worst = np.zeros([self.dimension]).reshape(-1, 1)    

    def best_worst(self):
        f_value = self.cost_f(self.population)

        k = np.argmin(f_value)
        self.best = self.population[k]
        self.best_value = f_value.min()

        q = np.argmax(f_value)
        self.worst = self.population[q]

    def update(self):
        self.best_worst()
        r1, r2 = np.random.rand(2)
        self.population_updt = self.population + r1 * (self.best - abs(self.population)) - r2 * (self.worst - abs(self.population))
        
        # f_value = self.function_value(self.population_updt)
        b = self.cost_f(self.population_updt) <= self.cost_f(self.population) 
        self.population[b] = self.population_updt[b] 

        self.population = np.minimum(self.population,  self.domain)
        self.population = np.maximum(self.population, -self.domain)