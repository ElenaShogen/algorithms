import numpy as np

class ParticleSwarm():
    def __init__(self,  x_max, v_max, swarm, cost_f, dimension, swarmsize, 
                alpha = 0.72984, #0.72, 0.3925
                beta = 2.05, #1.5, 2.5586, 2.05
                gamma = 2.05): #1.5, 1.3358, 2.05
        self.swarm = swarm
        self.dimension = dimension
        self.swarmsize = swarmsize
        self.cost_f = cost_f
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.x_max = x_max
        self.v_max = v_max
        self.bestLocal  = self.swarm.copy() # лучшие положения частиц
        self.bestGlobal = np.zeros([self.dimension]).reshape(-1, 1) # вектор с наилучшими координатами среди всех частиц
        self.velocity = np.zeros_like(self.swarm)
        self.f_best = self.cost_f(self.bestLocal) 
        self.best = self.f_best.min()

    def rate_particles(self):
        self.f_best = self.cost_f(self.bestLocal) 
        b = self.f_best >= self.cost_f(self.swarm) 
        self.bestLocal[b] = self.swarm[b] #из каждой строки "bestLocal" и "swarm" формируем массив с индексами из "b"

        k = np.argmin(self.f_best)
        self.bestGlobal = self.bestLocal[k]
        self.best = self.f_best.min()
     
        r1, r2 = np.random.rand(2)
        self.velocity = self.alpha * self.velocity + self.beta * r1 * (self.bestLocal - self.swarm) + self.gamma * r2 * (self.bestGlobal - self.swarm)
        self.velocity = np.minimum(self.velocity,  self.v_max)
        self.velocity = np.maximum(self.velocity, -self.v_max)

        self.swarm = self.swarm + self.velocity #перемещение частицы
        self.swarm = np.minimum(self.swarm,  self.x_max)
        self.swarm = np.maximum(self.swarm, -self.x_max)