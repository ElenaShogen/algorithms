import numpy as np
import random
import sklearn.metrics

class NeuroNet:
    def __init__(self, n_feat, m_out, swarmsize, dw, up, hid=5, lmbd= 0.01):
        self.n_feat = n_feat
        self.m_out = m_out
        self.hid = hid
        self.lmbd = lmbd
        self.swarmsize = swarmsize
        self.dw = dw
        self.up = up
        self.dimension = self.n_feat * self.hid + 1 * self.hid + self.hid * self.m_out + 1 * self.m_out 
        
    def initialize_particle(self, x):
        # random.seed(63)
        x = np.random.uniform(self.dw, self.up, (1, self.dimension))
        return x 
    
    def initialize_swarm(self):
        x = np.zeros([self.swarmsize, self.dimension]) # матрица, состоящая из частиц (количество строк; одна строка - одна частица со своими весами и биас) и весов+биас (количество столбцов)
        p_swarm = [self.initialize_particle(x[i]) for i in range(self.swarmsize)]
        return np.array(p_swarm)
    
    def sigmoida(self, x, w, bias):
        summa  = np.dot(x, w) + bias
        output = 1 / (1 + np.exp(-0.5* summa))
        return output
        
    def call_sigmoida(self, x, p):
        w1 = p[:,0:self.n_feat*self.hid].reshape(self.n_feat, self.hid)
        bias1 = p[:,self.n_feat*self.hid:self.n_feat*self.hid+self.hid].reshape(1, self.hid)
        w2 = p[:,self.n_feat*self.hid+self.hid:self.n_feat*self.hid+self.hid+self.hid*self.m_out].reshape(self.hid, self.m_out)
        bias2 = p[:,self.n_feat*self.hid+self.hid+self.hid*self.m_out:self.n_feat*self.hid+self.hid+self.hid*self.m_out+self.m_out].reshape(1, self.m_out)
        out1 = self.sigmoida(x, w1, bias1)
        out2 = self.sigmoida(out1, w2, bias2)
        return out2

    def error(self, y, out):
        e = (y - out)**2
        return np.array(e.mean())
    
    def converter_to_bin(self, out_x):
        array_bin = np.zeros(self.m_out, dtype=np.int8)
        ind = out_x[0].argmax()
        array_bin[ind] = 1
        return list(array_bin)
       
    def err_for_each_particle(self, x_set, y_set, swarm): 
        err_each = [] 
        for s in swarm:
            qq = self.call_sigmoida(x_set, s)
            qqq = sklearn.metrics.mean_squared_error(y_set, qq) + self.lmbd * (s**2).sum()
            err_each.append(qqq)
        return np.array(err_each) 
        
    def predict(self, x, p): 
        out2 = self.call_sigmoida(x, p)
        out = self.converter_to_bin(out2)
        return out