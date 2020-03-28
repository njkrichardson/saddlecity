import numpy as np 
import numpy.random as npr
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm

class Hopfield: 

    def __init__(self, dimension : int = None): 
        self._dimension = dimension
        self.params = np.zeros((dimension, dimension)) if dimension is not None else None 
        self.max_iters = 100 
        self.bias = 0.0

    # def __repr__(self):
    #     if self.params is not None: 
    #         plt.imshow(self.params, cmap='Greys')
    #         plt.show()

    #         return (f'{self.__class__.__name__}')
    #     else: 
    #         return (f'{self.__class__.__name__}')
        
    @property
    def dimension(self): 
        return self._dimension

    @dimension.setter 
    def dimension(self, dimension : int): 
        self._dimension = dimension
        self.params = np.zeros((dimension, dimension))

    def add_memories(self, memories : list): 
        """configures the network connections according to provided memories. 
        
        Parameters
        ----------
        memories : list of np.ndarrays
            list of memories to be "memorized" by the neural net 
        """
        self.memories = deepcopy(memories)
        base_activation = np.sum([np.sum(memory) for memory in self.memories]) / (len(self.memories) * self.dimension) 
        
        for memory in self.memories: 
            memory -= base_activation
            self.params += np.outer(memory, memory)

        self.params /= len(self.memories)
        np.fill_diagonal(self.params, 0)

    def decode(self, inputs : np.ndarray, max_iters : int = 1000, bias : float = None) -> np.ndarray:
        """processes an input stimulus by updating the states of the neurons 
        according to the Hebbian learning rule. 
        
        Parameters
        ----------
        inputs : np.ndarray
            input stimulus to be processed
        max_iters : int, optional
            maximum number of iterations to update the neuron's states (convergence is 
            detected automatically), by default 100
        
        Returns
        -------
        stimulus : np.ndarray (same shape as original inputs)
            processed stimulus 
        """
        if bias is not None: 
            self.bias = bias

        stimulus = deepcopy(inputs)
        
        # initial network energy 
        previous_energy = self.energy(stimulus)

        for i in tqdm(range(max_iters)): 
            for j in range(100): 
                # select a neuron to update
                idx = npr.randint(0, self.dimension)
                activation = np.dot(self.params[idx], stimulus) - self.bias
                stimulus[idx] = 1 if activation >= 0 else -1

            # current network energy 
            current_energy = self.energy(stimulus)

            # detect convergence 
            if previous_energy == current_energy: 
                return stimulus

            # otherwise, update previous energy 
            previous_energy = current_energy

        return stimulus

    def energy(self, state : np.ndarray) -> float: 
        return -0.5 * (state @ self.params @ state) + np.sum(state * self.bias)

    

        
            







