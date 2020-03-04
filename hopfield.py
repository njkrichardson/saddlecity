import numpy as np 
import matplotlib.pyplot as plt
from copy import deepcopy

class Hopfield: 

    def __init__(self, dimension : int = None): 
        self._dimension = dimension
        self.params = np.zeros((dimension, dimension)) if dimension is not None else None 
        self.max_iters = 100 

    def __repr__(self):
        if self.params is not None: 
            plt.imshow(self.params, cmap='Greys')
            plt.show()

            return (f'{self.__class__.__name__}')
        else: 
            return (f'{self.__class__.__name__}')
        
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
        self.memories = memories 
        
        for memory in self.memories: 
            self.params += np.outer(memory, memory)

        self.params *= 1 / self.memories[0].shape[0]
        np.fill_diagonal(self.params, 0)

    def decode(self, inputs : np.ndarray, max_iters : int = 100):
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
        processed inputs : np.ndarray (same shape as original inputs)
        """
        for iter in range(max_iters): 
            previous_inputs = deepcopy(inputs) 
    
            for i in range(self.dimension): 
                activation = np.dot(self.params[i], inputs)
                inputs[i] = 1 if activation >= 0 else -1 

            if np.all(previous_inputs == inputs): 
                print("converged after {} iterations".format(iter+1))
                break

        return inputs
    

        
            







