# Nonlinear dynamical systems for game theory, neural networks, and state space models 

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![Open Source Love svg2](https://badges.frapsoft.com/os/v2/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)

This package uses ideas from nonlinear dynamical system theory to model and interrogate agent state trajectories within games, biological associative memory, and switching autoregressive processes. 

Authors: [Nick Richardson](https://github.com/njkrichardson) and Yoni Maltsman 

## Installing from git source

```
git clone https://github.com/njkrichardson/saddlecity.git
pip install -e saddlecity
```

# Background 

Linear dynamical systems are simple but powerful abstractions for modelling time-evolving processes of interest. Despite their broad application domain, often the linear constraint on the evolution of a state vector lacks the capacity to appropriately model processes with sufficiently nonlinear structure. The domain of nonlinear dynamical systems provides a theoretical framework to reason about dynamical systems in which this linear constraint has been lifted. 

This package showcases three applications of nonlinear dynamical system theory: simulations of agent state trajectories from game theory, [computational models of biological memory systems](#hopfield), and switching autoregressive processes from the signal processing/stochastic processes literature. 

## Example:  Computational models of biological (associative) memory <a name="hopfield"></a>

The Hopfield network is a fully connected, unsupervised neural network designed to act as a model of associative memory. Here we summarize the distinguishing features between conventional digital memory and associative biological memory with a passage from David MacKay's [_Information Theory, Inference, and Learning Algorithms_](https://www.inference.org.uk/itprnn/book.pdf) textbook. 

> 1. Biological memory is associative. Memory recall is content-addressable. Given a person’s name, we can often recall their face; and vice versa. Memories are apparently recalled spontaneously, not just at the request of some CPU.
> 2. Biological memory recall is error-tolerant and robust.
   >     *  Errors in the cues for memory recall can be corrected. An example asks you to recall ‘An American politician who was very intelligent and whose politician father did not like broccoli’. Many people think of president Bush – even though one of the cues contains an error.
   >     * Hardware faults can also be tolerated. Our brains are noisy lumps of meat that are in a continual state of change, with cells being damaged by natural processes, alcohol, and boxing. While the cells in our brains and the proteins in our cells are continually changing, many of our memories persist unaffected.
> 3. Biological memory is parallel and distributed, not completely distributed throughout the whole brain: there does appear to be some functional specialization – but in the parts of the brain where memories are stored, it seems that many neurons participate in the storage of multiple mem- ories.

One can use ````saddlecity```` to instantiate general Hopfield networks. 

```python
from hopfield import Hopfield

net = Hopfield() 
```

A network can be provided memories to memorize with the ```add_memories``` method. Here we add the following three memories corresponding to binary images of handwritten digits. 

<img src="https://raw.githubusercontent.com/njkrichardson/saddlecity/master/figures/mnist_memories.png" alt="drawing" height="200" width="300" class="center"/>

Recall that this is an unsupervised neural network model; and thus doesn't require any parameter estimation. We can now use the neural network to attempt to process and decode various stimuli corresponding to corrupted versions of the learned memories. 

<img src="https://raw.githubusercontent.com/njkrichardson/saddlecity/master/figures/mnist_stimulus.png" alt="drawing" height="200" width="300" class="center"/>

```python 
decoded = net.decode(stimulus) 
```

We can then visualize the processed stimulus, which has converged to the true memory corresponding to the stimulus. 

<img src="https://raw.githubusercontent.com/njkrichardson/saddlecity/master/figures/mnist_decoded.png" alt="drawing" height="200" width="300" class="center"/>

This example and more can be found in the ```examples``` directory. 

# References 

Text references and resources can be found under the ```resources``` directory. We follow [David MacKay](http://www.inference.org.uk/mackay/itila/book.html)'s text as a guiding reference on Hopfield networks, and [David Barber](http://web4.cs.ucl.ac.uk/staff/D.Barber/pmwiki/pmwiki.php?n=Brml.HomePage)'s for a reference on switching autoregressive processes (both texts are freely available online). 

We utilize [Scott Linderman](https://github.com/slinderman)'s Python [package](https://github.com/slinderman/ssm) to implement state space models and demonstrate procedures for Bayesian learning of parameters in these models.  

