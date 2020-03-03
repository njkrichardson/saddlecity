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

This package showcases three applications of nonlinear dynamical system theory: simulations of agent state trajectories from game theory, computational models of biological memory systems, and switching autoregressive processes from the signal processing/stochastic processes literature. 

# References 

Text references and resources can be found under the ```resources``` directory. We follow [David MacKay](http://www.inference.org.uk/mackay/itila/book.html)'s text as a guiding reference on Hopfield networks, and [David Barber](http://web4.cs.ucl.ac.uk/staff/D.Barber/pmwiki/pmwiki.php?n=Brml.HomePage)'s for a reference on switching autoregressive processes (both texts are freely available online). 

We utilize [Scott Linderman](https://github.com/slinderman)'s Python [package](https://github.com/slinderman/ssm) to implement state space models and demonstrate procedures for Bayesian learning of parameters in these models.  

