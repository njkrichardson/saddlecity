from hopfield import Hopfield
import numpy as np 
import numpy.random as npr 
import matplotlib.pyplot as plt
import matplotlib
try: 
    matplotlib.rcParams['text.usetex'] = True
except: 
    pass



if __name__=="__main__": 
    # stimulus size parameters 
    stimulus_width, stimulus_height = 5, 5

    # model size parameters 
    num_neurons = stimulus_width * stimulus_height
    num_memories = 5

    # instantiate a network 
    net = Hopfield(num_neurons) 

    # generate some sample memories 
    memories = [npr.choice([-1., 1.], replace=True, size=num_neurons) for _ in range(num_memories)]

    # add the memories to the network 
    net.add_memories(memories)
    
    # corrupt one of the memories 
    memory = memories[0]
    noise_std = 0.1
    noise_vec = npr.choice([-1., 1.], replace=True, size=num_neurons, p=[noise_std, 1. - noise_std])
    corrupted_memory = np.multiply(memory, noise_vec)

    # decode one of the memories 
    decoded = net.decode(corrupted_memory, bias=0.)
    
    # compute reconstruction error 
    reconstruction_error = round(np.sum(np.array(memory != decoded).astype(int))/num_neurons, 3)
    
    # show memories
    fig, axes = plt.subplots(1, num_memories)
    for i in range(num_memories): 
        axes[i].imshow(memories[i].reshape(stimulus_height, stimulus_width), cmap="Greys")
        try: 
            axes[i].set_title(r'memory $x$')
        except:
            axes[i].set_title('memory {}'.format(i+1))
        axes[i].set_yticks([])
        axes[i].set_xticks([])



    # visualize the results 
    fig, axes = plt.subplots(3, 1)
    axes[0].imshow(memory.reshape(stimulus_height, stimulus_width), cmap="Greys")
    try:
        axes[0].set_title(r'memory $x$')
    except:
        axes[0].set_title('memory x')
    axes[0].set_yticks([])
    axes[0].set_xticks([])
    axes[1].imshow(corrupted_memory.reshape(stimulus_height, stimulus_width), cmap="Greys")
    try: 
        axes[1].set_title(r'transmitted memory (stimulus) $t(s) = \widetilde{x}$')
    except:
        axes[1].set_title('transmitted memory (stimulus)')
    axes[1].set_yticks([])
    axes[1].set_xticks([])
    axes[2].imshow(decoded.reshape(stimulus_height, stimulus_width), cmap="Greys")
    try: 
        axes[2].set_title(r'decoded (processed stimulus) $\widehat{x}$' + ' reconstruction error: {}\%'.format(reconstruction_error))
    except: 
        axes[2].set_title('decoded (processed stimulus) reconstruction error: {}%'.format(reconstruction_error))
    axes[2].set_yticks([])
    axes[2].set_xticks([])
    plt.show()
