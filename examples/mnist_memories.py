from data import mnist, binarize
from hopfield import Hopfield
import numpy as np 
import numpy.random as npr 
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True

if __name__=="__main__": 
    # mnist image parameters 
    image_width, image_height = 28, 28 
    vectorized_image_length = image_height * image_width

    # model size parameters 
    num_neurons = vectorized_image_length
    num_memories = 3

    # instantiate a network 
    net = Hopfield(num_neurons) 

    # generate some sample memories 
    images, _, _, _ = mnist() 
    images = images.reshape(-1, vectorized_image_length)
    memories = [binarize(images[i]).ravel() for i in range(num_memories)]

    # add the memories to the network 
    net.add_memories(memories)
    
    # corrupt one of the memories 
    memory = memories[0]
    noise_std = 0.2
    noise_vec = npr.choice([-1., 1.], replace=True, size=num_neurons, p=[noise_std, 1. - noise_std])
    corrupted_memory = np.multiply(memory, noise_vec)

    # decode one of the memories 
    decoded = net.decode(memory)
    
    # compute reconstruction error 
    reconstruction_error = round(np.sum(np.array(memory != decoded).astype(int))/num_neurons, 3)

    # visualize the results 
    fig, axes = plt.subplots(3, 1)
    axes[0].imshow(memory.reshape(image_height, image_width), cmap="Greys")
    axes[0].set_title(r'memory $x$')
    axes[0].set_yticks([])
    axes[0].set_xticks([])
    axes[1].imshow(corrupted_memory.reshape(image_height, image_width), cmap="Greys")
    axes[1].set_title(r'transmitted memory (stimulus) $t(s) = \widetilde{x}$')
    axes[1].set_yticks([])
    axes[1].set_xticks([])
    axes[2].imshow(decoded.reshape(image_height, image_width), cmap="Greys")
    axes[2].set_title(r'decoded (processed stimulus) $\widehat{x}$' + ' reconstruction error: {}\%'.format(reconstruction_error))
    axes[2].set_yticks([])
    axes[2].set_xticks([])
    plt.show()