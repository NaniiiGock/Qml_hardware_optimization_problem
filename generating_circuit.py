import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import qiskit
import pennylane as qml
from qiskit.circuit.library import StatePreparation
from numpy.random import randint

# G = nx.Graph([(0, 1), (0, 2), (1, 2), (1, 4), (2, 3), (2, 4), (3, 4)])
edgelist = [(0, 1), (0, 2), (1, 2), (1, 4), (2, 3), (2, 4), (3, 4)]


# nx.draw(G, with_labels=True)
# plt.show()

def draw(circuit, *args, **kwargs):
    qml.draw_mpl(circuit, style="solarized_dark", decimals=2)(*args, **kwargs)
    plt.gcf().set_dpi(50)


def generate_random_graph(n):
    G = nx.erdos_renyi_graph(n, 0.5, seed=123, directed=False)
    return G


# function that encodes an image into a state
def encode_image(image):
    qml.AmplitudeEmbedding(image, wires, pad_with=0, normalize=True)


# function that generates a layer from given num of params and edgelist
def generate_layer(n_params, edgelist, n_qubit=5):

    # layer representation as array. in first list we will save cnots, in second - rotations
    layer = [[], []]

    n_cnot = randint(0, min(2, (n_qubit - n_params)))

    free_wires = list(range(n_qubit))

    # adding cnots
    for i in range(n_cnot):
        cnot = edgelist.pop(randint(len(edgelist)))
        print(cnot)
        layer[0].append(cnot)

    # adding rotations
    for i in range(n_params):
        rot_type = ["RY", "RZ"][0]
        # eval(f"qml.{rot_type}({params[i]}, {free_wires.pop(randint(len(free_wires)))})")
        layer[1].append(free_wires.pop(randint(len(free_wires))))

    # print(f"layer: {layer}\n cnots: {layer[0]} \n rs: {layer[1]}")
    return layer


# function that constructs a layer from given array
def construct_layer(layer, params):
    # print("level to build: ", layer)
    for cnot in layer[0]:
        cnot = tuple(cnot)
        # print(f"cnot: {cnot}, type of cnot: {type(cnot)}")
        if len(cnot) != 0: qml.CNOT(cnot)
    for i in range(len(params)):
        qml.RY(params[i], layer[1][i])
    qml.Barrier(only_visual=True)

    layer_variants = generate_layer(4, edgelist)
    return layer_variants

# print(generate_algorithm(randint(2, size=16), edgelist))
#
# image = np.array([0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0])
# draw(generate_algorithm, image, edgelist)
# plt.show()