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


# quantum circuit for amplitude encoding
dev = qml.device("default.qubit", wires=5)


# function that generates algorithm
def generate_algorithm(image, edgelist, n_qubits=5):
    wires = range(n_qubits)

    # function that encodes image into a state
    def encode_image():
        qml.AmplitudeEmbedding(image, wires, pad_with=0, normalize=True)

def generate_layer(n_r, edgelist, n_qubit=5):

        # layer representation as array. in first list we will save cnots, in second - rotations
        layer = [[], []]

        n_cnot = randint(0, min(2, (n_qubit - n_r)))

        free_wires = list(range(n_qubit))

        # adding cnots
        for i in range(n_cnot):
            cnot = edgelist.pop(randint(len(edgelist)))
            print(cnot)
            layer[0].append(cnot)

        # adding rotations
        for i in range(n_r):
            rot_type = ["RY", "RZ"][0]
            # eval(f"qml.{rot_type}({params[i]}, {free_wires.pop(randint(len(free_wires)))})")
            layer[1].append(0)

        print(f"layer: {layer}\n cnots: {layer[0]} \n rs: {layer[1]}")
        return layer

    # def build_model(n_layers, params):
    #
    #     model = []
    #     model_params = []
    #
    #     while (len(params) != 0):
    #         batch_size = randint(5)
    #         model_params.append(params[:batch_size])
    #         model.append(generate_layer(params[:batch_size]))
    #         params = params[batch_size:]
    #
    #     return model, model_params

# function that constructs a layer from given array
def construct_layer(layer, params):
    print("level to build: ", layer)
    for cnot in layer[0]:
        cnot = tuple(cnot)
        print(f"cnot: {cnot}, type of cnot: {type(cnot)}")
        if len(cnot) != 0: qml.CNOT(cnot)
    for i in range(len(params)):
        qml.RY(params[i], layer[1][i])
    qml.Barrier(only_visual=True)


    # params = randint(100, size=10)
    # print("all params", + params)
    # model = build_model(5, params)
    #
    # print("model that was built:")
    # for i in model:
    #     print(i)
    #
    # count = 0
    # for i in range(len(model[0])):
    #     construct_layer(model[0][i], model[1][i])
    #     count += 1
    #     print(f"layer {count} is constructed!\n")

    layer_variants = generate_layer(4, edgelist)
    return layer_variants


# print(generate_algorithm(randint(2, size=16), edgelist))
#
# image = np.array([0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0])
# draw(generate_algorithm, image, edgelist)
# plt.show()

test_layer = generate_layer(3, edgelist)
print(test_layer)
construct_layer(test_layer, [3, 2, 1])