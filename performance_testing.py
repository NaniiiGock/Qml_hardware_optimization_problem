import pennylane as qml
from generate_dataset import create_dataset
from generating_circuit import construct_layer, generate_layer
from pennylane_device import *
from pennylane import numpy as np
from encoding import encode
import matplotlib.pyplot as plt
from generate_dataset import create_dataset
import qiskit.providers.fake_provider
from generating_circuit import draw

# function that checks layer productivity

dev = qml.device('qiskit.aer', wires=5)
BAS, labels = create_dataset(4)

print(BAS)


def layer_test(layer):
    @qml.qnode(dev, interface="autograd")
    def circuit(image, template_weights):
        qml.AmplitudeEmbedding(image, wires=range(5), pad_with=0, normalize=True)
        construct_layer(layer, template_weights)

        # something like "measure layer" here, just to connect all the outputs
        # might not work for current layout, needs to be fixed
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 3])
        qml.CNOT(wires=[3, 4])

        out = qml.expval(qml.PauliZ(wires=4))
        return out

    params = np.random.random(3, requires_grad=True) * 6
    optimizer = qml.GradientDescentOptimizer(stepsize=0.2)

    draw(circuit, BAS[0], params)
    plt.show()

    for k in range(5):
        if k % 20 == 0:
            print(f"Step {k}, cost: {costfunc(params)}")
        params = optimizer.step(costfunc, params)
        print(params, costfunc(params))

    print(params)
    return benchmark(params)


def benchmark(params):
    y_pred = [abs(float(circuit(image, params))) for image in BAS[14:]]

    def ceil(num): return [0, 1][num > 0.5]

    y_pred = list(map(ceil, y_pred))

    print(y_pred)
    print(labels[14:])
    true = [y_pred[i]==labels[14:][i] for i in range(len(y_pred))]

    print(true)
    return sum(true)/len(true)

def costfunc(params):
    cost = 0
    for i in range(int(len(BAS) / 2)):
        cost += abs(labels[i] - circuit(BAS[i], params))
    return cost


# params = np.random.randint(10, size=8)
# edgelist = [(0, 1), (0, 2), (1, 2), (1, 4), (2, 3), (2, 4), (3, 4)]
#
# # create a batch of layers and compare their accuracies
# layers_batch = []
# n = 5
# for i in range(n_layer):
#     print("testing layer", i)
#     layers_batch.append(generate_layer(3, edgelist))
#     stats = [layer_test(layer) for layer in layers_batch]
#     print(stats)
#     print("best layer: ", layers_batch[stats.index(max(stats))])
