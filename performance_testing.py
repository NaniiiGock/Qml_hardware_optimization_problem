import pennylane as qml
from generate_dataset import create_dataset
from generating_circuit import construct_layer, generate_layer
import numpy as np
from encoding import encode
import matplotlib.pyplot as plt
from generate_dataset import create_dataset
import qiskit.providers.fake_provider

# function that checks layer productivity

dev = qml.device('qiskit.aer', wires=5)
BAS, labels = create_dataset(4)

print(BAS)

def layer_test(layer):

    @qml.qnode(dev, interface="autograd")
    def circuit(image, template_weights):
        encode(image, wires=range(5))
        construct_layer(layer, template_weights)
        return qml.expval(qml.PauliZ(wires=4))

    def costfunc(params):
        cost = 0
        for i in range(int(len(BAS) / 2)):
            if ((i % 2) == 0):
                cost += circuit(BAS[i], params)
            else:
                cost -= circuit(BAS[i], params)
        return cost


    params = [1, 2, 3]
    optimizer = qml.GradientDescentOptimizer(stepsize=0.1)

    for k in range(100):
        if k % 20 == 0:
            print(f"Step {k}, cost: {costfunc(params)}")
        params = optimizer.step(costfunc, params)
    false_count = 0
    for indx, image in enumerate(BAS[14:]):
        fig, ax = qml.draw_mpl(circuit, expansion_strategy="device")(image, params)
        plt.figure(figsize=[1.8, 1.8])
        plt.imshow(np.reshape(image, [4, 4]), cmap="gray")
        if circuit(image, params) < 0:
            if indx % 2 == 0:
                false_count += 1
        else:
            if indx % 2 == 1:
                false_count += 1

        plt.title(
            f"Exp. Val. = {circuit(image, params):.0f};"
            + f" Label = {'Bars' if circuit(image, params) < 0 else 'Stripes'}",

            fontsize=8,
        )
        plt.xticks([])
        plt.yticks([])
        plt.show()


edgelist = [(0, 1), (0, 2), (1, 2), (1, 4), (2, 3), (2, 4), (3, 4)]

test_layer = generate_layer(3, edgelist)
print(test_layer)
construct_layer(test_layer, [3, 2, 1])
layer_test(test_layer)