import pennylane as qml
from generate_dataset import create_dataset
from generating_circuit import construct_layer, generate_layer
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
        encode(image, wires=range(5))
        construct_layer(layer, template_weights)

        # something like "measure layer" here, just to connect all the outputs
        # might not work for current layout, needs to be fixed
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 3])
        qml.CNOT(wires=[3, 4])

        out = qml.expval(qml.PauliZ(wires=4))
        return out

    def costfunc(params):
        cost = 0
        for i in range(int(len(BAS) / 2)):
            cost += abs(labels[i]-circuit(BAS[i], params))
        return cost


    params = np.random.random(3, requires_grad=True)*6
    optimizer = qml.GradientDescentOptimizer(stepsize=0.2)

    draw(circuit, BAS[0], params)
    plt.show()

    for k in range(20):
        if k % 20 == 0:
            print(f"Step {k}, cost: {costfunc(params)}")
        params = optimizer.step(costfunc, params)
        print(params, costfunc(params))

    false_count = 0

    print(params)

    for indx, image in enumerate(BAS[14:]):
        #fig, ax = qml.draw_mpl(circuit, expansion_strategy="device")(image, params)
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