def encode_image(image):

    # image = np.array
    # returns circuit

    qc = QuantumCircuit(4)
    norm_image = image / (np.linalg.norm(image))
    qc.prepare_state(norm_image)

    # display a state
    # ket = Statevector(qc)
    # display(ket.draw(output ='latex'))

    qc.decompose(reps=10).draw(output='mpl')
    return qc
