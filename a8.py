from neural import *

print("\n\nTraining XOR\n\n")
xor_training_data = [
    ([1, 1], [0]), 
    ([1, 0], [1]), 
    ([0, 1], [1]), 
    ([0, 0], [0])
    ]

# (Input, Hidden Nods ,Output)
xorn = NeuralNet(2, 1, 1)

xorn.train(xor_training_data, learning_rate= .5)

print(xorn.test_with_expected(xor_training_data))


print("<<<<<<<<<<<<<< XOR >>>>>>>>>>>>>>\n")
