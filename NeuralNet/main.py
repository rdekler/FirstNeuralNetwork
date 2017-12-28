

import NeuralNetwork
import numpy


def main() :
    scorecard = []

    innodes = 784
    hidnodes = 200
    outnodes = 10
    alpha = 0.2
    epochs = 5

    nn = NeuralNetwork.NeuralNetwork(innodes , hidnodes , outnodes , alpha)

    train_data_file = open("data/mnist_train.csv" , "r")
    train_data_list = train_data_file.readlines()
    train_data_file.close()

    for e in range(epochs) :
        for record in train_data_list :
            all = record.split(",")
            inputs = (numpy.asfarray(all[1:]) / 255.0 * 0.99) + 0.01
            targets = numpy.zeros(outnodes) + 0.01
            targets[int(all[0])] = 0.99

            nn.train(inputs , targets)

    test_data_file = open("data/mnist_test.csv" , "r")
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    for record in test_data_list :
        all = record.split(",")

        correct = int(all[0])

        inputs = (numpy.asfarray(all[1:]) / 255.0 * 0.99) + 0.01
        outputs = nn.query(inputs)
        label = numpy.argmax(outputs)

        if(label == correct) :
            scorecard.append(1)
        else :
            scorecard.append(0)


    array = numpy.asarray(scorecard)
    print(array.sum() / array.size)


main()