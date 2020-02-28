from neural_network import neuralNetwork
import numpy as np

# number of input, hidden and output nodes
input_nodes = 784  # 28*28
hidden_nodes = 100
output_nodes = 10

# learning rate is 0.3
learning_rate = 0.3

# create instance of neural network
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

save_wih_file = "save_model/wih.npy"
save_who_file = "save_model/who.npy"
n.wih = np.load(save_wih_file)
n.who = np.load(save_who_file)

# load the mnist test data csv file into a list
test_data_file = open("mnist_dataset/mnist_test_10.csv", "r")
test_data_list = test_data_file.readlines()
test_data_file.close()

# test the neural network
# scorecard for how well the network performs, initially empty
scorecard = []
# go through all records in the test data set
for record in test_data_list:
    # split the record by ',' commas
    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])
    # print(correct_label, "correct_label")
    # scale and shift the inputs
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # query the network
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label
    label = np.argmax(outputs)
    # print(label, "network's answer")
    # append correct or incorrect to list
    if label == correct_label:
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        # network's answer doesn't match correct answer, add 1 to scorecard
        scorecard.append(0)

# calculate the performance score, the fraction of correct answers
scorecard_array = np.asarray(scorecard)
performance = scorecard_array.sum() / scorecard_array.size
print("performance = ", performance)

save_accuracy = "accuracy.txt"
with open(save_accuracy, 'a') as f:
    f.write("Learning rate is : " + str(n.lr))
    f.write("       Accuracy is : " + str(performance) + '\n')