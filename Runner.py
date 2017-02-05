import mnist_loader
import network
import time

class run(object):

    def __init__(self, sizes = [784, 30, 10]):
        print("Starting...");
        start = time.time()
        trainingData, validationData, testData = mnist_loader.load_data_wrapper()
        self.net = network.Network(sizes)
        (self.net).SGD(trainingData, 30, 10, 3.0, testData=testData)
        end = time.time()
        print("Done!\nTime Taken: " + str(end - start) + " Seconds");
        #(self.net).save("NNData2.txt")
