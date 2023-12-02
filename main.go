package main

import (
	"fmt"
	"log"
)

func main() {
	// Load the training and testing matrices
	inputs, labels := makeInputsAndLabels(trainingData)
	testInputs, testLabels := makeInputsAndLabels(testData)

	// Create and train the neural network
	config := neuralNetConfig{
		inputNeurons:  4,
		outputNeurons: 3,
		hiddenNeurons: 3,
		numEpochs:     5000,
		learningRate:  0.3,
	}
	network := newNetwork(config)
	if err := network.train(inputs, labels); err != nil {
		log.Fatal(err)
	}

	// Evaluate the neural network
	accuracy, err := network.evaluate(testInputs, testLabels)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("\nAccuracy = %0.2f\n\n", accuracy)
}
