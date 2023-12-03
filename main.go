package main

import (
	"fmt"
	"log"
)

func main() {
	config, inputs, labels, testInputs, testLabels := iris()
	config, inputs, labels, testInputs, testLabels = handwriting()

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
