package main

import (
	"fmt"
	"log"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// evaluate tests the accuracy of a neural network
func evaluate(network *neuralNet, testInputs, testLabels *mat.Dense) (float64, error) {
	// Make the predictions using the trained model
	predictions, err := network.predict(testInputs)
	if err != nil {
		return 0.0, err
	}

	// Calculate the accuracy of our model
	var truePosNeg int
	numPreds, _ := predictions.Dims()
	for i := 0; i < numPreds; i++ {
		// Get the label
		labelRow := mat.Row(nil, i, testLabels)
		var prediction int
		for idx, label := range labelRow {
			if label == 1.0 {
				prediction = idx
				break
			}
		}

		// Accumulate the true positive/negative count
		if predictions.At(i, prediction) == floats.Max(mat.Row(nil, i, predictions)) {
			truePosNeg++
		}
	}

	// Return the accuracy (subset accuracy)
	return float64(truePosNeg) / float64(numPreds), nil
}

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
	accuracy, err := evaluate(network, testInputs, testLabels)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("\nAccuracy = %0.2f\n\n", accuracy)
}
