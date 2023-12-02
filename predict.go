package main

import (
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// predict makes a prediction based on a trained neural network
func (nn *neuralNet) predict(x *mat.Dense) (*mat.Dense, error) {
	if err := nn.trained(); err != nil {
		return nil, err
	}

	// Define the output of the neural network
	output := new(mat.Dense)

	// Complete the feed forward process
	hiddenLayerInput := new(mat.Dense)
	hiddenLayerInput.Mul(x, nn.wHidden)
	addBHidden := func(_, col int, v float64) float64 { return v + nn.bHidden.At(0, col) }
	hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

	hiddenLayerActivations := new(mat.Dense)
	applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
	hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

	outputLayerInput := new(mat.Dense)
	outputLayerInput.Mul(hiddenLayerActivations, nn.wOut)
	addBOut := func(_, col int, v float64) float64 { return v + nn.bOut.At(0, col) }
	outputLayerInput.Apply(addBOut, outputLayerInput)
	output.Apply(applySigmoid, outputLayerInput)

	return output, nil
}

// evaluate tests the accuracy of a neural network
func (nn *neuralNet) evaluate(testInputs, testLabels *mat.Dense) (float64, error) {
	// Make the predictions using the trained model
	predictions, err := nn.predict(testInputs)
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
