package main

import (
	"gonum.org/v1/gonum/mat"
)

// predict makes a prediction based on a trained neural network
func (nn *neuralNet) predict(x *mat.Dense) (*mat.Dense, error) {
	if err := nn.trained(); err != nil {
		return nil, err
	}

	// Complete the feed forward process
	hiddenLayerInput := new(mat.Dense)
	hiddenLayerInput.Mul(x, nn.wHidden)
	addBHidden := func(_, col int, v float64) float64 { return v + nn.bHidden.At(0, col) }
	hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

	hiddenLayerActivations := new(mat.Dense)
	activationFcn := func(_, _ int, v float64) float64 { return actFunc(v) }
	hiddenLayerActivations.Apply(activationFcn, hiddenLayerInput)

	outputLayerInput := new(mat.Dense)
	outputLayerInput.Mul(hiddenLayerActivations, nn.wOut)
	addBOut := func(_, col int, v float64) float64 { return v + nn.bOut.At(0, col) }
	outputLayerInput.Apply(addBOut, outputLayerInput)

	// Define the output of the neural network
	output := new(mat.Dense)
	output.Apply(activationFcn, outputLayerInput)

	return output, nil
}

// correct returns true if the prediction is correct
func correct(pRow, tRow []float64) bool {
	maxI := 0
	maxVal := 0.0

	for i, val := range pRow {
		if val > maxVal {
			maxVal = val
			maxI = i
		}
	}

	return tRow[maxI] == 1.0
}

// accuracy tests the accuracy of a neural network
func (nn *neuralNet) accuracy(testInputs, testLabels *mat.Dense) (float64, error) {
	// Make the predictions using the trained model
	predictions, err := nn.predict(testInputs)
	if err != nil {
		return 0.0, err
	}

	predCount, _ := predictions.Dims()
	predCorrect := 0
	for i := 0; i < predCount; i++ {
		if correct(mat.Row(nil, i, predictions), mat.Row(nil, i, testLabels)) {
			predCorrect++
		}
	}

	return float64(predCorrect) / float64(predCount), nil
}
