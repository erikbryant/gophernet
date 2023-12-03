package main

import (
	"gonum.org/v1/gonum/mat"
)

// backpropagate implements the backpropagation method
func (nn *neuralNet) backpropagate(x, y *mat.Dense) error {
	// Define the output of the neural network
	output := new(mat.Dense)

	// Loop over the epochs, using backpropagation to train the model
	for i := 0; i < nn.config.numEpochs; i++ {
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

		// Complete the backpropagation
		networkError := new(mat.Dense)
		networkError.Sub(y, output)

		slopeOutputLayer := new(mat.Dense)
		applySigmoidPrime := func(_, _ int, v float64) float64 { return sigmoidPrime(v) }
		slopeOutputLayer.Apply(applySigmoidPrime, output)
		slopeHiddenLayer := new(mat.Dense)
		slopeHiddenLayer.Apply(applySigmoidPrime, hiddenLayerActivations)

		dOutput := new(mat.Dense)
		dOutput.MulElem(networkError, slopeOutputLayer)
		errorAtHiddenLayer := new(mat.Dense)
		errorAtHiddenLayer.Mul(dOutput, nn.wOut.T())

		dHiddenLayer := new(mat.Dense)
		dHiddenLayer.MulElem(errorAtHiddenLayer, slopeHiddenLayer)

		// Adjust the parameters
		wOutAdj := new(mat.Dense)
		wOutAdj.Mul(hiddenLayerActivations.T(), dOutput)
		wOutAdj.Scale(nn.config.learningRate, wOutAdj)
		nn.wOut.Add(nn.wOut, wOutAdj)

		bOutAdj, err := sumAlongAxis(0, dOutput)
		if err != nil {
			return err
		}
		bOutAdj.Scale(nn.config.learningRate, bOutAdj)
		nn.bOut.Add(nn.bOut, bOutAdj)

		wHiddenAdj := new(mat.Dense)
		wHiddenAdj.Mul(x.T(), dHiddenLayer)
		wHiddenAdj.Scale(nn.config.learningRate, wHiddenAdj)
		nn.wHidden.Add(nn.wHidden, wHiddenAdj)

		bHiddenAdj, err := sumAlongAxis(0, dHiddenLayer)
		if err != nil {
			return err
		}
		bHiddenAdj.Scale(nn.config.learningRate, bHiddenAdj)
		nn.bHidden.Add(nn.bHidden, bHiddenAdj)
	}

	return nil
}

// train trains the neural network on the given dataset
func (nn *neuralNet) train(inputs, labels *mat.Dense) error {
	// Start with random weights and biases
	nn.randomize()

	// Use backpropagation to tune the weights and biases
	if err := nn.backpropagate(inputs, labels); err != nil {
		return err
	}

	nn.isTrained = true

	return nil
}
