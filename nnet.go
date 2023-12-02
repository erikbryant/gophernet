package main

import (
	"errors"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

// neuralNet contains the information that defines a trained neural network
type neuralNet struct {
	config    neuralNetConfig
	wHidden   *mat.Dense
	bHidden   *mat.Dense
	wOut      *mat.Dense
	bOut      *mat.Dense
	isTrained bool
}

// neuralNetConfig defines the neural network architecture and learning parameters
type neuralNetConfig struct {
	inputNeurons  int
	outputNeurons int
	hiddenNeurons int
	numEpochs     int
	learningRate  float64
}

// newNetwork initializes a new neural network
func newNetwork(config neuralNetConfig) *neuralNet {
	return &neuralNet{
		config:    config,
		isTrained: false,
	}
}

// randomize randomizes the weights and biases
func (nn *neuralNet) randomize() {
	// Initialize weights and biases
	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)

	wHidden := mat.NewDense(nn.config.inputNeurons, nn.config.hiddenNeurons, nil)
	bHidden := mat.NewDense(1, nn.config.hiddenNeurons, nil)
	wOut := mat.NewDense(nn.config.hiddenNeurons, nn.config.outputNeurons, nil)
	bOut := mat.NewDense(1, nn.config.outputNeurons, nil)

	wHiddenRaw := wHidden.RawMatrix().Data
	bHiddenRaw := bHidden.RawMatrix().Data
	wOutRaw := wOut.RawMatrix().Data
	bOutRaw := bOut.RawMatrix().Data

	for _, param := range [][]float64{
		wHiddenRaw,
		bHiddenRaw,
		wOutRaw,
		bOutRaw,
	} {
		for i := range param {
			param[i] = randGen.Float64()
		}
	}

	// Define our trained neural network
	nn.wHidden = wHidden
	nn.bHidden = bHidden
	nn.wOut = wOut
	nn.bOut = bOut
}

// trained returns nil if the model is trained, an error otherwise
func (nn *neuralNet) trained() error {
	if !nn.isTrained {
		return errors.New("the neural net has not been trained")
	}

	return nil
}
