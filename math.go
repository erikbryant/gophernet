package main

import (
	"math"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// relu implements the REctified Linear Unit fcn, with clamping
func relu(x float64) float64 {
	if x < 0.0001 {
		x = 0.0
	}
	if x > 0.9999 {
		x = 1.0
	}
	return x
}

// reluPrime implements the derivative of the REctified Linear Unit fcn
func reluPrime(x float64) float64 {
	if x < 0 {
		return 0.0
	}
	return 1.0
}

// sigmoid implements the sigmoid function
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// sigmoidPrime implements the derivative of the sigmoid function
func sigmoidPrime(x float64) float64 {
	return sigmoid(x) * (1.0 - sigmoid(x))
}

func clamp(x float64) float64 {
	if math.Abs(x) < 0.0001 {
		x = 0.0
	}
	if x > 0.9999 {
		x = 1.0
	}
	return x
}

func actFunc(x float64) float64 {
	return clamp(sigmoid(x))
	// return relu(x)
}

func actFuncPrime(x float64) float64 {
	return clamp(sigmoidPrime(x))
	// return reluPrime(x)
}

// sumColumns sums a matrix's columns
func sumColumns(m *mat.Dense) *mat.Dense {
	_, numCols := m.Dims()
	var output *mat.Dense
	data := make([]float64, numCols)

	for i := 0; i < numCols; i++ {
		col := mat.Col(nil, i, m)
		data[i] = floats.Sum(col)
	}
	output = mat.NewDense(1, numCols, data)

	return output
}
