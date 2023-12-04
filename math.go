package main

import (
	"errors"
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
}

func actFuncPrime(x float64) float64 {
	return clamp(sigmoidPrime(x))
}

// sumAlongAxis sums a matrix along one dimension, preserving the other
func sumAlongAxis(axis int, m *mat.Dense) (*mat.Dense, error) {
	numRows, numCols := m.Dims()

	var output *mat.Dense

	switch axis {
	case 0:
		data := make([]float64, numCols)
		for i := 0; i < numCols; i++ {
			col := mat.Col(nil, i, m)
			data[i] = floats.Sum(col)
		}
		output = mat.NewDense(1, numCols, data)
	case 1:
		data := make([]float64, numRows)
		for i := 0; i < numRows; i++ {
			row := mat.Row(nil, i, m)
			data[i] = floats.Sum(row)
		}
		output = mat.NewDense(numRows, 1, data)
	default:
		return nil, errors.New("invalid axis, must be 0 or 1")
	}

	return output, nil
}
