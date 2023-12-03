package main

import (
	"encoding/csv"
	"log"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

var (
	trainingData = "irisData/train.csv"
	testData     = "irisData/test.csv"
	inputCols    = 4
	labelCols    = 3
)

func irisInputsAndLabels(fileName string) (*mat.Dense, *mat.Dense) {
	// Open the dataset file
	f, err := os.Open(fileName)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	// Create a new CSV reader reading from the opened file
	reader := csv.NewReader(f)
	reader.FieldsPerRecord = inputCols + labelCols

	// Read in all of the CSV records
	rawCSVData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	// inputsData and labelsData will hold all the
	// float values that will eventually be
	// used to form matrices
	inputsData := make([]float64, inputCols*len(rawCSVData))
	labelsData := make([]float64, labelCols*len(rawCSVData))

	// Will track the current index of matrix values
	var inputsIndex int
	var labelsIndex int

	// Sequentially move the rows into a slice of floats
	for idx, record := range rawCSVData {

		// Skip the header row
		if idx == 0 {
			continue
		}

		// Loop over the float columns
		for i, val := range record {

			// Convert the value to a float
			parsedVal, err := strconv.ParseFloat(val, 64)
			if err != nil {
				log.Fatal(err)
			}

			// Add to the labelsData if relevant
			if i == 4 || i == 5 || i == 6 {
				labelsData[labelsIndex] = parsedVal
				labelsIndex++
				continue
			}

			// Add the float value to the slice of floats
			inputsData[inputsIndex] = parsedVal
			inputsIndex++
		}
	}

	inputs := mat.NewDense(len(rawCSVData), inputCols, inputsData)
	labels := mat.NewDense(len(rawCSVData), labelCols, labelsData)

	return inputs, labels
}

// iris returns the network config and train/test data for the iris dataset
func iris() (neuralNetConfig, *mat.Dense, *mat.Dense, *mat.Dense, *mat.Dense) {
	config := neuralNetConfig{
		inputNeurons:  inputCols,
		outputNeurons: labelCols,
		hiddenNeurons: 3,
		numEpochs:     5000,
		learningRate:  0.3,
	}

	inputs, labels := irisInputsAndLabels(trainingData)
	testInputs, testLabels := irisInputsAndLabels(testData)

	return config, inputs, labels, testInputs, testLabels
}
