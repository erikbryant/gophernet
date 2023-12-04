package main

import (
	"fmt"
	"log"
	"os"
	"path"

	"gonum.org/v1/gonum/mat"
)

var (
	trainingDataDir = "chars/by_class/%2x/hsf_0/"
	testDataDir     = "chars/by_class/%2x/hsf_1/"
	imageLen        = 5450
	symbolCount     = 10
	symbolStart     = 0x30
	maxDatasets     = 50
)

func filesInDir(dir string) ([]string, error) {
	files, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	filenames := []string{}
	for _, file := range files {
		filenames = append(filenames, file.Name())
	}

	return filenames, nil
}

// makeLabels returns a slice with the given symbol 1.0 and the rest 0.0
func makeLabels(symbol int) []float64 {
	labels := make([]float64, symbolCount)
	labels[symbol-symbolStart] = 1.0
	return labels
}

func handwritingInputsAndLabels(dirName string) (*mat.Dense, *mat.Dense) {
	inputsData := []float64{}
	labelsData := []float64{}
	datasetsFound := 0

	for symbol := symbolStart; symbol < symbolStart+symbolCount; symbol++ {
		dir := fmt.Sprintf(dirName, symbol)
		filenames, err := filesInDir(dir)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("Loading %s...\n", dir)
		for count, f := range filenames {
			if count >= maxDatasets {
				break
			}
			filename := path.Join(dir, f)
			img, err := slicePNG(filename)
			if err != nil {
				log.Fatal(err)
			}
			if len(img) != imageLen {
				log.Fatal(fmt.Errorf("image: %s, length is: %d", filename, len(img)))
			}
			datasetsFound++
			inputsData = append(inputsData, img...)
			labelsData = append(labelsData, makeLabels(symbol)...)
		}
	}

	inputs := mat.NewDense(datasetsFound, imageLen, inputsData)
	labels := mat.NewDense(datasetsFound, symbolCount, labelsData)

	return inputs, labels
}

// handwriting returns the network config and train/test data for the handwriting dataset
func handwriting() (neuralNetConfig, *mat.Dense, *mat.Dense, *mat.Dense, *mat.Dense) {
	config := neuralNetConfig{
		inputNeurons:  imageLen,
		outputNeurons: symbolCount,
		hiddenNeurons: imageLen,
		numEpochs:     400,
		learningRate:  0.3,
	}

	inputs, labels := handwritingInputsAndLabels(trainingDataDir)
	testInputs, testLabels := handwritingInputsAndLabels(testDataDir)

	return config, inputs, labels, testInputs, testLabels
}
