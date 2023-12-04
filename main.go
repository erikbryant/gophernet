package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"runtime/pprof"
)

var (
	cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")
)

func main() {
	flag.Parse()
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	config, inputs, labels, testInputs, testLabels := iris()

	// The neural network is not scalable enough to handing the
	// handwriting images. The network would need at least one
	// more hidden layer and would need hours (days?) of training
	// time to generate a reliable model.
	config, inputs, labels, testInputs, testLabels = handwriting()

	network := newNetwork(config)
	if err := network.train(inputs, labels); err != nil {
		log.Fatal(err)
	}

	// Evaluate the neural network's prediction accuracy
	accuracy, err := network.accuracy(testInputs, testLabels)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("\nAccuracy = %0.2f\n\n", accuracy)
}
