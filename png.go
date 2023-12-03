package main

import (
	"image"
	"image/draw"
	"image/png"
	"os"
)

// readPNG reads a PNG file and returns it as a grayscale image
func readPNG(filename string) (*image.Gray, error) {
	// Open the PNG file
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	// Decode the RGB PNG file into an image
	img, err := png.Decode(f)
	if err != nil {
		return nil, err
	}

	// Convert the RGB image to grayscale
	grayscale := image.NewGray(img.Bounds())
	draw.Draw(grayscale, grayscale.Bounds(), img, image.Point{}, draw.Src)

	return grayscale, nil
}

// imageToSlice converts a grayscale PNG image to a slice of floats
func imageToSlice(img *image.Gray) []float64 {
	bounds := img.Bounds()
	s := []float64{}

	// Skip the framing pixels; they are almost always just white
	frameWidth := 10
	for y := bounds.Min.Y + frameWidth; y < bounds.Max.Y-frameWidth; y++ {
		for x := bounds.Min.X + frameWidth; x < bounds.Max.X-frameWidth; x++ {
			c := img.GrayAt(y, x)
			s = append(s, float64(c.Y))
		}
	}

	return s
}

func slicePNG(filename string) ([]float64, error) {
	img, err := readPNG(filename)
	if err != nil {
		return nil, err
	}
	return imageToSlice(img), nil
}
