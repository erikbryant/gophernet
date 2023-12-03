package main

import (
	"image"
	"image/draw"
	"image/png"
	"os"
)

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

func slicePNG(img *image.Gray) []uint8 {
	bounds := img.Bounds()
	s := []uint8{}

	// Skip the framing pixels, as they are almost always just white
	frameWidth := 10
	for y := bounds.Min.Y + frameWidth; y < bounds.Max.Y-frameWidth; y++ {
		for x := bounds.Min.X + frameWidth; x < bounds.Max.X-frameWidth; x++ {
			c := img.GrayAt(y, x)
			s = append(s, c.Y)
		}
	}

	return s
}
