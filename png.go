package main

import (
	"image"
	"image/draw"
	"image/png"
	"os"
)

var (
	loosestBounds = image.Rectangle{
		Min: image.Point{Y: 99999, X: 99999},
		Max: image.Point{Y: 0, X: 0},
	}
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

func miny(img *image.Gray) int {
	bounds := img.Bounds()

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			if img.GrayAt(y, x).Y != 255 {
				return y
			}
		}
	}

	return bounds.Max.Y
}

func minx(img *image.Gray) int {
	bounds := img.Bounds()

	for x := bounds.Min.X; x < bounds.Max.X; x++ {
		for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
			if img.GrayAt(y, x).Y != 255 {
				return x
			}
		}
	}

	return bounds.Max.X
}

func maxy(img *image.Gray) int {
	bounds := img.Bounds()

	for y := bounds.Max.Y - 1; y >= bounds.Min.Y; y-- {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			if img.GrayAt(y, x).Y != 255 {
				return y
			}
		}
	}

	return bounds.Min.Y
}

func maxx(img *image.Gray) int {
	bounds := img.Bounds()

	for x := bounds.Max.X - 1; x >= bounds.Min.X; x-- {
		for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
			if img.GrayAt(y, x).Y != 255 {
				return y
			}
		}
	}

	return bounds.Min.X
}

// boundingBox returns the bounding dimensions of the figure
func boundingBox(img *image.Gray) image.Rectangle {
	bbox := image.Rectangle{}

	bbox.Min.Y = miny(img)
	bbox.Max.Y = maxy(img)
	bbox.Min.X = minx(img)
	bbox.Max.X = maxx(img)

	return bbox
}

// imageToSlice converts a grayscale PNG image to a slice of floats
func imageToSlice(img *image.Gray) []float64 {
	s := []float64{}

	// Use the uniform minimal bounding box
	minbbox := image.Rectangle{
		Min: image.Point{Y: 5, X: 22},
		Max: image.Point{Y: 114, X: 72},
	}

	for y := minbbox.Min.Y; y < minbbox.Max.Y; y++ {
		for x := minbbox.Min.X; x < minbbox.Max.X; x++ {
			c := img.GrayAt(y, x)
			scalar := 0.0
			if c.Y == 0 {
				scalar = 1.0
			}
			s = append(s, scalar)
		}
	}

	return s
}

func slicePNG(filename string) ([]float64, error) {
	img, err := readPNG(filename)
	if err != nil {
		return nil, err
	}

	// bbox := boundingBox(img)
	// loosestBounds.Min.Y = min(bbox.Min.Y, loosestBounds.Min.Y)
	// loosestBounds.Min.X = min(bbox.Min.X, loosestBounds.Min.X)
	// loosestBounds.Max.Y = max(bbox.Max.Y, loosestBounds.Max.Y)
	// loosestBounds.Max.X = max(bbox.Max.X, loosestBounds.Max.X)
	// fmt.Println(loosestBounds)

	return imageToSlice(img), nil
}
