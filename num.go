package main

import (
	"math"
	"math/rand"
)

// generateCoords generates a random coordinate in our world based off of our max world width constant
func generatePolarCoords(worldLimit int) Coordinate {
	return Coordinate{radius: rand.Float64() * float64(worldLimit), radian: rand.Float64() * (2 * math.Pi)}
}

func distanceBetweenPolar(c2, c1 Coordinate) float64 {
	b := c1.radius
	c := c2.radius
	radii := c2.radian - c1.radian
	return math.Sqrt(((b * b) + (c * c)) - (2 * b * c * math.Cos(radii)))
}

func scaledDistanceBetweenPolar(c2, c1 Coordinate, maxDistance int) float64 {
	distance := distanceBetweenPolar(c1, c2)
	return 1 - (distance / float64(maxDistance))
}

func sigmoid(activation float64) float64 {
	value := 1.0 / (1.0 + math.Exp(-activation))
	// fmt.Printf("calculating sigmoid for acivation: %.4f, value: %.4f \n", activation, value)
	return value
}

func sigmoidPrime(activation float64) float64 {
	value := sigmoid(activation) * (1.0 - sigmoid(activation))
	// fmt.Printf("calculating sigmoid for acivation: %.4f, value: %.4f \n", activation, value)
	return value
}

func tanhDerivative(output float64) float64 {
	return 1 - output*output
}

func relu(activation float64) float64 {
	if activation > 0 {
		return activation
	}
	return 0
}

func reluDerivative(output float64) float64 {
	if output > 0 {
		return 1
	}
	return 0
}
