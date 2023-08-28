package main

type NeuralNet struct {
	layers       []Layer
	learningRate float64
}

type Layer struct {
	id      int
	neurons []Neuron
	outputs []float64
}

type Neuron struct {
	id          int
	connections []Connection
	output      float64
	bias        float64
	location    Coordinate
	errMargin   float64
}

type Connection struct {
	input      int
	inputLayer int
	weight     float64
}

type Coordinate struct {
	radius float64
	radian float64
}
