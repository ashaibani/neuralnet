package main

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
)

var (
	NN                  = NeuralNet{}
	MAX_NEURON_DISTANCE = 1024

	HIDDEN_LAYERS     = 1
	AMOUNT_OF_NEURONS = 5
	OUTPUT_NEURONS    = 2

	INPUT_WIDTH     = 1
	INPUTS          = []float64{1, 2., 3.4535, 4.5, 5.67, 6.67867, 7.876}
	EXPECTED_OUTPUT = []float64{0, 1}
)

func main() {
	NN, err := loadNeuralNetFromDumpWithInput(INPUTS, "71846214.dat")
	if err != nil {
		fmt.Println(err)
		os.Exit(0)
	}
	// NN = createNeuralNet(0.3)
	epochCount := 10

	for epoch := 0; epoch < epochCount; epoch++ {
		NN.feedForward(false)
		NN.backPropogate()
		for _, l := range NN.layers {
			tag := "HIDDEN"
			if l.id == 0 {
				tag = "INPUT"
			} else if l.id == len(NN.layers)-1 {
				tag = "OUTPUT"
			}
			fmt.Printf("{E: %d}[%s LAYER: %d] %+v \n", epoch, tag, l.id, l.outputs)
		}
		time.Sleep(1 * time.Second)
	}

	NN.saveWeights()

}

func createNeuralNet(learningRate float64) NeuralNet {
	neuralNet := NeuralNet{layers: make([]Layer, 0), learningRate: learningRate}
	inputLayer := Layer{id: 0, neurons: make([]Neuron, 0), outputs: INPUTS}
	neuralNet.addLayer(inputLayer)
	for i := 1; i <= HIDDEN_LAYERS; i++ {
		hiddenLayer := Layer{id: i, neurons: createNeurons(neuralNet.layers[i-1], AMOUNT_OF_NEURONS)}
		neuralNet.calculateOutputs(&hiddenLayer, false)
		neuralNet.addLayer(hiddenLayer)
	}

	outputLayer := Layer{id: len(neuralNet.layers), neurons: createNeurons(neuralNet.layers[len(neuralNet.layers)-1], OUTPUT_NEURONS)}
	neuralNet.calculateOutputs(&outputLayer, false)
	neuralNet.addLayer(outputLayer)
	return neuralNet
}

func createNeurons(input Layer, amount int) []Neuron {
	neurons := make([]Neuron, 0)

	for i := 0; i < amount; i++ {
		n := Neuron{id: i, bias: 1, location: generatePolarCoords(MAX_NEURON_DISTANCE)}
		n.createConnections(input)
		neurons = append(neurons, n)
	}

	return neurons
}

func (neuralNet *NeuralNet) calculateOutputs(layer *Layer, softmax bool) {
	if layer.id == 0 {
		return
	}
	layer.outputs = make([]float64, 0)

	var sum float64

	for i := 0; i < len(layer.neurons); i++ {
		neuron := &layer.neurons[i]
		neuron.output = neuron.bias
		for z := 0; z < len(neuron.connections); z++ {
			connection := neuron.connections[z]
			neuron.output += neuralNet.layers[connection.inputLayer].outputs[connection.input] * connection.weight
		}
		// can we use an independent activation function per neuron and does it effect results? e.g. define activation and its derivitive for each neuron
		// fmt.Printf("output: %.5f - sigmoid: %.5f - hyperbolic tan: %.5f \n", neuron.output, sigmoid(neuron.output), math.Tanh(neuron.output))
		neuron.output = math.Tanh(neuron.output)

		layer.outputs = append(layer.outputs, neuron.output)
		sum += math.Exp(float64(neuron.output))

	}

	if softmax {
		result := make([]float64, 0)
		// Calculate softmax value for each element
		for _, output := range layer.outputs {
			result = append(result, math.Exp(output)/sum)
		}

		layer.outputs = result
	}

	// fmt.Printf("[%d] layer outputs: %+v \n", layer.id, layer.outputs)
}

func (neuralNet *NeuralNet) feedForward(softmax bool) {
	for i := 0; i < len(neuralNet.layers); i++ {
		neuralNet.calculateOutputs(&neuralNet.layers[i], softmax && i == len(neuralNet.layers)-1)
	}
}

func (neuralNet *NeuralNet) backPropogate() {
	// check error in output layer and carry that down neural net
	for i := len(neuralNet.layers) - 1; i >= 0; i-- {
		layer := &neuralNet.layers[i]

		if i != len(neuralNet.layers)-1 {
			for j := 0; j < len(layer.neurons); j++ {
				err := 0.0
				for _, neuron := range neuralNet.layers[i+1].neurons {
					for _, conn := range neuron.connections {
						if conn.inputLayer == layer.id && conn.input == neuron.id {
							err += (conn.weight * neuron.errMargin)
						}
					}
				}

				// layer.neurons[j].errMargin = err * sigmoidPrime(layer.neurons[j].output)
				layer.neurons[j].errMargin = err * tanhDerivative(layer.neurons[j].output)
			}
		} else {
			for j := 0; j < len(layer.neurons); j++ {
				layer.neurons[j].errMargin = (layer.neurons[j].output - EXPECTED_OUTPUT[j]) * tanhDerivative(layer.neurons[j].output)
			}
		}
	}
	// backprop like a normal pleb here
	neuralNet.applyNewMargins()

	// or polar scaling here
	// neuralNet.applyPolarScaling()
}

func (neuralNet *NeuralNet) applyNewMargins() {
	for i := 1; i < len(neuralNet.layers); i++ {
		for j := 0; j < len(neuralNet.layers[i].neurons); j++ {
			n := &neuralNet.layers[i].neurons[j]
			for k := 0; k < len(neuralNet.layers[i].neurons[j].connections); k++ {
				conn := &neuralNet.layers[i].neurons[j].connections[k]
				if i > 0 {
					conn.weight -= n.errMargin * neuralNet.learningRate * neuralNet.layers[i-1].outputs[conn.input]
				} else {
					conn.weight -= n.errMargin * neuralNet.learningRate * neuralNet.layers[0].outputs[conn.input]
				}

			}
			n.bias -= n.errMargin * neuralNet.learningRate
		}
	}
}

// test effectivity: set and adjust weight based on distance in polar plane?
func (neuralNet *NeuralNet) applyPolarScaling() {
	for i := 1; i < len(neuralNet.layers); i++ {
		for j := 0; j < len(neuralNet.layers[i].neurons); j++ {
			errMargin := neuralNet.layers[i].neurons[j].errMargin
			neuralNet.layers[i].neurons[j].location.radius -= errMargin * neuralNet.learningRate * neuralNet.layers[i].neurons[j].location.radius
			for k := 0; k < len(neuralNet.layers[i].neurons[j].connections); k++ {
				conn := &neuralNet.layers[i].neurons[j].connections[k]
				if conn.inputLayer != 0 {
					conn.weight = scaledDistanceBetweenPolar(neuralNet.layers[i].neurons[j].location, neuralNet.layers[conn.inputLayer].neurons[conn.input].location, MAX_NEURON_DISTANCE)
				} else {
					conn.weight -= errMargin * neuralNet.learningRate * neuralNet.layers[0].outputs[conn.input]
				}
			}

			neuralNet.layers[i].neurons[j].bias -= errMargin * neuralNet.learningRate
		}
	}
}

func (neuralNet *NeuralNet) addLayer(layer Layer) {
	neuralNet.layers = append(neuralNet.layers, layer)
}

// test: recreating connections if weights drop too low?
func (neuron *Neuron) createConnections(input Layer) {
	neuron.connections = make([]Connection, 0)
	for i := 0; i < len(input.outputs); i++ {
		c := Connection{
			input:      i,
			inputLayer: input.id,
			weight:     rand.Float64(),
		}
		// if len(input.neurons) > i {
		// 	c.calculateWeight(*neuron, input.neurons[i])
		// }
		neuron.connections = append(neuron.connections, c)
	}
}

// for recalculating weights based on distance between neurons
func (connection *Connection) calculateWeight(self, input Neuron) {
	if input.id == -1 {
		connection.weight = rand.Float64()
	} else {
		connection.weight = scaledDistanceBetweenPolar(self.location, input.location, MAX_NEURON_DISTANCE)
	}
}

func (neuralNet *NeuralNet) saveWeights() {
	fileName := fmt.Sprintf("%d.dat", rand.Intn(100000000))
	f, err := os.Create(fileName)

	if err != nil {
		log.Fatal(err)
	}

	defer f.Close()

	// first line = neuralnet config, each line after is a layer and its data
	s := fmt.Sprintf("%f:%d\n", neuralNet.learningRate, len(neuralNet.layers))
	for _, layer := range neuralNet.layers {
		if layer.id == 0 {
			continue
		}
		s += fmt.Sprintf("%d,%s", layer.id, layer.neuronsToString())
		// splitting this by , returns all neurons. index 0 is layer id
		// splitting each neuron string by : give attributes and last attribute is list of connections.
		// splitting that by ; return all connection details
	}
	_, err2 := f.WriteString(s)

	if err2 != nil {
		log.Fatal(err2)
	}

	fmt.Printf("SAVED TO %s \n", fileName)
}

func (layer *Layer) neuronsToString() string {
	s := ""
	for _, n := range layer.neurons {
		s += fmt.Sprintf("%d:%f:%f:%f:%f:%s,", n.id, n.bias, n.errMargin, n.location.radius, n.location.radian, n.connectionsToString())
	}
	s += "\n"
	return s
}

func (n *Neuron) connectionsToString() string {
	s := ""
	for _, conn := range n.connections {
		s += fmt.Sprintf("%d;%d;%f#", conn.input, conn.inputLayer, conn.weight)
	}
	return s
}

func loadNeuralNetFromDumpWithInput(inputs []float64, file string) (NeuralNet, error) {
	NN := NeuralNet{}
	readFile, err := os.Open(file)

	if err != nil {
		return NN, err
	}

	fileScanner := bufio.NewScanner(readFile)
	fileScanner.Split(bufio.ScanLines)
	lineIndex := 0

	maxLayers := 0
	for fileScanner.Scan() {
		s := fileScanner.Text()
		if lineIndex == 0 {
			data := strings.Split(s, ":")
			if len(data) < 2 {
				return NN, fmt.Errorf("incorrect amount of parameters for neuralnet, came with %d parameters", len(data))
			}
			NN.learningRate, err = strconv.ParseFloat(data[0], 64)
			if err != nil {
				return NN, err
			}
			maxLayers, err = strconv.Atoi(data[1])
			if err != nil {
				return NN, err
			}
			inputLayer := Layer{id: 0, neurons: make([]Neuron, 0), outputs: inputs}
			NN.addLayer(inputLayer)
		} else {
			data := strings.Split(s, ",")
			layer := Layer{}
			layer.id, err = strconv.Atoi(data[0])
			if err != nil {
				return NN, err
			}
			for i := 1; i < len(data); i++ {
				neuron := Neuron{}
				neuronData := strings.Split(data[i], ":")
				if data[i] == "" {
					continue
				}
				if len(neuronData) < 6 {
					return NN, fmt.Errorf("incorrect amount of parameters for neuron, came with %d parameters", len(data))
				}
				neuron.id, err = strconv.Atoi(neuronData[0])
				if err != nil {
					return NN, err
				}
				neuron.bias, err = strconv.ParseFloat(neuronData[1], 64)
				if err != nil {
					return NN, err
				}
				neuron.errMargin, err = strconv.ParseFloat(neuronData[2], 64)
				if err != nil {
					return NN, err
				}
				radius, err := strconv.ParseFloat(neuronData[3], 64)
				if err != nil {
					return NN, err
				}
				radian, err := strconv.ParseFloat(neuronData[4], 64)
				if err != nil {
					return NN, err
				}
				neuron.location = Coordinate{radius: radius, radian: radian}

				connectionData := strings.Split(neuronData[len(neuronData)-1], "#")
				for _, connData := range connectionData {
					connection := Connection{}
					if connData == "" {
						continue
					}
					da := strings.Split(connData, ";")
					if len(da) < 3 {
						return NN, fmt.Errorf("incorrect amount of parameters for connections, came with %d parameters", len(da))
					}
					connection.input, err = strconv.Atoi(da[0])
					if err != nil {
						return NN, err
					}
					connection.inputLayer, err = strconv.Atoi(da[1])
					if err != nil {
						return NN, err
					}
					connection.weight, err = strconv.ParseFloat(da[2], 64)
					if err != nil {
						return NN, err
					}
					neuron.connections = append(neuron.connections, connection)
				}

				layer.neurons = append(layer.neurons, neuron)
			}
			NN.addLayer(layer)
		}
		lineIndex++
	}

	if len(NN.layers) < maxLayers || len(NN.layers) > maxLayers {
		return NN, fmt.Errorf("loaded an incorrect amount of layers %d", len(NN.layers))
	}

	readFile.Close()
	return NN, nil
}
