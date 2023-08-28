build: main.go
	go build -o bin/neuralnet .

clean:
	rm bin/neuralnet

image:
	docker build -t neuralnet .

run: build
	./bin/neuralnet