build: main.go
	go build -o bin/neuralnet .

clean:
	rm bin/nel

image:
	docker build -t nel .

run: build
	./bin/neuralnet