all:
	g++ histogram.cpp -o histogram -lOpenCL
clean:
	rm -f histogram