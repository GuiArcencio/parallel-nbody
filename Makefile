FLAGS := -O3 -Wall -lm
CC := gcc

all: sequential multicore gpu
	make clear

sequential: sequential.c generate.o
	$(CC) $(FLAGS) sequential.c generate.o -o sequential

multicore: multicore.c generate.o
	$(CC) $(FLAGS) -fopenmp multicore.c generate.o -o multicore

gpu: gpu.cu generate.o
	nvcc $(FLAGS) gpu.cu generate.o -o gpu

generate.o: generate.c generate.h
	$(CC) $(FLAGS) -c generate.c -o generate.o

clear:
	rm *.o
