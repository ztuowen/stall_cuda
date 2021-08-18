CUOPT=-arch=sm_70
CC=nvcc

all:stall kernel.cubin

stall:stall.cu
	$(CC) $(CUOPT) -lcuda -o $@ $^

kernel.cubin:kernel.cu
	$(CC) $(CUOPT) -Xptxas -O1 --cubin -o $@ $^

clean:
	-rm kernel.cubin
	-rm stall

.PHONY:all clean
