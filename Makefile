CC = gcc
CFLAGS = -Wall -Wextra -O2
TARGETS = branch_predictor

all: $(TARGETS)

branch_predictor: branch_predictor.c
	$(CC) $(CFLAGS) -o branch_predictor branch_predictor.c

run_predictor: branch_predictor
	./branch_predictor traces/trace_01

run: run_predictor

clean:
	rm -f $(TARGETS)

.PHONY: all clean run_predictor run
