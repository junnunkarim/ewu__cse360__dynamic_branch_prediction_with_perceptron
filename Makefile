CC = gcc
CFLAGS = -Wall -Wextra -O2
TARGETS = branch_predictor *.log

all: $(TARGETS)

branch_predictor: branch_predictor.c
	$(CC) $(CFLAGS) -o branch_predictor branch_predictor.c

run_predictor: branch_predictor
	./branch_predictor traces/trace_01

run_predictor_debug: branch_predictor
	./branch_predictor traces/trace_01 --debug

run: run_predictor

run_debug: run_predictor_debug

clean:
	rm -f $(TARGETS)

.PHONY: all clean run_predictor run_predictor_debug run run_debug
