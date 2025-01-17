#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// configurable parameters
#define NUM_PERCEPTRONS 1024 // increased number of perceptrons
#define HISTORY_LENGTH 64    // increased history length
#define THETA                                                                  \
  (int)(2.14 * HISTORY_LENGTH +                                                \
        20.58)         // dynamic threshold based on history length
#define MAX_WEIGHT 127 // weight saturation bounds
#define MIN_WEIGHT -128
#define PATH_HISTORY_MASK 0xF

// improved hash function parameters
#define FNV_PRIME 16777619
#define FNV_OFFSET_BASIS 2166136261

typedef struct {
  int8_t weights[HISTORY_LENGTH + 1]; // using int8_t to save memory
  uint32_t tag;                       // added tag for better prediction
} perceptron_t;

typedef struct {
  uint64_t total_predictions;
  uint64_t correct_predictions;
  uint64_t mispredictions;
  uint64_t btb_misses; // branch target buffer misses
} statistics_t;

// global variables
static perceptron_t *perceptron_table;
static int *global_history;
static int *path_history; // added path history
static statistics_t statistics;

// improved hash function using fnv-1a
static inline uint32_t calculate_hash(uint32_t address) {
  uint32_t hash = FNV_OFFSET_BASIS;
  hash ^= (address >> 2); // remove lower bits to avoid aliasing
  hash *= FNV_PRIME;
  hash ^= (hash >> 17); // additional mixing

  return hash & (NUM_PERCEPTRONS - 1); // ensure num_perceptrons is power of 2
}

// initialize predictor state
void initialize_predictor(void) {
  // allocate memory dynamically
  perceptron_table = calloc(NUM_PERCEPTRONS, sizeof(perceptron_t));
  global_history = calloc(HISTORY_LENGTH, sizeof(int));
  path_history = calloc(HISTORY_LENGTH, sizeof(int));

  if (!perceptron_table || !global_history || !path_history) {
    fprintf(stderr, "memory allocation failed\n");
    exit(EXIT_FAILURE);
  }

  memset(&statistics, 0, sizeof(statistics));
}

// cleanup allocated memory
void cleanup_predictor(void) {
  free(perceptron_table);
  free(global_history);
  free(path_history);
}

// make prediction using both global and path history
int make_prediction(uint32_t address) {
  uint32_t index = calculate_hash(address);
  perceptron_t *perceptron = &perceptron_table[index];

  // check if this is a new branch
  if (perceptron->tag != (address >> 2)) {
    statistics.btb_misses++;
    perceptron->tag = (address >> 2);
    return 0; // default to not taken for new branches
  }

  int y = perceptron->weights[0]; // bias weight

  // combine global and path history
  for (int j = 1; j <= HISTORY_LENGTH; j++) {
    y += perceptron->weights[j] *
         (global_history[j - 1] + (path_history[j - 1] & 1));
  }

  return y;
}

// train the perceptron with saturating arithmetic
void train_perceptron(uint32_t address, int actual_outcome, int y) {
  uint32_t index = calculate_hash(address);
  perceptron_t *perceptron = &perceptron_table[index];

  // only train if prediction was wrong or confidence is below threshold
  if ((y >= 0 ? 1 : -1) != actual_outcome || abs(y) <= THETA) {
    // update weights with saturation
    for (int j = 0; j <= HISTORY_LENGTH; j++) {
      int new_weight =
          perceptron->weights[j] +
          actual_outcome *
              (j == 0 ? 1
                      : (global_history[j - 1] + (path_history[j - 1] & 1)));

      // saturate weights
      if (new_weight > MAX_WEIGHT)
        new_weight = MAX_WEIGHT;
      if (new_weight < MIN_WEIGHT)
        new_weight = MIN_WEIGHT;

      perceptron->weights[j] = new_weight;
    }
  }
}

// update history registers
void update_history(uint32_t address, int actual_outcome) {
  // shift histories
  memmove(&global_history[1], global_history,
          (HISTORY_LENGTH - 1) * sizeof(int));
  memmove(&path_history[1], path_history, (HISTORY_LENGTH - 1) * sizeof(int));

  // update with new values
  global_history[0] = actual_outcome;
  path_history[0] =
      address & PATH_HISTORY_MASK; // use lower bits for path history
}

// process branch trace
void process_trace_file(FILE *file) {
  uint32_t branch_address;
  int actual_outcome;

  while (fscanf(file, "%x %d", &branch_address, &actual_outcome) == 2) {
    actual_outcome = actual_outcome == 1 ? 1 : -1;

    int y = make_prediction(branch_address);
    int prediction = y >= 0 ? 1 : -1;

    statistics.total_predictions++;
    if (prediction == actual_outcome) {
      statistics.correct_predictions++;
    } else {
      statistics.mispredictions++;
      train_perceptron(branch_address, actual_outcome, y);
    }

    update_history(branch_address, actual_outcome);
  }
}

// print detailed statistics
void print_statistics(void) {
  printf("\n\t────────────────────────────────────────────────\n");
  printf("\t           Branch Predictor Statistics         \n");
  printf("\t────────────────────────────┬───────────────────\n");
  printf("\t Total Branches             │ %13lu \n",
         statistics.total_predictions);
  printf("\t Correct Predictions        │ %13lu \n",
         statistics.correct_predictions);
  printf("\t Mispredictions             │ %13lu \n", statistics.mispredictions);
  printf("\t Branch Target Buffer misses│ %13lu \n", statistics.btb_misses);
  printf("\t────────────────────────────┼───────────────────\n");
  printf("\t Prediction Accuracy        │ %16.2f%%    \n",
         100.0 * statistics.correct_predictions / statistics.total_predictions);
  printf("\t Mispredictions per 1K      │ %16.2f     \n",
         1000.0 * statistics.mispredictions / statistics.total_predictions);
  printf("\t────────────────────────────┴───────────────────\n\n");
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <trace-file>\n", argv[0]);

    return EXIT_FAILURE;
  }

  FILE *file = fopen(argv[1], "r");
  if (!file) {
    perror("failed to open trace file");

    return EXIT_FAILURE;
  }

  initialize_predictor();
  process_trace_file(file);
  print_statistics();
  cleanup_predictor();

  fclose(file);

  return EXIT_SUCCESS;
}
