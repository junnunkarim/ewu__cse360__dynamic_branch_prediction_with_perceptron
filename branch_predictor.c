#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// configuration parameters
#define NUM_PERCEPTRONS 1024
#define HISTORY_LENGTH 64
#define THETA (int)(2.14 * HISTORY_LENGTH + 20.58)
#define MAX_WEIGHT 127
#define MIN_WEIGHT -128
#define PATH_HISTORY_MASK 0xF
#define FNV_PRIME 16777619
#define FNV_OFFSET_BASIS 2166136261

// debug logging setup
static FILE *debug_log_file = NULL;
static bool debug_enabled = false;
#define DEBUG_LOG(fmt, ...)                                                    \
  if (debug_enabled && debug_log_file) {                                       \
    fprintf(debug_log_file, "[%s] " fmt "\n", get_timestamp(), ##__VA_ARGS__); \
    fflush(debug_log_file);                                                    \
  }

// data structures
typedef struct {
  int8_t weights[HISTORY_LENGTH + 1]; // weight vector including bias
  uint32_t tag;                       // branch address tag
  uint32_t last_update_time;          // timestamp of last update
  uint32_t times_accessed;            // usage counter
} perceptron_t;

typedef struct {
  uint64_t total_predictions;
  uint64_t correct_predictions;
  uint64_t mispredictions;
  uint64_t btb_misses;
  uint64_t training_events;
  uint64_t strong_predictions;
  uint64_t weak_predictions;
  double avg_confidence;
} statistics_t;

// global state
static perceptron_t *perceptron_table;
static int *global_history;
static int *path_history;
static statistics_t statistics;
static uint32_t current_time = 0;

// utility functions for logging
static char *get_timestamp(void) {
  static char buffer[32];
  time_t timer;
  struct tm *tm_info;

  time(&timer);
  tm_info = localtime(&timer);
  strftime(buffer, 32, "%Y%m%d_%H%M%S", tm_info);
  return buffer;
}

// handle debug logging initialization
static bool initialize_logging(void) {
  if (!debug_enabled) {
    return true; // skip logging setup if debugging is disabled
  }

  // create filename with timestamp
  char filename[64];
  snprintf(filename, sizeof(filename), "branch_predictor_%s.log",
           get_timestamp());

  debug_log_file = fopen(filename, "w");
  if (!debug_log_file) {
    fprintf(stderr, "failed to open debug log file: %s\n", filename);
    return false;
  }

  DEBUG_LOG("debug logging initialized in file: %s", filename);
  return true;
}

// memory management functions
static bool allocate_predictor_memory(void) {
  // allocate all required memory structures
  perceptron_table = calloc(NUM_PERCEPTRONS, sizeof(perceptron_t));
  global_history = calloc(HISTORY_LENGTH, sizeof(int));
  path_history = calloc(HISTORY_LENGTH, sizeof(int));

  if (!perceptron_table || !global_history || !path_history) {
    DEBUG_LOG("memory allocation failed");
    return false;
  }
  return true;
}

// compute perceptron index using fnv hash
static uint32_t compute_perceptron_index(uint32_t address) {
  uint32_t hash = FNV_OFFSET_BASIS;
  hash ^= (address >> 2);
  hash *= FNV_PRIME;
  hash ^= (hash >> 17);
  return hash & (NUM_PERCEPTRONS - 1);
}

// calculate weighted sum for prediction
static int compute_perceptron_output(perceptron_t *perceptron) {
  int y = perceptron->weights[0]; // bias term
  for (int j = 1; j <= HISTORY_LENGTH; j++) {
    y += perceptron->weights[j] *
         (global_history[j - 1] + (path_history[j - 1] & 1));
  }
  return y;
}

// update confidence metrics
static void update_confidence_statistics(double confidence) {
  if (confidence >= 1.0) {
    statistics.strong_predictions++;
  } else {
    statistics.weak_predictions++;
  }

  statistics.avg_confidence =
      (statistics.avg_confidence * statistics.total_predictions + confidence) /
      (statistics.total_predictions + 1);
}

// update perceptron weights during training
static void update_perceptron_weights(perceptron_t *perceptron,
                                      int actual_outcome) {
  for (int j = 0; j <= HISTORY_LENGTH; j++) {
    int history_val =
        (j == 0) ? 1 : (global_history[j - 1] + (path_history[j - 1] & 1));

    // calculate new weight with saturation
    int new_weight = perceptron->weights[j] + actual_outcome * history_val;
    new_weight = (new_weight > MAX_WEIGHT)   ? MAX_WEIGHT
                 : (new_weight < MIN_WEIGHT) ? MIN_WEIGHT
                                             : new_weight;

    perceptron->weights[j] = new_weight;
  }
  perceptron->last_update_time = current_time;
}

// debug functions
static void dump_perceptron_weights(perceptron_t *p, uint32_t index) {
  DEBUG_LOG("perceptron[%u] weights:", index);
  for (int i = 1; i <= HISTORY_LENGTH; i++) {
    if (i % 8 == 1) {
      DEBUG_LOG("  weights[%3d-%3d]:", i, i + 7);
    }

    if (debug_enabled) {
      fprintf(debug_log_file, " %4d", p->weights[i]);
      if (i % 8 == 0 || i == HISTORY_LENGTH) {
        fprintf(debug_log_file, "\n");
      }
    }
  }
}

// print detailed perceptron state
static void dump_perceptron_state(uint32_t index) {
  perceptron_t *p = &perceptron_table[index];
  DEBUG_LOG("perceptron[%u] state:", index);
  DEBUG_LOG("  tag: 0x%x", p->tag);
  DEBUG_LOG("  times accessed: %u", p->times_accessed);
  DEBUG_LOG("  last update: %u cycles ago", current_time - p->last_update_time);
  DEBUG_LOG("  bias weight: %d", p->weights[0]);
  dump_perceptron_weights(p, index);
}

// core prediction logic
int make_prediction(uint32_t address, double *confidence) {
  uint32_t index = compute_perceptron_index(address);
  perceptron_t *perceptron = &perceptron_table[index];

  // handle new branch
  if (perceptron->tag != (address >> 2)) {
    statistics.btb_misses++;
    perceptron->tag = (address >> 2);
    perceptron->times_accessed = 0;
    *confidence = 0.0;
    DEBUG_LOG("btb miss for address 0x%x", address);
    return 0;
  }

  perceptron->times_accessed++;
  current_time++;

  int y = compute_perceptron_output(perceptron);
  *confidence = (double)abs(y) / THETA;

  update_confidence_statistics(*confidence);

  DEBUG_LOG("prediction for 0x%x: y=%d, confidence=%.2f", address, y,
            *confidence);
  return y;
}

// history management functions
void update_history(uint32_t address, int actual_outcome) {
  // update global history
  memmove(&global_history[1], global_history,
          (HISTORY_LENGTH - 1) * sizeof(int));
  global_history[0] = actual_outcome;

  // update path history
  memmove(&path_history[1], path_history, (HISTORY_LENGTH - 1) * sizeof(int));
  path_history[0] = address & PATH_HISTORY_MASK;

  DEBUG_LOG("updated history: outcome=%d, path=0x%x", actual_outcome,
            path_history[0]);
}

// training logic
void train_perceptron(uint32_t address, int actual_outcome, int y) {
  uint32_t index = compute_perceptron_index(address);
  perceptron_t *perceptron = &perceptron_table[index];

  // train only if prediction was wrong or confidence is low
  if ((y >= 0 ? 1 : -1) != actual_outcome || abs(y) <= THETA) {
    DEBUG_LOG("training perceptron[%u] for address 0x%x", index, address);
    statistics.training_events++;

    update_perceptron_weights(perceptron, actual_outcome);
    dump_perceptron_state(index);
  }
}

// initialization and cleanup
bool initialize_predictor(void) {
  DEBUG_LOG("initializing predictor");

  if (!initialize_logging() || !allocate_predictor_memory()) {
    return false;
  }

  memset(&statistics, 0, sizeof(statistics));
  DEBUG_LOG("predictor initialization complete");
  return true;
}

void cleanup_predictor(void) {
  DEBUG_LOG("cleaning up predictor resources");
  free(perceptron_table);
  free(global_history);
  free(path_history);

  if (debug_log_file) {
    fclose(debug_log_file);
  }
}

// trace processing
void process_trace_file(FILE *file) {
  uint32_t branch_address;
  int actual_outcome;
  double confidence;

  DEBUG_LOG("starting trace processing");

  while (fscanf(file, "%x %d", &branch_address, &actual_outcome) == 2) {
    actual_outcome = actual_outcome == 1 ? 1 : -1;

    int y = make_prediction(branch_address, &confidence);
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

  DEBUG_LOG("trace processing complete");
}

// statistics reporting
void print_statistics(void) {
  DEBUG_LOG("printing final statistics");

  printf("\n\t────────────────────────────────────────────────\n");
  printf("\t           Branch Predictor Statistics           \n");
  printf("\t────────────────────────────┬───────────────────\n");
  printf("\t Total Branches             │ %13lu \n",
         statistics.total_predictions);
  printf("\t Correct Predictions        │ %13lu \n",
         statistics.correct_predictions);
  printf("\t Mispredictions             │ %13lu \n", statistics.mispredictions);
  printf("\t BTB Misses                 │ %13lu \n", statistics.btb_misses);
  printf("\t Training Events            │ %13lu \n",
         statistics.training_events);
  printf("\t Strong Predictions         │ %13lu \n",
         statistics.strong_predictions);
  printf("\t Weak Predictions           │ %13lu \n",
         statistics.weak_predictions);
  printf("\t────────────────────────────┼───────────────────\n");
  printf("\t Prediction Accuracy        │ %16.2f%%\n",
         100.0 * statistics.correct_predictions / statistics.total_predictions);
  printf("\t Mispredictions per 1K      │ %16.2f \n",
         1000.0 * statistics.mispredictions / statistics.total_predictions);
  printf("\t Average Confidence         │ %16.2f \n",
         statistics.avg_confidence);
  printf("\t────────────────────────────┴───────────────────\n\n");
}

int main(int argc, char *argv[]) {
  if (argc < 2 || argc > 3) {
    fprintf(stderr, "usage: %s <trace-file> [--debug]\n", argv[0]);
    return EXIT_FAILURE;
  }

  if (argc == 3 && strcmp(argv[2], "--debug") == 0) {
    debug_enabled = true;
  }

  FILE *file = fopen(argv[1], "r");
  if (!file) {
    perror("failed to open trace file");
    return EXIT_FAILURE;
  }

  if (!initialize_predictor()) {
    fclose(file);
    return EXIT_FAILURE;
  }

  process_trace_file(file);
  print_statistics();
  cleanup_predictor();

  fclose(file);
  return EXIT_SUCCESS;
}
