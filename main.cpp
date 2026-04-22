#include <cstdio>
#include <cstring>

bool run_transpose_benchmark();
bool run_vector_add_benchmark();

struct OperatorEntry {
  const char *name;
  bool (*run_benchmark)();
};

static const OperatorEntry kOperators[] = {
    {"transpose", run_transpose_benchmark},
    {"vector_add", run_vector_add_benchmark},
};

static const OperatorEntry *find_operator(const char *name) {
  for (const auto &op : kOperators) {
    if (std::strcmp(op.name, name) == 0) {
      return &op;
    }
  }
  return nullptr;
}

static void print_usage(const char *program_name) {
  std::printf("Usage: %s <operator_name>\n", program_name);
  std::printf("Available operators:\n");
  for (const auto &op : kOperators) {
    std::printf("  %s\n", op.name);
  }
  std::printf("\nEach operator type has one benchmark entry here. To add a new "
              "implementation\n");
  std::printf("of an existing type, register it inside the matching benchmark "
              "file only.\n");
  std::printf("To add a new operator type, add a new benchmark file and "
              "register it here.\n");
}

int main(int argc, char **argv) {
  if (argc < 2) {
    print_usage(argv[0]);
    return 1;
  }

  const char *requested_operator = argv[1];
  if (std::strcmp(requested_operator, "all") == 0) {
    bool all_passed = true;
    for (const auto &op : kOperators) {
      std::printf("== %s ==\n", op.name);
      all_passed = op.run_benchmark() && all_passed;
    }
    return all_passed ? 0 : 2;
  }

  const OperatorEntry *op = find_operator(requested_operator);
  if (op == nullptr) {
    std::fprintf(stderr, "Unknown operator: %s\n", requested_operator);
    print_usage(argv[0]);
    return 1;
  }

  return op->run_benchmark() ? 0 : 2;
}
