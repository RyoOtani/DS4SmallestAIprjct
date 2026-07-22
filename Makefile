# tinyllm Makefile
# ds4: simple, no autotools, no cmake required.
# Just type: make

CC       := cc
CFLAGS   := -std=c11 -O3 -march=native -flto
CFLAGS   += -Wall -Wextra -Wpedantic -Werror=implicit-function-declaration
CFLAGS   += -Wno-unused-parameter -Wno-missing-field-initializers -Wno-gnu-statement-expression
CFLAGS   += -Iinclude -D_POSIX_C_SOURCE=200809L -D_DARWIN_C_SOURCE

# Performance: auto-vectorization hints
CFLAGS   += -ffast-math -funroll-loops

# OpenMP (optional, clang on macOS doesn't ship libomp)
ifeq ($(shell $(CC) -fopenmp -E -x c /dev/null 2>/dev/null; echo $$?),0)
CFLAGS   += -fopenmp
LDFLAGS  += -fopenmp
$(info ✓ OpenMP enabled)
endif

# Platform-specific
UNAME_S  := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	CFLAGS += -DTL_HAS_ACCELERATE=1
	LDFLAGS += -framework Accelerate
	# Check for Apple Silicon native NEON
	ARCH := $(shell uname -m)
	ifeq ($(ARCH),arm64)
		CFLAGS += -mcpu=apple-m1  # enables all NEON features
	endif
endif
ifeq ($(UNAME_S),Linux)
	LDFLAGS += -lm -lpthread
endif

# Detect AVX2+FMA
AVX2 := $(shell $(CC) -mavx2 -dM -E - < /dev/null 2>/dev/null | grep -q AVX2 && echo 1 || echo 0)
FMA  := $(shell $(CC) -mfma -dM -E - < /dev/null 2>/dev/null | grep -q FMA && echo 1 || echo 0)
ifeq ($(AVX2)$(FMA),11)
CFLAGS += -mavx2 -mfma
$(info ✓ AVX2+FMA detected)
endif

# Detect NEON
NEON := $(shell $(CC) -march=armv8-a+simd -dM -E - < /dev/null 2>/dev/null | grep -q __ARM_NEON && echo 1 || echo 0)
ifeq ($(NEON),1)
CFLAGS += -march=armv8-a+simd
$(info ✓ NEON detected)
endif

SRCS := $(wildcard src/*.c)
OBJS := $(SRCS:.c=.o)
TARGET := tinyllm

.PHONY: all clean test install format check

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "✓ Built $(TARGET) ($(shell du -sh $(TARGET) | cut -f1))"

%.o: %.c include/*.h
	$(CC) $(CFLAGS) -c -o $@ $<

# Debug build (ASAN)
debug: CFLAGS = -std=c11 -g -O0 -fsanitize=address -fno-omit-frame-pointer
debug: CFLAGS += -Wall -Wextra -Wpedantic -Werror
debug: CFLAGS += -Iinclude -D_POSIX_C_SOURCE=200809L -D_DARWIN_C_SOURCE
debug: clean $(TARGET)

# Release binary (static linking on Linux)
release: CC := gcc
release: CFLAGS += -static -s
release: clean $(TARGET)

# Count lines of code
cloc:
	@echo "C source files:"
	@wc -l src/*.c
	@echo "Header files:"
	@wc -l include/*.h
	@echo "Python files:"
	@wc -l python/**/*.py
	@echo "---"
	@echo "Total C: $$(wc -l < /dev/null && cat src/*.c include/*.h | wc -l) lines"
	@echo "Total Python: $$(cat python/**/*.py 2>/dev/null | wc -l) lines"

# Test
test: $(TARGET)
	@echo "No test runner yet. Run: ./tinyllm info model.gguf"
	@cd tests && ls -la

# Install to /usr/local/bin
install: $(TARGET)
	install -m 755 $(TARGET) /usr/local/bin/tinyllm
	@echo "✓ Installed to /usr/local/bin/tinyllm"

# Clean
clean:
	rm -f src/*.o $(TARGET)
	@echo "✓ Clean"

# Format
format:
	clang-format -i src/*.c include/*.h 2>/dev/null || echo "⚠ clang-format not installed"

# Check for compilation errors only (no link)
check:
	$(CC) -fsyntax-only $(CFLAGS) src/*.c
