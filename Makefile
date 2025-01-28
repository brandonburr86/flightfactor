# Makefile for FlightFactor Project

CC      = gcc
CFLAGS  = -Wall -O0
LIBS    = -lm

SRC_DIR = ./src
OBJ_DIR = ./obj
BIN_DIR = ./bin

# Names (without paths) of the final executables
APP1 = ff_generate_data
APP2 = ff_train
APP3 = flightfactor

# Full paths for the final binaries
EXECUTABLES = $(BIN_DIR)/$(APP1) \
              $(BIN_DIR)/$(APP2) \
              $(BIN_DIR)/$(APP3)

# Default target: build everything
all: directories $(EXECUTABLES)

# Create bin/ and obj/ if they don't exist
directories:
	mkdir -p $(BIN_DIR)
	mkdir -p $(OBJ_DIR)

# Pattern rule for compiling .c into .o, storing .o in obj/
# Example: obj/ff_generate_data.o from src/ff_generate_data.c
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Link each executable from its corresponding .o
$(BIN_DIR)/ff_generate_data: $(OBJ_DIR)/ff_generate_data.o
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

$(BIN_DIR)/ff_train: $(OBJ_DIR)/ff_train.o
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

$(BIN_DIR)/flightfactor: $(OBJ_DIR)/flightfactor.o
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

# Clean up build artifacts
clean:
	rm -f $(OBJ_DIR)/*.o
	rm -f $(EXECUTABLES)

.PHONY: all clean directories
