# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/zhaomin/project/DeltaBoost/src/test

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zhaomin/project/DeltaBoost/src/test/cmake-build-debug-xhead

# Include any dependencies generated for this target.
include CMakeFiles/Project-test.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Project-test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Project-test.dir/flags.make

CMakeFiles/Project-test.dir/test_dataset.o: CMakeFiles/Project-test.dir/flags.make
CMakeFiles/Project-test.dir/test_dataset.o: ../test_dataset.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhaomin/project/DeltaBoost/src/test/cmake-build-debug-xhead/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Project-test.dir/test_dataset.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Project-test.dir/test_dataset.o -c /home/zhaomin/project/DeltaBoost/src/test/test_dataset.cpp

CMakeFiles/Project-test.dir/test_dataset.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Project-test.dir/test_dataset.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhaomin/project/DeltaBoost/src/test/test_dataset.cpp > CMakeFiles/Project-test.dir/test_dataset.i

CMakeFiles/Project-test.dir/test_dataset.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Project-test.dir/test_dataset.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhaomin/project/DeltaBoost/src/test/test_dataset.cpp -o CMakeFiles/Project-test.dir/test_dataset.s

CMakeFiles/Project-test.dir/test_find_feature_range.o: CMakeFiles/Project-test.dir/flags.make
CMakeFiles/Project-test.dir/test_find_feature_range.o: ../test_find_feature_range.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhaomin/project/DeltaBoost/src/test/cmake-build-debug-xhead/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/Project-test.dir/test_find_feature_range.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Project-test.dir/test_find_feature_range.o -c /home/zhaomin/project/DeltaBoost/src/test/test_find_feature_range.cpp

CMakeFiles/Project-test.dir/test_find_feature_range.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Project-test.dir/test_find_feature_range.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhaomin/project/DeltaBoost/src/test/test_find_feature_range.cpp > CMakeFiles/Project-test.dir/test_find_feature_range.i

CMakeFiles/Project-test.dir/test_find_feature_range.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Project-test.dir/test_find_feature_range.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhaomin/project/DeltaBoost/src/test/test_find_feature_range.cpp -o CMakeFiles/Project-test.dir/test_find_feature_range.s

CMakeFiles/Project-test.dir/test_gbdt.o: CMakeFiles/Project-test.dir/flags.make
CMakeFiles/Project-test.dir/test_gbdt.o: ../test_gbdt.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhaomin/project/DeltaBoost/src/test/cmake-build-debug-xhead/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/Project-test.dir/test_gbdt.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Project-test.dir/test_gbdt.o -c /home/zhaomin/project/DeltaBoost/src/test/test_gbdt.cpp

CMakeFiles/Project-test.dir/test_gbdt.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Project-test.dir/test_gbdt.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhaomin/project/DeltaBoost/src/test/test_gbdt.cpp > CMakeFiles/Project-test.dir/test_gbdt.i

CMakeFiles/Project-test.dir/test_gbdt.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Project-test.dir/test_gbdt.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhaomin/project/DeltaBoost/src/test/test_gbdt.cpp -o CMakeFiles/Project-test.dir/test_gbdt.s

CMakeFiles/Project-test.dir/test_gradient.o: CMakeFiles/Project-test.dir/flags.make
CMakeFiles/Project-test.dir/test_gradient.o: ../test_gradient.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhaomin/project/DeltaBoost/src/test/cmake-build-debug-xhead/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/Project-test.dir/test_gradient.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Project-test.dir/test_gradient.o -c /home/zhaomin/project/DeltaBoost/src/test/test_gradient.cpp

CMakeFiles/Project-test.dir/test_gradient.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Project-test.dir/test_gradient.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhaomin/project/DeltaBoost/src/test/test_gradient.cpp > CMakeFiles/Project-test.dir/test_gradient.i

CMakeFiles/Project-test.dir/test_gradient.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Project-test.dir/test_gradient.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhaomin/project/DeltaBoost/src/test/test_gradient.cpp -o CMakeFiles/Project-test.dir/test_gradient.s

CMakeFiles/Project-test.dir/test_main.o: CMakeFiles/Project-test.dir/flags.make
CMakeFiles/Project-test.dir/test_main.o: ../test_main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhaomin/project/DeltaBoost/src/test/cmake-build-debug-xhead/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/Project-test.dir/test_main.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Project-test.dir/test_main.o -c /home/zhaomin/project/DeltaBoost/src/test/test_main.cpp

CMakeFiles/Project-test.dir/test_main.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Project-test.dir/test_main.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhaomin/project/DeltaBoost/src/test/test_main.cpp > CMakeFiles/Project-test.dir/test_main.i

CMakeFiles/Project-test.dir/test_main.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Project-test.dir/test_main.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhaomin/project/DeltaBoost/src/test/test_main.cpp -o CMakeFiles/Project-test.dir/test_main.s

CMakeFiles/Project-test.dir/test_parser.o: CMakeFiles/Project-test.dir/flags.make
CMakeFiles/Project-test.dir/test_parser.o: ../test_parser.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhaomin/project/DeltaBoost/src/test/cmake-build-debug-xhead/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/Project-test.dir/test_parser.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Project-test.dir/test_parser.o -c /home/zhaomin/project/DeltaBoost/src/test/test_parser.cpp

CMakeFiles/Project-test.dir/test_parser.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Project-test.dir/test_parser.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhaomin/project/DeltaBoost/src/test/test_parser.cpp > CMakeFiles/Project-test.dir/test_parser.i

CMakeFiles/Project-test.dir/test_parser.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Project-test.dir/test_parser.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhaomin/project/DeltaBoost/src/test/test_parser.cpp -o CMakeFiles/Project-test.dir/test_parser.s

CMakeFiles/Project-test.dir/test_partition.o: CMakeFiles/Project-test.dir/flags.make
CMakeFiles/Project-test.dir/test_partition.o: ../test_partition.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhaomin/project/DeltaBoost/src/test/cmake-build-debug-xhead/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/Project-test.dir/test_partition.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Project-test.dir/test_partition.o -c /home/zhaomin/project/DeltaBoost/src/test/test_partition.cpp

CMakeFiles/Project-test.dir/test_partition.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Project-test.dir/test_partition.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhaomin/project/DeltaBoost/src/test/test_partition.cpp > CMakeFiles/Project-test.dir/test_partition.i

CMakeFiles/Project-test.dir/test_partition.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Project-test.dir/test_partition.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhaomin/project/DeltaBoost/src/test/test_partition.cpp -o CMakeFiles/Project-test.dir/test_partition.s

CMakeFiles/Project-test.dir/test_tree.o: CMakeFiles/Project-test.dir/flags.make
CMakeFiles/Project-test.dir/test_tree.o: ../test_tree.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhaomin/project/DeltaBoost/src/test/cmake-build-debug-xhead/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/Project-test.dir/test_tree.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Project-test.dir/test_tree.o -c /home/zhaomin/project/DeltaBoost/src/test/test_tree.cpp

CMakeFiles/Project-test.dir/test_tree.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Project-test.dir/test_tree.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhaomin/project/DeltaBoost/src/test/test_tree.cpp > CMakeFiles/Project-test.dir/test_tree.i

CMakeFiles/Project-test.dir/test_tree.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Project-test.dir/test_tree.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhaomin/project/DeltaBoost/src/test/test_tree.cpp -o CMakeFiles/Project-test.dir/test_tree.s

CMakeFiles/Project-test.dir/test_tree_builder.o: CMakeFiles/Project-test.dir/flags.make
CMakeFiles/Project-test.dir/test_tree_builder.o: ../test_tree_builder.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhaomin/project/DeltaBoost/src/test/cmake-build-debug-xhead/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/Project-test.dir/test_tree_builder.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Project-test.dir/test_tree_builder.o -c /home/zhaomin/project/DeltaBoost/src/test/test_tree_builder.cpp

CMakeFiles/Project-test.dir/test_tree_builder.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Project-test.dir/test_tree_builder.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhaomin/project/DeltaBoost/src/test/test_tree_builder.cpp > CMakeFiles/Project-test.dir/test_tree_builder.i

CMakeFiles/Project-test.dir/test_tree_builder.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Project-test.dir/test_tree_builder.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhaomin/project/DeltaBoost/src/test/test_tree_builder.cpp -o CMakeFiles/Project-test.dir/test_tree_builder.s

# Object files for target Project-test
Project__test_OBJECTS = \
"CMakeFiles/Project-test.dir/test_dataset.o" \
"CMakeFiles/Project-test.dir/test_find_feature_range.o" \
"CMakeFiles/Project-test.dir/test_gbdt.o" \
"CMakeFiles/Project-test.dir/test_gradient.o" \
"CMakeFiles/Project-test.dir/test_main.o" \
"CMakeFiles/Project-test.dir/test_parser.o" \
"CMakeFiles/Project-test.dir/test_partition.o" \
"CMakeFiles/Project-test.dir/test_tree.o" \
"CMakeFiles/Project-test.dir/test_tree_builder.o"

# External object files for target Project-test
Project__test_EXTERNAL_OBJECTS =

Project-test: CMakeFiles/Project-test.dir/test_dataset.o
Project-test: CMakeFiles/Project-test.dir/test_find_feature_range.o
Project-test: CMakeFiles/Project-test.dir/test_gbdt.o
Project-test: CMakeFiles/Project-test.dir/test_gradient.o
Project-test: CMakeFiles/Project-test.dir/test_main.o
Project-test: CMakeFiles/Project-test.dir/test_parser.o
Project-test: CMakeFiles/Project-test.dir/test_partition.o
Project-test: CMakeFiles/Project-test.dir/test_tree.o
Project-test: CMakeFiles/Project-test.dir/test_tree_builder.o
Project-test: CMakeFiles/Project-test.dir/build.make
Project-test: lib/libgtestd.a
Project-test: CMakeFiles/Project-test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zhaomin/project/DeltaBoost/src/test/cmake-build-debug-xhead/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Linking CXX executable Project-test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Project-test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Project-test.dir/build: Project-test

.PHONY : CMakeFiles/Project-test.dir/build

CMakeFiles/Project-test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Project-test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Project-test.dir/clean

CMakeFiles/Project-test.dir/depend:
	cd /home/zhaomin/project/DeltaBoost/src/test/cmake-build-debug-xhead && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zhaomin/project/DeltaBoost/src/test /home/zhaomin/project/DeltaBoost/src/test /home/zhaomin/project/DeltaBoost/src/test/cmake-build-debug-xhead /home/zhaomin/project/DeltaBoost/src/test/cmake-build-debug-xhead /home/zhaomin/project/DeltaBoost/src/test/cmake-build-debug-xhead/CMakeFiles/Project-test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Project-test.dir/depend

