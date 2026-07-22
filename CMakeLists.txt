cmake_minimum_required(VERSION 3.16)
project(tinyllm VERSION 1.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -flto")
set(CMAKE_C_FLAGS_DEBUG "-g -O0 -fsanitize=address")

# Platform detection
if(APPLE)
  add_compile_definitions(TL_HAS_ACCELERATE=1)
  find_library(ACCELERATE_LIB Accelerate REQUIRED)
endif()

find_package(OpenMP)
if(OpenMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif()

# Check SIMD
include(CheckCCompilerFlag)
check_c_compiler_flag("-mavx2" HAS_AVX2)
check_c_compiler_flag("-mfma" HAS_FMA)
if(HAS_AVX2 AND HAS_FMA)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2 -mfma")
endif()

check_c_compiler_flag("-march=armv8-a+simd" HAS_NEON)
if(HAS_NEON)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8-a+simd")
endif()

# Sources
file(GLOB TINYLLM_SOURCES "src/*.c")
file(GLOB TINYLLM_HEADERS "include/*.h")

add_executable(tinyllm ${TINYLLM_SOURCES} ${TINYLLM_HEADERS})

target_include_directories(tinyllm PRIVATE include)

if(APPLE)
  target_link_libraries(tinyllm PRIVATE ${ACCELERATE_LIB})
else()
  target_link_libraries(tinyllm PRIVATE m pthread)
endif()

if(OpenMP_FOUND)
  target_link_libraries(tinyllm PRIVATE OpenMP::OpenMP_C)
endif()

# Install
install(TARGETS tinyllm DESTINATION bin)
