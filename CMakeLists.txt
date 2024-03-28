cmake_minimum_required(VERSION 3.10)

# 工具链
set(VCPKG_ROOT "F:/newapp/vcpkg/scripts/buildsystems/vcpkg.cmake")
set(CMAKE_TOOLCHAIN_FILE ${VCPKG_ROOT})

project(disp)


set(CMAKE_CXX_STANDARD 17)
## find package
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

# include
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ./
    ./Elas
)

## library
add_library(elas STATIC
    ./Elas/descriptor.cpp
    ./Elas/elas.cpp
    ./Elas/filter.cpp
    ./Elas/matrix.cpp
    ./Elas/triangle.cpp
)


target_link_libraries(elas
    ${OpenCV_LIBS}
)

add_executable(main 
    main.cc
    utility/executor.cc
    utility/utility.cc
    utility/executor.h
    utility/utility.h
)

target_link_libraries(main
    elas
    ${PCL_LIBRARIES}
)

