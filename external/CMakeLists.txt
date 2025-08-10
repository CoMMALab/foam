add_subdirectory(spheretree)
add_subdirectory(manifold)
add_executable(simplify simplify/src.cmd/Main.cpp)


# add_subdirectory(manifold_old)
include_directories(manifold_old/src)
include_directories(manifold_old/3rd)

add_executable(
    manifold_old
    manifold_old/src/BVH.cpp
    manifold_old/src/BVH.h
    manifold_old/src/Intersection.cpp
    manifold_old/src/Intersection.h
    manifold_old/src/main.cpp
    manifold_old/src/Model_OBJ.cpp
    manifold_old/src/Model_OBJ.h
    manifold_old/src/Octree.h
)

find_package(Threads REQUIRED)
add_executable(
    simplify_old
    manifold_old/src/simplify.cpp
)
target_link_libraries(simplify_old PRIVATE Threads::Threads)
