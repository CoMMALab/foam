add_subdirectory(spheretree)

find_package(Threads REQUIRED)
add_executable(
  manifold
  manifold/src/GridIndex.h
  manifold/src/Intersection.h
  manifold/src/Intersection.cc
  manifold/src/IO.h
  manifold/src/IO.cc
  manifold/src/main.cc
  manifold/src/Manifold.h
  manifold/src/Manifold.cc
  manifold/src/MeshProjector.cc
  manifold/src/MeshProjector.h
  manifold/src/Octree.h
  manifold/src/Octree.cc
  manifold/src/Parser.h
  manifold/src/Parser.cc
  manifold/src/types.h
)

target_include_directories(manifold PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/manifold/src)
target_include_directories(manifold PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/manifold/3rd_party/eigen/)
target_include_directories(manifold PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/manifold/3rd_party/libigl/include)
target_link_libraries(manifold PRIVATE Threads::Threads)

add_executable(simplify simplify/src.cmd/Main.cpp)
target_link_libraries(simplify PRIVATE Threads::Threads)

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

# Wonky include paths due to issues with old versions of Eigen vendored with manifold_old - the ones bundled with manifold work
target_include_directories(manifold_old PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/manifold/3rd_party/eigen/)
target_include_directories(manifold_old PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/manifold/3rd_party/libigl/include)
target_include_directories(manifold_old PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/manifold_old/3rd)
target_link_libraries(manifold_old PRIVATE Threads::Threads)

add_executable(
  simplify_old
  manifold_old/src/simplify.cpp
)

target_include_directories(simplify_old PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/manifold/3rd_party/eigen/)
target_include_directories(simplify_old PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/manifold/3rd_party/libigl/include)
target_include_directories(simplify_old PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/manifold_old/3rd)
target_link_libraries(simplify_old PRIVATE Threads::Threads)
