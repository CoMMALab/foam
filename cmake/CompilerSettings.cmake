set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g ${CMAKE_CXX_FLAGS_DEBUG}")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -flto ${CMAKE_CXX_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS "-march=native ${CMAKE_CXX_FLAGS}")

set(CMAKE_C_FLAGS_DEBUG "-g ${CMAKE_C_FLAGS_DEBUG}")
set(CMAKE_C_FLAGS_RELEASE "-O3 -flto ${CMAKE_C_FLAGS_RELEASE}")
set(CMAKE_C_FLAGS "-march=native ${CMAKE_C_FLAGS}")
