cmake_minimum_required (VERSION 3.14)
project(nori LANGUAGES CUDA CXX)

add_subdirectory(ext ext_build)


#find_package(CUDA)
#enable_language(CUDA)

include(CheckLanguage)
check_language(CUDA)

#SET(CMAKE_BUILD_TYPE Debug)

include_directories(
  # Nori include files
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  # tinyformat string formatting library
  ${TFM_INCLUDE_DIR}
  # Eigen linear algebra library
  SYSTEM ${EIGEN_INCLUDE_DIR}
  # OpenEXR high dynamic range bitmap library
  SYSTEM ${OPENEXR_INCLUDE_DIRS}
  # Intel Thread Building Blocks
  SYSTEM ${TBB_INCLUDE_DIR}
  # Pseudorandom number generator
  ${PCG32_INCLUDE_DIR}
  # PugiXML parser
  ${PUGIXML_INCLUDE_DIR}
  # Helper functions for statistical hypothesis tests
  ${HYPOTHESIS_INCLUDE_DIR}
  # GLFW library for OpenGL context creation
  SYSTEM ${GLFW_INCLUDE_DIR}
  # GLEW library for accessing OpenGL functions
  SYSTEM ${GLEW_INCLUDE_DIR}
  # NanoVG drawing library
  SYSTEM ${NANOVG_INCLUDE_DIR}
  # NanoGUI user interface library
  SYSTEM ${NANOGUI_INCLUDE_DIR}
  SYSTEM ${NANOGUI_EXTRA_INCS}
  # Portable filesystem API
  SYSTEM ${FILESYSTEM_INCLUDE_DIR}
  # STB Image Write
  SYSTEM ${STB_IMAGE_WRITE_INCLUDE_DIR}
)

# The following lines build the main executable. If you add a source
# code file to Nori, be sure to include it in this list.
add_executable(nori

  # Header files
  include/nori/bbox.h
  include/nori/bitmap.h
  include/nori/block.h
  include/nori/bsdf.h
  include/nori/accel.h
  include/nori/camera.h
  include/nori/color.h
  include/nori/common.h
  include/nori/dpdf.h
  include/nori/frame.h
  include/nori/integrator.h
  include/nori/emitter.h
  include/nori/mesh.h
  include/nori/object.h
  include/nori/parser.h
  include/nori/proplist.h
  include/nori/ray.h
  include/nori/rfilter.h
  include/nori/sampler.h
  include/nori/scene.h
  include/nori/timer.h
  include/nori/transform.h
  include/nori/vector.h
  include/nori/warp.h

  # Source code files
  src/bitmap.cpp
  src/block.cpp
  src/accel.cpp
  src/chi2test.cpp
  src/common.cpp
  src/diffuse.cpp
  src/gui.cpp
  src/independent.cpp
  src/main.cpp
  src/mesh.cpp
  src/obj.cpp
  src/object.cpp
  src/parser.cpp
  src/perspective.cpp
  src/proplist.cpp
  src/rfilter.cpp
  src/scene.cpp
  src/ttest.cpp
  src/warp.cpp
  src/microfacet.cpp
  src/mirror.cpp
  src/dielectric.cpp
)

add_definitions(${NANOGUI_EXTRA_DEFS})

# The following lines build the warping test application
add_executable(warptest
  include/nori/warp.h
  src/warp.cpp
  src/warptest.cpp
  src/microfacet.cpp
  src/object.cpp
  src/proplist.cpp
  src/common.cpp
)

add_executable(pathgraph
  include/nori/shadingPoint.h
  include/nori/pathgraph.h
  include/nori/warp.h
  include/nori/bitmap.h
  src/bitmap.cpp
  src/warp.cpp
  src/common.cpp
  src/pathgraph.cpp
  src/visual.cpp
  src/pbsdf.cu
  )

add_executable(pg
  include/nori/shadingPoint.h
  include/nori/pathgraph.h
  include/nori/warp.h
  include/nori/bitmap.h
  src/bitmap.cpp
  src/warp.cpp
  src/common.cpp
  src/pathgraph.cpp
  src/cluster.cpp
  src/pbsdf.cu
  )

add_executable(test
  src/test.cu
  )

if (WIN32)
  target_link_libraries(nori tbb_static pugixml IlmImf nanogui ${NANOGUI_EXTRA_LIBS} zlibstatic)
else()
  target_link_libraries(nori tbb_static pugixml IlmImf nanogui ${NANOGUI_EXTRA_LIBS})
endif()

target_link_libraries(warptest tbb_static nanogui ${NANOGUI_EXTRA_LIBS})
target_link_libraries(pathgraph tbb_static IlmImf nanogui ${NANOGUI_EXTRA_LIBS})
target_link_libraries(pg tbb_static IlmImf nanogui ${NANOGUI_EXTRA_LIBS})
#target_link_libraries(test tbb_static nanogui IlmImf ${NANOGUI_EXTRA_LIBS})

# Force colored output for the ninja generator
if (CMAKE_GENERATOR STREQUAL "Ninja")
  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fcolor-diagnostics")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcolor-diagnostics")
  elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fdiagnostics-color=always")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always")
  endif()
endif()

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libstdc++ -lc++abi")


target_compile_features(warptest PRIVATE cxx_std_17)
target_compile_features(nori PRIVATE cxx_std_17)
target_compile_features(pathgraph PRIVATE cxx_std_17 cuda_std_11)
target_compile_features(pg PRIVATE cxx_std_17 cuda_std_11)
#target_compile_features(test PRIVATE cxx_std_17 cuda_std_11)
SET_TARGET_PROPERTIES(pathgraph PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
SET_TARGET_PROPERTIES(pathgraph PROPERTIES CUDA_ARCHITECTURES 75)

SET_TARGET_PROPERTIES(pg PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
SET_TARGET_PROPERTIES(pg PROPERTIES CUDA_ARCHITECTURES 75)


SET_TARGET_PROPERTIES(test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
SET_TARGET_PROPERTIES(test PROPERTIES CUDA_ARCHITECTURES 75)

# vim: set et ts=2 sw=2 ft=cmake nospell:
