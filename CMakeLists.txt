CMAKE_MINIMUM_REQUIRED( VERSION 3.2 FATAL_ERROR )

#===============================================================================
# project wide settings
PROJECT( pf-localization )

SET( PFL_ROOT  "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH "root directory" )
SET( CMAKE_INSTALL_PREFIX "${PFL_ROOT}" )
set( CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_STANDARD 11 )

#===============================================================================
# packages required to compile the project
FIND_PACKAGE( CUDA REQUIRED )

#===============================================================================
# compiler specific configuration
IF( MSVC )
  ADD_DEFINITIONS(
    -DNOMINMAX
    -DVC_EXTRALEAN
    -DWINVER=0x0601
    -D_WIN32_WINNT=0x0601
    -DWIN32_LEAN_AND_MEAN
    -D_CRT_SECURE_NO_WARNINGS )
ENDIF()
ADD_DEFINITIONS( -DBUILDING_DLL )

LIST( APPEND CUDA_NVCC_FLAGS "-Wno-deprecated-gpu-targets" )
INCLUDE_DIRECTORIES( "helpers/glm" )

#===============================================================================
# Build targets
CUDA_ADD_LIBRARY( p2c "p2c.cu" "p2c.h" "pf.h" "pf.cu" "pf.cuh" MODULE )

#===============================================================================
# Packaging and Installing
INSTALL( TARGETS p2c DESTINATION bin )
INSTALL( FILES p2c.h DESTINATION bin )
