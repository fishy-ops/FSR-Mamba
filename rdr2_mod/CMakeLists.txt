cmake_minimum_required(VERSION 3.20)
project(fsrmamba_rdr2 CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
if(NOT WIN32)
  message(FATAL_ERROR "Target must be Windows: use cmake/mingw-w64-x86_64.cmake or an MSVC x64 shell")
endif()

# Inputs fetched outside the repository (see README):
#   DML_ROOT  unpacked Microsoft.AI.DirectML NuGet package (include/DirectML.h, bin/x64-win/DirectML.dll)
#   DXC       dxc.exe (Windows SDK / DXC release) or the dxc_host helper built from tools/dxc_host.cpp
set(DML_ROOT "" CACHE PATH "Unpacked Microsoft.AI.DirectML NuGet package")
set(DXC "" CACHE FILEPATH "dxc executable or dxc_host helper")
set(PYTHON "python" CACHE STRING "Python 3 interpreter used to embed shaders")
if(NOT DML_ROOT OR NOT EXISTS "${DML_ROOT}/include/DirectML.h")
  message(FATAL_ERROR "Set -DDML_ROOT to the unpacked Microsoft.AI.DirectML package")
endif()
if(NOT DXC)
  find_program(DXC NAMES dxc dxc.exe)
endif()
if(NOT DXC)
  message(FATAL_ERROR "Set -DDXC to dxc.exe (or dxc_host)")
endif()

# ---- shaders: HLSL -> DXIL (cs_6_0) -> embedded C arrays -------------------------------------------
set(SHADER_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_DIR})
set(SHADER_HEADERS)
foreach(s pack resolve pack_fast resolve_fast kpn_pack kpn_resolve kpn_resolve3 exposure rcas)
  string(REGEX REPLACE "_fast$" "" source "${s}")
  set(defines)
  if(s STREQUAL "kpn_resolve3")
    set(source kpn_resolve)
    set(defines --define FSRM_KPN_TAPS=3)
  endif()
  if(s MATCHES "_fast$")
    set(defines --define FSRM_FAST=1)
  endif()
  add_custom_command(OUTPUT ${SHADER_DIR}/${s}_dxil.h
    COMMAND ${PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_shader.py
      ${SHADER_DIR}/${s}.dxil ${SHADER_DIR}/${s}_dxil.h ${s}_dxil
      --compiler ${DXC} --source ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${source}.hlsl
      --include ${CMAKE_CURRENT_SOURCE_DIR}/shaders ${defines}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${source}.hlsl ${CMAKE_CURRENT_SOURCE_DIR}/shaders/common.hlsli ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_shader.py
    COMMENT "dxc ${s}" VERBATIM)
  list(APPEND SHADER_HEADERS ${SHADER_DIR}/${s}_dxil.h)
endforeach()
add_custom_target(shaders DEPENDS ${SHADER_HEADERS})

# ---- common library ----------------------------------------------------------------------------------
add_library(fsrmamba_core STATIC src/gpu.cpp src/cpu_math.cpp src/weights.cpp)
add_dependencies(fsrmamba_core shaders)
target_include_directories(fsrmamba_core PUBLIC src third_party/fsr2 ${SHADER_DIR} ${DML_ROOT}/include)
target_link_libraries(fsrmamba_core PUBLIC d3d12 dxgi version)
if(MSVC)
  target_compile_options(fsrmamba_core PUBLIC /W3 /utf-8)
  target_compile_definitions(fsrmamba_core PUBLIC NOMINMAX _WIN32_WINNT=0x0A00 WINVER=0x0A00)
else()
  target_compile_options(fsrmamba_core PUBLIC -Wall -Wno-unknown-pragmas)
  target_compile_definitions(fsrmamba_core PUBLIC NOMINMAX _WIN32_WINNT=0x0A00 WINVER=0x0A00)
  target_link_options(fsrmamba_core PUBLIC -static)
endif()

# ---- the drop-in DLL ------------------------------------------------------------------------------------
add_library(ffx_fsr2_api_x64 SHARED src/proxy.cpp src/exports.def)
set_target_properties(ffx_fsr2_api_x64 PROPERTIES PREFIX "" OUTPUT_NAME ffx_fsr2_api_x64)
target_link_libraries(ffx_fsr2_api_x64 PRIVATE fsrmamba_core bcrypt)

# ---- headless test program --------------------------------------------------------------------------------
add_executable(offline_test tests/offline_test.cpp)
target_link_libraries(offline_test PRIVATE fsrmamba_core)
add_custom_command(TARGET offline_test POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${DML_ROOT}/bin/x64-win/DirectML.dll $<TARGET_FILE_DIR:offline_test>)

# ---- CPU maths / weight parser test (runs on Windows too) and ABI layout check ---------------------------
add_executable(cpu_test tests/cpu_test.cpp src/cpu_math.cpp src/weights.cpp)
target_include_directories(cpu_test PRIVATE src)
add_executable(abi_layout tests/abi_layout.cpp)
target_include_directories(abi_layout PRIVATE third_party/fsr2)
