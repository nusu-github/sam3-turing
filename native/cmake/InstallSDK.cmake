include(CMakePackageConfigHelpers)
# ELF RUNPATH is not transitive. RPATH keeps bundled ICU and other nested
# dependencies beside this library instead of falling back to host copies.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_options(sam3_native PRIVATE "-Wl,--disable-new-dtags")
endif()
set_target_properties(sam3_native PROPERTIES EXPORT_NAME c INSTALL_RPATH "$ORIGIN" INSTALL_RPATH_USE_LINK_PATH FALSE)
install(TARGETS sam3_native EXPORT Sam3NativeTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/sam3" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/sam3_native_export.h" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/../sam3/assets/bpe_simple_vocab_16e6.txt.gz"
  DESTINATION ${CMAKE_INSTALL_DATADIR}/sam3-native)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE" DESTINATION ${CMAKE_INSTALL_DATADIR}/sam3-native RENAME LICENSE-SAM)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/examples/c-client" DESTINATION ${CMAKE_INSTALL_DATADIR}/sam3-native/examples)
set(SAM3_PACKAGE_DIRECTORY "${CMAKE_INSTALL_LIBDIR}/cmake/Sam3Native")
configure_package_config_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/Sam3NativeConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/Sam3NativeConfig.cmake" INSTALL_DESTINATION ${SAM3_PACKAGE_DIRECTORY})
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/Sam3NativeConfigVersion.cmake"
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(EXPORT Sam3NativeTargets NAMESPACE sam3:: DESTINATION ${SAM3_PACKAGE_DIRECTORY})
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/Sam3NativeConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/Sam3NativeConfigVersion.cmake"
  DESTINATION ${SAM3_PACKAGE_DIRECTORY})
# Runtime dependencies are explicit external inputs, never the Python wheel.
# This metadata records the producer build; it is not an absolute search path.
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/build-info.txt.in" "${CMAKE_CURRENT_BINARY_DIR}/sam3-build-info.txt" @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/sam3-build-info.txt" DESTINATION ${CMAKE_INSTALL_DATADIR}/sam3-native)

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/cmake/BundleRuntime.cmake" DESTINATION ${CMAKE_INSTALL_DATADIR}/sam3-native/cmake)
