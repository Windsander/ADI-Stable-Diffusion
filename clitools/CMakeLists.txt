set(TARGET "onnxruntime-sd-clitools")

add_executable(${TARGET} main.cc)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE ort-sd ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PUBLIC cxx_std_11)