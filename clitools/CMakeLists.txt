set(TARGET "adi-cli")

add_executable(${TARGET} main.cc)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE adi ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)

auto_copy_reference_dynamic(${TARGET} onnxruntime ${ONNXRUNTIME_LIB_PATH} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

add_custom_command(
        TARGET ${TARGET} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E rename
        "$<TARGET_FILE_DIR:${TARGET}>/${TARGET}"
        "$<TARGET_FILE_DIR:${TARGET}>/adi"
)
