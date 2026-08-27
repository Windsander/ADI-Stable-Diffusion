# Post-build helper (Linux): normalize the adi binary's ORT DT_NEEDED entry to
# the unversioned libonnxruntime.so that we ship in lib/.
#
#   cmake -DORT_LIB=onnxruntime -DORT_VERSION=1.28.0 -DEXE=<path-to-adi> \
#         -P patch_ort_soname.cmake
#
# ORT >= 1.19 soname is libonnxruntime.so.1; <= 1.18 was libonnxruntime.so.<ver>.
# Both are rewritten; a non-matching source name or a missing patchelf binary
# is non-fatal (the build must not fail on cosmetic normalization).
foreach (old_name IN ITEMS "lib${ORT_LIB}.so.1" "lib${ORT_LIB}.so.${ORT_VERSION}")
    execute_process(
            COMMAND patchelf --replace-needed ${old_name} lib${ORT_LIB}.so ${EXE}
            RESULT_VARIABLE patch_rc
            OUTPUT_QUIET ERROR_QUIET)
    message(STATUS "patchelf ${old_name} -> lib${ORT_LIB}.so on ${EXE}: rc=${patch_rc}")
endforeach ()
