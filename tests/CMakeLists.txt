include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

# good practice for Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)
add_executable(SVM_test SVM_test.cpp)

target_link_libraries(
  SVM_test
  GTest::gtest_main
)
target_link_libraries(SVM_test SVM_lib)

include(GoogleTest)
foreach(THREADS 1 4 8)
    gtest_discover_tests(SVM_test
        TEST_PREFIX "OMP_${THREADS}T_"
        PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=${THREADS}"
    )
endforeach()