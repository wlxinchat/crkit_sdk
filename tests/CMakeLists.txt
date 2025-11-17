cmake_minimum_required(VERSION 3.16)

# 测试可执行文件
add_executable(test_tensor test_tensor.cpp)
target_link_libraries(test_tensor crkit_core)

add_executable(test_nms test_nms.cpp)
target_link_libraries(test_nms crkit_core)

add_executable(test_comprehensive test_comprehensive.cpp)
target_link_libraries(test_comprehensive crkit_core)

# 添加测试
add_test(NAME TensorTest COMMAND test_tensor)
add_test(NAME NMSTest COMMAND test_nms)
add_test(NAME ComprehensiveTest COMMAND test_comprehensive)

# 设置测试属性
set_tests_properties(TensorTest PROPERTIES
    TIMEOUT 10
    LABELS "unit"
)

set_tests_properties(NMSTest PROPERTIES
    TIMEOUT 10
    LABELS "unit"
)

set_tests_properties(ComprehensiveTest PROPERTIES
    TIMEOUT 30
    LABELS "comprehensive"
)
