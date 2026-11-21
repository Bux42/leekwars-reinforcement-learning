cmake_minimum_required(VERSION 3.20)
project(leekrl VERSION 0.1.0 LANGUAGES C)

add_subdirectory(leekwars-generator-c)

add_library(leekrl STATIC
    src/collect.c
    src/env.c
    src/leekwars.c
    src/ppo.c
)
target_include_directories(leekrl PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_features(leekrl PUBLIC c_std_99)
target_link_libraries(leekrl PUBLIC lwengine)
if(NOT MSVC)
    target_link_libraries(leekrl PUBLIC m)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(test_env tests/test_env.c)
    target_link_libraries(test_env PRIVATE leekrl)
    add_test(NAME environment COMMAND test_env)

    add_executable(test_ppo tests/test_ppo.c)
    target_link_libraries(test_ppo PRIVATE leekrl)
    add_test(NAME masked_ppo COMMAND test_ppo)

    add_executable(test_collect tests/test_collect.c)
    target_link_libraries(test_collect PRIVATE leekrl)
    add_test(NAME rollout_collection COMMAND test_collect)

    add_executable(test_leekwars tests/test_leekwars.c)
    target_link_libraries(test_leekwars PRIVATE leekrl)
    add_test(NAME leekwars_adapter COMMAND test_leekwars)

    add_executable(test_leekwars_training tests/test_leekwars_training.c)
    target_link_libraries(test_leekwars_training PRIVATE leekrl)
    add_test(NAME leekwars_training COMMAND test_leekwars_training)
endif()
