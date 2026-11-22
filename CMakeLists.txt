cmake_minimum_required(VERSION 3.20)
project(leekrl VERSION 0.1.0 LANGUAGES C)

add_subdirectory(leekwars-generator-c)

add_library(leekrl STATIC
    src/actions.c
    src/battle.c
    src/collect.c
    src/env.c
    src/grid.c
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

    add_executable(test_actions tests/test_actions.c)
    target_link_libraries(test_actions PRIVATE leekrl)
    add_test(NAME leekwars_actions COMMAND test_actions)

    add_executable(test_grid tests/test_grid.c)
    target_link_libraries(test_grid PRIVATE leekrl)
    add_test(NAME leekwars_grid COMMAND test_grid)

    add_executable(test_battle tests/test_battle.c)
    target_link_libraries(test_battle PRIVATE leekrl)
    add_test(NAME leekwars_battle COMMAND test_battle)

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/weapons.json" AND
       EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/chips.json" AND
       EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/summons.json")
        add_executable(test_item_battle tests/test_item_battle.c)
        target_link_libraries(test_item_battle PRIVATE leekrl)
        add_test(NAME leekwars_item_battle COMMAND test_item_battle)
    endif()
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/weapons.json" AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/chips.json" AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/summons.json")
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    set(LEEKRL_SNAPSHOT_C "${CMAKE_CURRENT_BINARY_DIR}/item_snapshot.c")
    add_custom_command(
        OUTPUT "${LEEKRL_SNAPSHOT_C}"
        COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_item_snapshot.py"
            --weapons "${CMAKE_CURRENT_SOURCE_DIR}/weapons.json"
            --chips "${CMAKE_CURRENT_SOURCE_DIR}/chips.json"
            --summons "${CMAKE_CURRENT_SOURCE_DIR}/summons.json"
            --output "${LEEKRL_SNAPSHOT_C}"
        DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_item_snapshot.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/weapons.json"
            "${CMAKE_CURRENT_SOURCE_DIR}/chips.json"
            "${CMAKE_CURRENT_SOURCE_DIR}/summons.json"
        VERBATIM)
    target_sources(leekrl PRIVATE "${LEEKRL_SNAPSHOT_C}")
    target_compile_definitions(leekrl PUBLIC LEEKRL_HAS_ITEM_SNAPSHOT=1)
    if(BUILD_TESTING)
        add_test(NAME leekwars_snapshot_generator
            COMMAND "${Python3_EXECUTABLE}" -B -m unittest discover -s tests -p test_item_snapshot_generator.py)
        set_tests_properties(leekwars_snapshot_generator PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    endif()
endif()
