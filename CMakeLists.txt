cmake_minimum_required(VERSION 3.20)
project(leekrl VERSION 0.1.0 LANGUAGES C)

# The library Python loads is optimized across files when linked: about 15% faster. Its
# objects also hold ordinary code, for the tests to link quickly.
option(LEEKRL_LTO "Optimize the shared library at link time" ON)
if(LEEKRL_LTO AND CMAKE_C_COMPILER_ID STREQUAL "GNU")
    set(LEEKRL_RELEASE_LTO "$<$<CONFIG:Release>:-flto;-ffat-lto-objects>")
    add_link_options(
        "$<$<AND:$<CONFIG:Release>,$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>>:-fno-lto>")
endif()

add_subdirectory(leekwars-generator-c)

add_library(leekrl STATIC
    src/actions.c
    src/battle.c
    src/collect.c
    src/env.c
    src/leekwars.c
    src/observation.c
    src/plant_ai.c
    src/ppo.c
    src/reward.c
    src/rules.c
    src/threads.c
)
target_include_directories(leekrl PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_features(leekrl PUBLIC c_std_99)
find_package(Threads REQUIRED)
target_link_libraries(leekrl PUBLIC lwengine Threads::Threads)
set_target_properties(leekrl PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(NOT MSVC)
    target_link_libraries(leekrl PUBLIC m)
endif()
if(LEEKRL_RELEASE_LTO)
    target_compile_options(leekrl PRIVATE "${LEEKRL_RELEASE_LTO}")
    target_compile_options(lwengine PRIVATE "${LEEKRL_RELEASE_LTO}")
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

    add_executable(test_battle tests/test_battle.c)
    target_link_libraries(test_battle PRIVATE leekrl)
    add_test(NAME leekwars_battle COMMAND test_battle)

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/weapons.json" AND
       EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/chips.json" AND
       EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/summons.json" AND
       EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/components.json")
        add_executable(test_item_battle tests/test_item_battle.c)
        target_link_libraries(test_item_battle PRIVATE leekrl)
        add_test(NAME leekwars_item_battle COMMAND test_item_battle)

        add_executable(test_stepping tests/test_stepping.c)
        target_link_libraries(test_stepping PRIVATE leekrl)
        add_test(NAME leekwars_stepping COMMAND test_stepping)

        add_executable(test_plant_ai tests/test_plant_ai.c)
        target_link_libraries(test_plant_ai PRIVATE leekrl)
        add_test(NAME leekwars_plant_ai COMMAND test_plant_ai)

        add_executable(test_build tests/test_build.c)
        target_link_libraries(test_build PRIVATE leekrl)
        add_test(NAME leekwars_build COMMAND test_build)

        add_executable(test_observation tests/test_observation.c)
        target_link_libraries(test_observation PRIVATE leekrl)
        add_test(NAME leekwars_observation COMMAND test_observation)

        add_executable(test_reward tests/test_reward.c)
        target_link_libraries(test_reward PRIVATE leekrl)
        add_test(NAME leekwars_reward COMMAND test_reward)

        add_executable(test_useful tests/test_useful.c)
        target_link_libraries(test_useful PRIVATE leekrl)
        add_test(NAME leekwars_useful COMMAND test_useful)

        add_executable(test_moves tests/test_moves.c)
        target_link_libraries(test_moves PRIVATE leekrl)
        add_test(NAME leekwars_moves COMMAND test_moves)

        add_executable(test_vec tests/test_vec.c)
        target_link_libraries(test_vec PRIVATE leekrl)
        add_test(NAME leekwars_vec COMMAND test_vec)
    endif()
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/weapons.json" AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/chips.json" AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/summons.json" AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/components.json")
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    set(LEEKRL_SNAPSHOT_C "${CMAKE_CURRENT_BINARY_DIR}/item_snapshot.c")
    add_custom_command(
        OUTPUT "${LEEKRL_SNAPSHOT_C}"
        COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_item_snapshot.py"
            --weapons "${CMAKE_CURRENT_SOURCE_DIR}/weapons.json"
            --chips "${CMAKE_CURRENT_SOURCE_DIR}/chips.json"
            --summons "${CMAKE_CURRENT_SOURCE_DIR}/summons.json"
            --components "${CMAKE_CURRENT_SOURCE_DIR}/components.json"
            --output "${LEEKRL_SNAPSHOT_C}"
        DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_item_snapshot.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/weapons.json"
            "${CMAKE_CURRENT_SOURCE_DIR}/chips.json"
            "${CMAKE_CURRENT_SOURCE_DIR}/summons.json"
            "${CMAKE_CURRENT_SOURCE_DIR}/components.json"
        VERBATIM)
    target_sources(leekrl PRIVATE "${LEEKRL_SNAPSHOT_C}" src/build.c src/vec.c)
    target_compile_definitions(leekrl PUBLIC LEEKRL_HAS_ITEM_SNAPSHOT=1)

    # Loaded by the Python package in python/leekrl.
    add_library(leekrl_shared SHARED src/shared.c)
    set_target_properties(leekrl_shared PROPERTIES OUTPUT_NAME leekrl PREFIX ""
                          WINDOWS_EXPORT_ALL_SYMBOLS ON)
    if(MINGW)
        target_link_libraries(leekrl_shared PRIVATE
            -Wl,--whole-archive leekrl -Wl,--no-whole-archive
            -static -Wl,--export-all-symbols)
    elseif(MSVC)
        target_link_libraries(leekrl_shared PRIVATE leekrl)
        target_link_options(leekrl_shared PRIVATE "/WHOLEARCHIVE:leekrl")
    else()
        target_link_libraries(leekrl_shared PRIVATE
            -Wl,--whole-archive leekrl -Wl,--no-whole-archive)
    endif()
    if(LEEKRL_RELEASE_LTO)
        target_link_options(leekrl_shared PRIVATE $<$<CONFIG:Release>:-flto=auto>)
    endif()
    if(BUILD_TESTING)
        add_test(NAME leekwars_snapshot_generator
            COMMAND "${Python3_EXECUTABLE}" -B -m unittest discover -s tests -p test_item_snapshot_generator.py)
        set_tests_properties(leekwars_snapshot_generator PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
        add_test(NAME leekwars_python
            COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_python.py")
        # Skips itself if PyTorch is not installed.
        add_test(NAME leekwars_learner
            COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_learner.py")
        set_tests_properties(leekwars_python leekwars_learner PROPERTIES
            ENVIRONMENT "LEEKRL_LIBRARY=$<TARGET_FILE:leekrl_shared>")
    endif()
endif()
