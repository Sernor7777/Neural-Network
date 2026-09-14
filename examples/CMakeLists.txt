file(GLOB_RECURSE SOURCES
    CONFIGURE_DEPENDS
    *.cpp
)

add_executable(example
    ${SOURCES}
)

target_link_libraries(example
    PRIVATE
        neuralnetwork
)