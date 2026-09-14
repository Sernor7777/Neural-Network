file(GLOB_RECURSE SOURCES
    CONFIGURE_DEPENDS
    src/*.cpp
)

file(GLOB_RECURSE HEADERS
    CONFIGURE_DEPENDS
    include/*.hpp
)

add_library(neuralnetwork
    ${SOURCES}
    ${HEADERS}
)

target_link_libraries(neuralnetwork
    PUBLIC Eigen3::Eigen
)

target_include_directories(neuralnetwork
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)