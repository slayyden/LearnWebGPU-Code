# custom CMakeLists file for imgui

add_library(imgui STATIC
    # connect imgui to GLFW and webGPU
    backends/imgui_impl_wgpu.h 
    backends/imgui_impl_wgpu.cpp 
    backends/imgui_impl_glfw.h
    backends/imgui_impl_glfw.cpp

    # add some C++ features (imgui is a C library)
    misc/cpp/imgui_stdlib.h 
    misc/cpp/imgui_stdlib.cpp 

    # core imgui files
    imconfig.h
    imgui.h
    imgui.cpp
    imgui_draw.cpp
    imgui_internal.h
    imgui_tables.cpp
    imgui_widgets.cpp
    imstb_rectpack.h
    imstb_textedit.h
    imstb_truetype.h

)
target_include_directories(imgui PUBLIC .)
target_link_libraries(imgui PUBLIC webgpu glfw)


# C++11 is required by ImGui
set_target_properties(imgui PROPERTIES
	CXX_STANDARD 11
	CXX_STANDARD_REQUIRED ON
)