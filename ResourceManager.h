#pragma once
#include <webgpu/webgpu.hpp>
#include <filesystem>
#include <glm/glm.hpp> // all types inspired from GLSL
#include <glm/ext.hpp>
#include <fstream>
#include <sstream>


class ResourceManager {
public:
    // (Just aliases to make notations lighter)
    using path = std::filesystem::path;
    using vec3 = glm::vec3;
    using vec2 = glm::vec2;

    // @brief A structure that describes the data layout in the vertex buffer,
    // used by loadGeometryFromObj and used it in `sizeof` and `offsetof`
    // when uploading data to the GPU.
    struct VertexAttributes {
        vec3 position;
        vec3 normal;
        vec3 color;
        vec2 uv;
    };

    /// @brief Load a shader from a WGSL file into a new shader module
    /// @param path path of the shader file
    /// @param device the WGPU device
    /// @return the new shader module
    static wgpu::ShaderModule loadShaderModule(const path& path, wgpu::Device device);


    /// @brief Load an 3D mesh from a standard .obj file into a vertex data buffer
    /// @param path path of the obj file
    /// @param vertexData a vector of VertexAttributes to load the .obj data into
    /// @return true iff the geometry was successfully loaded 
    static bool loadGeometryFromObj(const path& path, std::vector<VertexAttributes>& vertexData);

    // 

    /// @brief Load an image from a standard image file into a new texture object
    //  The texture must be destroyed after use
    /// @param path the path of the texture file
    /// @param device the WGPU device
    /// @param pTextureView a wgpu::TextureView of the texture
    /// @return a wgpu::Texture of the texture
    static wgpu::Texture loadTexture(const path& path, wgpu::Device device, wgpu::TextureView* pTextureView = nullptr);
};
