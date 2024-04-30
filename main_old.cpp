/**
 * This file is part of the "Learn WebGPU for C++" book.
 *   https://github.com/eliemichel/LearnWebGPU
 *
 * MIT License
 * Copyright (c) 2022-2023 Elie Michel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "glm/ext/matrix_clip_space.hpp"
#include <GLFW/glfw3.h>
#include <glfw3webgpu.h>

#define TINYOBJLOADER_IMPLEMENTATION // add this to exactly 1 of your C++ files
#include "tiny_obj_loader.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_LEFT_HANDED
#include <glm/glm.hpp> // all types inspired from GLSL
#include <glm/ext.hpp>

#define WEBGPU_CPP_IMPLEMENTATION
#include <webgpu/webgpu.hpp>

#include <cassert>
#include <iostream>

#include <array>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include <cstdlib>

namespace fs = std::filesystem;
using glm::mat4x4;
using glm::vec2;
using glm::vec3;
using glm::vec4;

/**
 * The same structure as in the shader, replicated in C++
 */
struct MyUniforms {
  mat4x4 projectionMatrix;
  mat4x4 viewMatrix;
  mat4x4 modelMatrix;
  vec4 color; // or float color[4]
  float time;
  float _pad[3];
};

/**
 * A structure that describes the data layout in the vertex buffer
 * We do not instantiate it but use it in `sizeof` and `offsetof`
 */
struct VertexAttributes {
  vec3 position; 
  vec3 normal; 
  vec3 color;
  vec2 uv;
};

// Have the compiler check byte alignment
static_assert(sizeof(MyUniforms) % 16 == 0);
// load geometry into buffers
bool loadGeometry(const fs::path &path, std::vector<float> &pointData,
                  std::vector<uint16_t> &indexData, const int dimensions) {
  std::ifstream file(path);
  if (!file.is_open()) {
    return false;
  }

  pointData.clear();
  indexData.clear();

  enum class Section {
    None,
    Points,
    Indices,
  };
  Section currentSection = Section::None;

  float value;
  uint16_t index;
  std::string line;
  while (!file.eof()) {
    getline(file, line);

    // overcome the `CRLF` problem
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }

    if (line == "[points]") {
      currentSection = Section::Points;
    } else if (line == "[indices]") {
      currentSection = Section::Indices;
    } else if (line[0] == '#' || line.empty()) {
      // Do nothing, this is a comment
    } else if (currentSection == Section::Points) {
      std::istringstream iss(line);
      // Get x, y, r, g, b
      for (int i = 0; i < dimensions + 3; ++i) {
        iss >> value;
        pointData.push_back(value);
      }
    } else if (currentSection == Section::Indices) {
      std::istringstream iss(line);
      // Get corners #0 #1 and #2
      for (int i = 0; i < 3; ++i) {
        iss >> index;
        indexData.push_back(index);
      }
    }
  }
  return true;
}

bool loadGeometryFromObj(const fs::path& path, std::vector<VertexAttributes>& vertexData) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;

    std::string warn;
    std::string err;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, path.string().c_str());

    if (!warn.empty()) {
        std::cout << warn << std::endl;
    }

    if (!err.empty()) {
        std::cerr << err << std::endl;
    }

    if (!ret) {
        return false;
    }
    
    // WARNING: only works for objects with 1 shape
    // https://eliemichel.github.io/LearnWebGPU/basic-3d-rendering/3d-meshes/loading-from-file.html 
    vertexData.clear();
    for (const auto& shape : shapes) {
      size_t offset = vertexData.size();
      vertexData.resize(offset + shape.mesh.indices.size());
      for (size_t i = 0; i < shape.mesh.indices.size(); ++i) {
          const tinyobj::index_t& idx = shape.mesh.indices[i];

          vertexData[offset + i].position = {
              attrib.vertices[3 * idx.vertex_index + 0],
              -attrib.vertices[3 * idx.vertex_index + 2],
              attrib.vertices[3 * idx.vertex_index + 1]
          };

          vertexData[offset + i].normal = {
              attrib.normals[3 * idx.normal_index + 0],
              attrib.normals[3 * idx.normal_index + 1],
              attrib.normals[3 * idx.normal_index + 2]
          };

          vertexData[offset + i].color = {
              attrib.colors[3 * idx.vertex_index + 0],
              attrib.colors[3 * idx.vertex_index + 2],
              attrib.colors[3 * idx.vertex_index + 1]
          };

          
          vertexData[offset + i].uv = {
              attrib.texcoords[2 * idx.texcoord_index + 0],
              1 - attrib.texcoords[2 * idx.texcoord_index + 1]
          };
      }
    }
    return true;
}
/**
 * Round 'value' up to the next multiplier of 'step'.
 */
uint32_t ceilToNextMultiple(uint32_t value, uint32_t step) {
  uint32_t divide_and_ceil = value / step + (value % step == 0 ? 0 : 1);
  return step * divide_and_ceil;
}

using namespace wgpu;

// create shader module
ShaderModule loadShaderModule(const fs::path &path, Device device) {
  std::ifstream file(path);
  if (!file.is_open()) {
    return nullptr;
  }
  file.seekg(0, std::ios::end);
  size_t size = file.tellg();
  std::string shaderSource(size, ' ');
  file.seekg(0);
  file.read(shaderSource.data(), size);

  ShaderModuleWGSLDescriptor shaderCodeDesc{};
  shaderCodeDesc.chain.next = nullptr;
  shaderCodeDesc.chain.sType = SType::ShaderModuleWGSLDescriptor;
  shaderCodeDesc.code = shaderSource.c_str();
  ShaderModuleDescriptor shaderDesc{};
  // shaderDesc.hintCount = 0;
  // shaderDesc.hints = nullptr;
  shaderDesc.nextInChain = &shaderCodeDesc.chain;
  return device.createShaderModule(shaderDesc);
}

// Auxiliary function for loadTexture
static void writeMipMaps(
    Device device,
    Texture texture,
    Extent3D textureSize,
    [[maybe_unused]] uint32_t mipLevelCount, // not used yet
    const unsigned char* pixelData)
{
    ImageCopyTexture destination;
    destination.texture = texture;
    destination.mipLevel = 0;
    destination.origin = { 0, 0, 0 };
    destination.aspect = TextureAspect::All;

    TextureDataLayout source;
    source.offset = 0;
    source.bytesPerRow = 4 * textureSize.width;
    source.rowsPerImage = textureSize.height;

    Queue queue = device.getQueue();
    queue.writeTexture(destination, pixelData, 4 * textureSize.width * textureSize.height, source, textureSize);

    // Create image data
    Extent3D mipLevelSize = textureSize;
    std::vector<unsigned char> previousLevelPixels;
    Extent3D previousMipLevelSize;
    for (uint32_t level = 0; level < mipLevelCount; ++level) {
        // Pixel data for the current level
        std::vector<unsigned char> pixels(4 * mipLevelSize.width * mipLevelSize.height);
        if (level == 0) {
            // We cannot really avoid this copy since we need this
            // in previousLevelPixels at the next iteration
            memcpy(pixels.data(), pixelData, pixels.size());
        }
        else {
            // Create mip level data
            for (uint32_t i = 0; i < mipLevelSize.width; ++i) {
                for (uint32_t j = 0; j < mipLevelSize.height; ++j) {
                    unsigned char* p = &pixels[4 * (j * mipLevelSize.width + i)];
                    // Get the corresponding 4 pixels from the previous level
                    unsigned char* p00 = &previousLevelPixels[4 * ((2 * j + 0) * previousMipLevelSize.width + (2 * i + 0))];
                    unsigned char* p01 = &previousLevelPixels[4 * ((2 * j + 0) * previousMipLevelSize.width + (2 * i + 1))];
                    unsigned char* p10 = &previousLevelPixels[4 * ((2 * j + 1) * previousMipLevelSize.width + (2 * i + 0))];
                    unsigned char* p11 = &previousLevelPixels[4 * ((2 * j + 1) * previousMipLevelSize.width + (2 * i + 1))];
                    // Average
                    p[0] = (p00[0] + p01[0] + p10[0] + p11[0]) / 4;
                    p[1] = (p00[1] + p01[1] + p10[1] + p11[1]) / 4;
                    p[2] = (p00[2] + p01[2] + p10[2] + p11[2]) / 4;
                    p[3] = (p00[3] + p01[3] + p10[3] + p11[3]) / 4;
                }
            }
        }

        // Upload data to the GPU texture
        destination.mipLevel = level;
        source.bytesPerRow = 4 * mipLevelSize.width;
        source.rowsPerImage = mipLevelSize.height;
        queue.writeTexture(destination, pixels.data(), pixels.size(), source, mipLevelSize);

        previousLevelPixels = std::move(pixels);
        previousMipLevelSize = mipLevelSize;
        mipLevelSize.width /= 2;
        mipLevelSize.height /= 2;
    }
    queue.release();
}


// Equivalent of std::bit_width that is available from C++20 onward
uint32_t bit_width(uint32_t m) {
    if (m == 0) return 0;
    else { uint32_t w = 0; while (m >>= 1) ++w; return w; }
}

Texture loadTexture(const fs::path& path, Device device, TextureView* pTextureView = nullptr) {
    int width, height, channels;
    unsigned char *pixelData = stbi_load(path.string().c_str(), &width, &height, &channels, 4 /* force 4 channels */);
    if (nullptr == pixelData) return nullptr;

    TextureDescriptor textureDesc;
    textureDesc.dimension = TextureDimension::_2D;
    textureDesc.format = TextureFormat::RGBA8Unorm; // by convention for bmp, png and jpg file. Be careful with other formats.
    textureDesc.mipLevelCount = 1;
    textureDesc.sampleCount = 1;
    textureDesc.size = { (unsigned int)width, (unsigned int)height, 1 };
    textureDesc.usage = TextureUsage::TextureBinding | TextureUsage::CopyDst;
    textureDesc.viewFormatCount = 0;
    textureDesc.viewFormats = nullptr;
    textureDesc.size = { (unsigned int)width, (unsigned int)height, 1 };
    textureDesc.mipLevelCount = bit_width(std::max(textureDesc.size.width, textureDesc.size.height));
    Texture texture = device.createTexture(textureDesc);

    // Upload data to the GPU texture (to be implemented!)
    writeMipMaps(device, texture, textureDesc.size, textureDesc.mipLevelCount, pixelData);

    stbi_image_free(pixelData);

    if (pTextureView) {
        TextureViewDescriptor textureViewDesc;
        textureViewDesc.aspect = TextureAspect::All;
        textureViewDesc.baseArrayLayer = 0;
        textureViewDesc.arrayLayerCount = 1;
        textureViewDesc.baseMipLevel = 0;
        textureViewDesc.mipLevelCount = textureDesc.mipLevelCount;
        textureViewDesc.dimension = TextureViewDimension::_2D;
        textureViewDesc.format = textureDesc.format;
        *pTextureView = texture.createView(textureViewDesc);
    }

    return texture;
}
// round up to a multiple of 4
inline size_t aligned_size(size_t n) { return (n + 3) & ~3; }

int main(int, char **) {
  Instance instance = createInstance(InstanceDescriptor{});
  if (!instance) {
    std::cerr << "Could not initialize WebGPU!" << std::endl;
    return 1;
  }

  if (!glfwInit()) {
    std::cerr << "Could not initialize GLFW!" << std::endl;
    return 1;
  }

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
  GLFWwindow *window = glfwCreateWindow(640, 480, "Learn WebGPU", NULL, NULL);
  if (!window) {
    std::cerr << "Could not open window!" << std::endl;
    return 1;
  }

  std::cout << "Requesting adapter..." << std::endl;
  Surface surface = glfwGetWGPUSurface(instance, window);
  RequestAdapterOptions adapterOpts;
  adapterOpts.compatibleSurface = surface;
  Adapter adapter = instance.requestAdapter(adapterOpts);
  std::cout << "Got adapter: " << adapter << std::endl;

  SupportedLimits supportedLimits;
  adapter.getLimits(&supportedLimits);

  // WARNING: this shit is sus
  // Limits deviceLimits = supportedLimits.limits;
  std::cout << "Requesting device..." << std::endl;
  RequiredLimits requiredLimits = Default;
  requiredLimits.limits.maxVertexAttributes = 4;
  requiredLimits.limits.maxVertexBuffers = 1;
  requiredLimits.limits.maxBufferSize = 150000 * sizeof(VertexAttributes);
  requiredLimits.limits.maxVertexBufferArrayStride = sizeof(VertexAttributes);
  requiredLimits.limits.minStorageBufferOffsetAlignment =
      supportedLimits.limits.minStorageBufferOffsetAlignment;
  requiredLimits.limits.minUniformBufferOffsetAlignment =
      supportedLimits.limits.minUniformBufferOffsetAlignment;
  requiredLimits.limits.maxInterStageShaderComponents = 8;
  // We use at most 1 bind group for now
  requiredLimits.limits.maxBindGroups = 1;
  // We use at most 1 uniform buffer per stage
  requiredLimits.limits.maxUniformBuffersPerShaderStage = 1;
  requiredLimits.limits.maxUniformBufferBindingSize = 16 * 4 * sizeof(float);
  // Extra limit requirement
  // requiredLimits.limits.maxDynamicUniformBuffersPerPipelineLayout = 1;
  // For the depth buffer, we enable textures (up to the size of the window):
  requiredLimits.limits.maxTextureDimension1D = 2048;
  requiredLimits.limits.maxTextureDimension2D = 2048;
  requiredLimits.limits.maxTextureArrayLayers = 1;
  requiredLimits.limits.maxSampledTexturesPerShaderStage = 1;
  requiredLimits.limits.maxSamplersPerShaderStage = 1;

  DeviceDescriptor deviceDesc;
  deviceDesc.label = "My Device";
  deviceDesc.requiredFeaturesCount = 0;
  deviceDesc.requiredLimits = &requiredLimits;
  deviceDesc.defaultQueue.label = "The default queue";
  Device device = adapter.requestDevice(deviceDesc);
  std::cout << "Got device: " << device << std::endl;

  bool terminate = false;
  // Add an error callback for more debug info
  auto h = device.setUncapturedErrorCallback(
      [&](ErrorType type, char const *message) {
        std::cout << "Device error: type " << type;
        if (message) {
          std::cout << " (message: " << message << ")";
          terminate = true;
        }
        std::cout << std::endl;
      });

  /*
  uint32_t uniformStride = ceilToNextMultiple(
      (uint32_t)sizeof(MyUniforms),
      (uint32_t)deviceLimits.minUniformBufferOffsetAlignment);
  */
  Queue queue = device.getQueue();

  std::cout << "Creating swapchain..." << std::endl;
  TextureFormat swapChainFormat = TextureFormat::BGRA8Unorm;
  // Dawn doesn't implement this yet
  // TextureFormat swapChainFormat = TextureFormat::BGRA8UnormSrgb;

  SwapChainDescriptor swapChainDesc;
  swapChainDesc.width = 640;
  swapChainDesc.height = 480;
  swapChainDesc.usage = TextureUsage::RenderAttachment;
  swapChainDesc.format = swapChainFormat;
  swapChainDesc.presentMode = PresentMode::Fifo;
  SwapChain swapChain = device.createSwapChain(surface, swapChainDesc);
  std::cout << "Swapchain: " << swapChain << std::endl;

  // create shader module
  std::cout << "Creating shader module..." << std::endl;
  ShaderModule shaderModule =
      loadShaderModule(RESOURCE_DIR "/shader.wgsl", device);
  std::cout << "Shader module: " << shaderModule << std::endl;

  std::cout << "Creating render pipeline..." << std::endl;
  RenderPipelineDescriptor pipelineDesc;

  // Vertex fetch
  std::vector<VertexAttribute> vertexAttribs(4);

  // Position attribute
  vertexAttribs[0].shaderLocation = 0;
  vertexAttribs[0].format = VertexFormat::Float32x3;
  vertexAttribs[0].offset = offsetof(VertexAttributes, position);

  // Normal attribute
  vertexAttribs[1].shaderLocation = 1;
  vertexAttribs[1].format = VertexFormat::Float32x3;
  vertexAttribs[1].offset = offsetof(VertexAttributes, normal);

  // Color attribute
  vertexAttribs[2].shaderLocation = 2;
  vertexAttribs[2].format = VertexFormat::Float32x3;
  vertexAttribs[2].offset = offsetof(VertexAttributes, color);
  
  // UV attribute 
  vertexAttribs[3].shaderLocation = 3;
  vertexAttribs[3].format = VertexFormat::Float32x2; 
  vertexAttribs[3].offset = offsetof(VertexAttributes, uv);

  VertexBufferLayout vertexBufferLayout;
  vertexBufferLayout.attributeCount = (uint32_t)vertexAttribs.size();
  vertexBufferLayout.attributes = vertexAttribs.data();
  vertexBufferLayout.arrayStride = sizeof(VertexAttributes);
  vertexBufferLayout.stepMode = VertexStepMode::Vertex;

  pipelineDesc.vertex.bufferCount = 1;
  pipelineDesc.vertex.buffers = &vertexBufferLayout;

  pipelineDesc.vertex.module = shaderModule;
  pipelineDesc.vertex.entryPoint = "vs_main";
  pipelineDesc.vertex.constantCount = 0;
  pipelineDesc.vertex.constants = nullptr;

  pipelineDesc.primitive.topology = PrimitiveTopology::TriangleList;
  pipelineDesc.primitive.stripIndexFormat = IndexFormat::Undefined;
  pipelineDesc.primitive.frontFace = FrontFace::CCW;
  pipelineDesc.primitive.cullMode = CullMode::None;

  FragmentState fragmentState;
  pipelineDesc.fragment = &fragmentState;
  fragmentState.module = shaderModule;
  fragmentState.entryPoint = "fs_main";
  fragmentState.constantCount = 0;
  fragmentState.constants = nullptr;

  BlendState blendState;
  blendState.color.srcFactor = BlendFactor::SrcAlpha;
  blendState.color.dstFactor = BlendFactor::OneMinusSrcAlpha;
  blendState.color.operation = BlendOperation::Add;
  blendState.alpha.srcFactor = BlendFactor::Zero;
  blendState.alpha.dstFactor = BlendFactor::One;
  blendState.alpha.operation = BlendOperation::Add;

  ColorTargetState colorTarget;
  colorTarget.format = swapChainFormat;
  colorTarget.blend = &blendState;
  colorTarget.writeMask = ColorWriteMask::All;

  fragmentState.targetCount = 1;
  fragmentState.targets = &colorTarget;

  DepthStencilState depthStencilState = Default;
  depthStencilState.depthCompare = CompareFunction::Less;
  depthStencilState.depthWriteEnabled = true;
  // store the format in a variable since we'll also use it later
  TextureFormat depthTextureFormat = TextureFormat::Depth24Plus;
  depthStencilState.format = depthTextureFormat;
  // deactivate the stencil
  depthStencilState.stencilReadMask = 0;
  depthStencilState.stencilWriteMask = 0;

  // Create the depth texture
  TextureDescriptor depthTextureDesc;
  depthTextureDesc.dimension = TextureDimension::_2D;
  depthTextureDesc.format = depthTextureFormat;
  depthTextureDesc.mipLevelCount = 1;
  depthTextureDesc.sampleCount = 1;
  depthTextureDesc.size = {640, 480, 1};
  depthTextureDesc.usage = TextureUsage::RenderAttachment;
  depthTextureDesc.viewFormatCount = 1;
  depthTextureDesc.viewFormats = (WGPUTextureFormat *)&depthTextureFormat;
  Texture depthTexture = device.createTexture(depthTextureDesc);

  // Create the view of the depth texture manipulated by the rasterizer
  TextureViewDescriptor depthTextureViewDesc;
  depthTextureViewDesc.aspect = TextureAspect::DepthOnly;
  depthTextureViewDesc.baseArrayLayer = 0;
  depthTextureViewDesc.arrayLayerCount = 1;
  depthTextureViewDesc.baseMipLevel = 0;
  depthTextureViewDesc.mipLevelCount = 1;
  depthTextureViewDesc.dimension = TextureViewDimension::_2D;
  depthTextureViewDesc.format = depthTextureFormat;
  TextureView depthTextureView = depthTexture.createView(depthTextureViewDesc);

  pipelineDesc.depthStencil = &depthStencilState;

  pipelineDesc.multisample.count = 1;
  pipelineDesc.multisample.mask = ~0u;
  pipelineDesc.multisample.alphaToCoverageEnabled = false;
    // Create a texture
  TextureView textureView = nullptr;
  Texture texture = loadTexture(RESOURCE_DIR "/fourareen2K_albedo.jpg", device, &textureView);
  if (!texture) {
      std::cerr << "Could not load texture!" << std::endl;
      return 1;
  }

  /*
  TextureDescriptor textureDesc; 
  textureDesc.dimension = TextureDimension::_2D; 
  textureDesc.size = {256, 256, 1};
  textureDesc.mipLevelCount = 8;
  textureDesc.sampleCount = 1; // only store 1 color per texel
  textureDesc.format = TextureFormat::RGBA8Unorm; 
  // tells allocator where to place texture in GPU memory 
  // Binding means we can use it in a shader 
  // CopyDst means we will be copying pixel data from C++
  textureDesc.usage = TextureUsage::TextureBinding | TextureUsage::CopyDst;
  textureDesc.viewFormatCount = 0;
  textureDesc.viewFormats = nullptr;
  Texture texture = device.createTexture(textureDesc);

  TextureViewDescriptor textureViewDesc; 
  textureViewDesc.aspect = TextureAspect::All; 
  textureViewDesc.baseArrayLayer = 0; 
  textureViewDesc.arrayLayerCount = 1; 
  textureViewDesc.baseMipLevel = 0; 
  textureViewDesc.mipLevelCount = textureDesc.mipLevelCount; 
  textureViewDesc.dimension = TextureViewDimension::_2D; 
  textureViewDesc.format = textureDesc.format; 
  TextureView textureView = texture.createView(textureViewDesc);
  */ 

  // Create a sampler
  SamplerDescriptor samplerDesc;
  samplerDesc.addressModeU = AddressMode::Repeat;
  samplerDesc.addressModeV = AddressMode::Repeat;
  samplerDesc.addressModeW = AddressMode::ClampToEdge;
  samplerDesc.magFilter = FilterMode::Linear;
  samplerDesc.minFilter = FilterMode::Linear;
  samplerDesc.mipmapFilter = MipmapFilterMode::Linear;
  samplerDesc.lodMinClamp = 0.0f;
  samplerDesc.lodMaxClamp = 8.0f;
  samplerDesc.compare = CompareFunction::Undefined;
  samplerDesc.maxAnisotropy = 1;
  Sampler sampler = device.createSampler(samplerDesc);
    
  std::vector<BindGroupLayoutEntry> bindingLayoutEntries(3, Default);
  // bind group layout for unfiorm buffer
  BindGroupLayoutEntry& bindingLayout = bindingLayoutEntries[0];
  // The binding index as used in the @binding attribute in the shader
  bindingLayout.binding = 0;
  // The stage that needs to access this resource
  bindingLayout.visibility = ShaderStage::Vertex | ShaderStage::Fragment;
  bindingLayout.buffer.type = BufferBindingType::Uniform;
  bindingLayout.buffer.minBindingSize = sizeof(MyUniforms);
  
  BindGroupLayoutEntry& textureBindingLayout = bindingLayoutEntries[1];
  textureBindingLayout.binding = 1; 
  textureBindingLayout.visibility = ShaderStage::Fragment; 
  textureBindingLayout.texture.sampleType = TextureSampleType::Float;
  textureBindingLayout.texture.viewDimension = TextureViewDimension::_2D;

  BindGroupLayoutEntry& samplerBindingLayout = bindingLayoutEntries[2];
  samplerBindingLayout.binding = 2; 
  samplerBindingLayout.visibility = ShaderStage::Fragment;
  samplerBindingLayout.sampler.type = SamplerBindingType::Filtering; 

  // Create a bind group layout
  BindGroupLayoutDescriptor bindGroupLayoutDesc{};
  bindGroupLayoutDesc.entryCount = (uint32_t)bindingLayoutEntries.size();
  bindGroupLayoutDesc.entries = bindingLayoutEntries.data();
  BindGroupLayout bindGroupLayout =
      device.createBindGroupLayout(bindGroupLayoutDesc);

  // Create the pipeline layout
  PipelineLayoutDescriptor layoutDesc{};
  layoutDesc.bindGroupLayoutCount = 1;
  layoutDesc.bindGroupLayouts = (WGPUBindGroupLayout *)&bindGroupLayout;
  PipelineLayout layout = device.createPipelineLayout(layoutDesc);
  pipelineDesc.layout = layout;

  RenderPipeline pipeline = device.createRenderPipeline(pipelineDesc);
  std::cout << "Render pipeline: " << pipeline << std::endl;
  // Vertex buffer
  // Read buffers from file
  std::vector<float> pointData;
  std::vector<uint16_t> indexData;
  
  std::vector<VertexAttributes> vertexData;
  bool success = loadGeometryFromObj(RESOURCE_DIR "/fourareen.obj", vertexData);
  if (!success) {
    std::cerr << "Could not load geometry!" << std::endl;
    return 1;
  }

  BufferDescriptor bufferDesc{};
  // Create vertex buffer
  bufferDesc.size = vertexData.size() * sizeof(VertexAttributes);
  bufferDesc.usage = BufferUsage::CopyDst | BufferUsage::Vertex;
  bufferDesc.mappedAtCreation = false;
  Buffer vertexBuffer = device.createBuffer(bufferDesc);
  queue.writeBuffer(vertexBuffer, 0, vertexData.data(), bufferDesc.size);

  // Index Buffer alignment
  int indexCount = static_cast<int>(vertexData.size());

  // Create uniform buffer
  bufferDesc.size = sizeof(MyUniforms);
  // Make sure to flag the buffer as BufferUsage::Uniform
  bufferDesc.usage = BufferUsage::CopyDst | BufferUsage::Uniform;
  bufferDesc.mappedAtCreation = false;
  Buffer uniformBuffer = device.createBuffer(bufferDesc);
  // Upload the initial value of the uniforms
  MyUniforms uniforms;
  // upload first value
  uniforms.time = 1.0f;
  uniforms.color = {0.4f, 0.0f, 1.0f, 1.0f};

  constexpr float PI = 3.14159265358979323846f;
  
  // Matrices
  uniforms.modelMatrix = mat4x4(1.0);
  uniforms.viewMatrix = glm::lookAt(vec3(-2.0f, -3.0f, 2.0f), vec3(0.0f), vec3(0, 0, 1));
  uniforms.projectionMatrix = glm::perspective(45 * PI / 180, 640.0f / 480.0f, 0.01f, 100.0f);
  queue.writeBuffer(uniformBuffer, 0, &uniforms, sizeof(MyUniforms));

  // Create a binding
  std::vector<BindGroupEntry> bindings(3);
  BindGroupEntry binding{};
  // The index of the binding (the entries in bindGroupDesc can be in any order)
  bindings[0].binding = 0;
  // The buffer it is actually bound to
  bindings[0].buffer = uniformBuffer;
  // We can specify an offset within the buffer, so that a single buffer can
  // hold multiple uniform blocks.
  bindings[0].offset = 0;
  // And we specify again the size of the buffer.
  bindings[0].size = sizeof(MyUniforms);

  bindings[1].binding = 1; 
  // can pass a texture view, but NOT a texture
  bindings[1].textureView = textureView;  

  bindings[2].binding = 2; 
  bindings[2].sampler = sampler; 
  // A bind group contains one or multiple bindings
  BindGroupDescriptor bindGroupDesc{};
  bindGroupDesc.layout = bindGroupLayout;
  // There must be as many bindings as declared in the layout!
  bindGroupDesc.entryCount = (uint32_t)bindings.size();
  bindGroupDesc.entries = bindings.data();
  BindGroup bindGroup = device.createBindGroup(bindGroupDesc);

  int loopcount = 0;
  
  while (!glfwWindowShouldClose(window) and loopcount < 3) {
    // terminate the program upon a Dawn error callback
    // std::cout << "terminate: " << terminate << std::endl;
    if (terminate) {
      std::cout << "Error occurred. Terminating..." << std::endl;
      return 1;
    }
    // ++loopcount;
    // TODO: remember what this does
    glfwPollEvents();

    // Update uniform buffer
    // Upload only the time, whichever its order in the struct
    // uniforms.time = static_cast<float>(glfwGetTime());
    uniforms.time = static_cast<float>(glfwGetTime());
    queue.writeBuffer(uniformBuffer, offsetof(MyUniforms, time), &uniforms.time,
                      sizeof(MyUniforms::time));
    
    /*
    // Update view matrix
		angle1 = uniforms.time;
		R1 = glm::rotate(mat4x4(1.0), angle1, vec3(0.0, 0.0, 1.0));
		uniforms.modelMatrix = R1 * T1 * S;
		queue.writeBuffer(uniformBuffer, offsetof(MyUniforms, modelMatrix), &uniforms.modelMatrix, sizeof(MyUniforms::modelMatrix));
    */
    
    /*
    float viewZ = glm::mix(0.0f, 0.25f, cos(2 * PI * uniforms.time / 4) * 0.5 + 0.5);
    uniforms.viewMatrix = glm::lookAt(vec3(-0.5f, -1.5f, viewZ + 0.25f), vec3(0.0f), vec3(0, 0, 1));
    queue.writeBuffer(uniformBuffer, offsetof(MyUniforms, viewMatrix), &uniforms.viewMatrix, sizeof(MyUniforms::viewMatrix));
    */

    TextureView nextTexture = swapChain.getCurrentTextureView();

    if (!nextTexture) {
      std::cerr << "Cannot acquire next swap chain texture" << std::endl;
      return 1;
    }

    CommandEncoderDescriptor commandEncoderDesc;
    commandEncoderDesc.label = "Command Encoder";
    CommandEncoder encoder = device.createCommandEncoder(commandEncoderDesc);

    RenderPassDescriptor renderPassDesc;

    RenderPassColorAttachment renderPassColorAttachment{};
    renderPassColorAttachment.view = nextTexture;
    renderPassColorAttachment.resolveTarget = nullptr;
    renderPassColorAttachment.loadOp = LoadOp::Clear;
    renderPassColorAttachment.storeOp = StoreOp::Store;
    renderPassColorAttachment.clearValue = Color{0.256, 0.256, 0.256, 1.0};
    renderPassDesc.colorAttachmentCount = 1;
    renderPassDesc.colorAttachments = &renderPassColorAttachment;

    // add depth stencil attachment
    RenderPassDepthStencilAttachment depthStencilAttachment;
    // set up depth/stencil attachment
    depthStencilAttachment.view = depthTextureView;
    // The initial value of the depth buffer, meaning "far"
    depthStencilAttachment.depthClearValue = 1.0f;
    // Operation settings comparable to the color attachment
    depthStencilAttachment.depthLoadOp = LoadOp::Clear;
    depthStencilAttachment.depthStoreOp = StoreOp::Store;
    // we could turn off writing to the depth buffer globally here
    depthStencilAttachment.depthReadOnly = false;
    // Stencil setup, mandatory but unused
    depthStencilAttachment.stencilClearValue = 0;
    // dawn specific load and store ops
    depthStencilAttachment.stencilLoadOp = LoadOp::Undefined;
    depthStencilAttachment.stencilStoreOp = StoreOp::Undefined;
    // WARNING: allegedly needed for Dawn but it syntax errors out
    // constexpr auto NaNf = std::numeric_limits<float>::quiet_NaN();
    // depthStencilAttachment.clearDepth = NaNf;
    depthStencilAttachment.stencilReadOnly = true;
    renderPassDesc.depthStencilAttachment = &depthStencilAttachment;
    renderPassDesc.timestampWriteCount = 0;
    renderPassDesc.timestampWrites = nullptr;
    RenderPassEncoder renderPass = encoder.beginRenderPass(renderPassDesc);

    renderPass.setPipeline(pipeline);

    renderPass.setVertexBuffer(0, vertexBuffer, 0,
                               vertexData.size() * sizeof(VertexAttributes));
    // set bind group
    renderPass.setBindGroup(0, bindGroup, 0, nullptr);
    renderPass.draw(indexCount, 1, 0, 0);

    renderPass.end();

    nextTexture.release();

    CommandBufferDescriptor cmdBufferDescriptor;
    cmdBufferDescriptor.label = "Command buffer";
    CommandBuffer command = encoder.finish(cmdBufferDescriptor);
    queue.submit(command);

    swapChain.present();

#ifdef WEBGPU_BACKEND_DAWN
    // Check for pending error callbacks
    device.tick();
#endif
  }
    
  texture.destroy()                   ; 
  texture.release();
  // Destroy the depth texture and its view
  depthTextureView.release();
  depthTexture.destroy();
  depthTexture.release();

  vertexBuffer.destroy();
  vertexBuffer.release();

  swapChain.release();
  device.release();
  adapter.release();
  instance.release();
  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
