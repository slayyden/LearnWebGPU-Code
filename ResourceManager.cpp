#include "ResourceManager.h"
#include "stb_image.h"
#include "tiny_obj_loader.h"

using namespace wgpu;
// Auxiliary function for loadTexture
void writeMipMaps(
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
        std::cout << "writing mipmap level " << level << std::endl;
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

Texture ResourceManager::loadTexture(const path& path, Device device, TextureView* pTextureView) {
    std::cout << "loadTexture called" << std::endl;
    int width, height, channels;
    unsigned char *pixelData = stbi_load(path.string().c_str(), &width, &height, &channels, 4 /* force 4 channels */);
    if (nullptr == pixelData) return nullptr;
    std::cout << "pixel data loaded" << std::endl;

    TextureDescriptor textureDesc{};
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
    std::cout << "about to create texture" << std::endl;
    Texture texture = device.createTexture(textureDesc);
    std::cout << "texture created on device" << std::endl;
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

bool ResourceManager::loadGeometryFromObj(const path &path, std::vector<VertexAttributes> &vertexData) {
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

ShaderModule ResourceManager::loadShaderModule(const path& path, wgpu::Device device) {
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