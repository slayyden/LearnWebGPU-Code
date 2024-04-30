// In Application.cpp
#include "Application.h"
#include "GLFW/glfw3.h"
#include "glfw3webgpu.h"
#include "webgpu/webgpu.hpp"
#include "ResourceManager.h"
#include <imgui.h>
#include <backends/imgui_impl_wgpu.h>
#include <backends/imgui_impl_glfw.h>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_LEFT_HANDED
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <glm/gtx/polar_coordinates.hpp>

using namespace wgpu;
using glm::vec2;
using glm::vec3;
using glm::vec4;
using glm::mat4x4;

namespace ImGui {
  bool DragDirection(const char* label, vec4& direction) {
    vec2 angles = glm::degrees(glm::polar(vec3(direction))); 
    bool changed = ImGui::DragFloat2(label, glm::value_ptr(angles)); 
    direction = vec4(glm::euclidean(glm::radians(angles)), direction.w); 
    return changed;
  }
}


using VertexAttributes = ResourceManager::VertexAttributes;

constexpr float PI = 3.14159265358979323846f;

bool Application::onInit() {
    if (!initWindowAndDevice()) return false;
  
    // In Application.cpp, in onInit()
    m_errorCallbackHandle = m_device.setUncapturedErrorCallback([](ErrorType type, char const* message) {
        std::cout << "Device error: type " << type;
        if (message) {
          std::cout << " (message: " << message << ")";
          // m_terminate = true;
        } 
        std::cout << std::endl;
    });

    if (!initSwapChain()) return false;
    if (!initDepthBuffer()) return false;
    if (!initRenderPipeline()) return false;
    if (!initTexture()) return false;
    if (!initGeometry()) return false;
    if (!initUniforms()) return false;
    if (!initBindGroup()) return false;
    if (!initGui()) return false; 
    return true;
}

void Application::onFrame() {
  glfwPollEvents(); 
  // if (m_terminate) return;

  // update time
  m_uniforms.time = static_cast<int>(glfwGetTime());
  m_queue.writeBuffer(m_uniformBuffer, offsetof(MyUniforms, time), 
    &m_uniforms.time, sizeof(MyUniforms::time));

  // get new display texture to write to
  TextureView nextTexture = m_swapChain.getCurrentTextureView();  
  if (!nextTexture) {
      std::cerr << "Cannot acquire next swap chain texture" << std::endl;
      return;
  }

  if (m_lightingUniformsChanged)  {
    updateLightingUniforms();
    m_lightingUniformsChanged = false;
  }


  // set up command encoder
  CommandEncoderDescriptor commandEncoderDesc{}; 
  commandEncoderDesc.label = "Command Encoder";
  CommandEncoder encoder = m_device.createCommandEncoder(commandEncoderDesc);

  // set up render pass
  RenderPassDescriptor renderPassDesc{}; 
  // color attachment
  RenderPassColorAttachment renderPassColorAttachment{}; 
  renderPassColorAttachment.view = nextTexture;
  renderPassColorAttachment.resolveTarget = nullptr; 
  renderPassColorAttachment.loadOp = LoadOp::Clear;  
  renderPassColorAttachment.storeOp = StoreOp::Store;
  renderPassColorAttachment.clearValue = Color{0.256, 0.256, 0.256, 1.0};
  renderPassDesc.colorAttachmentCount = 1; 
  renderPassDesc.colorAttachments = &renderPassColorAttachment;
  // depth buffer and stencil
  RenderPassDepthStencilAttachment depthStencilAttachment;
  depthStencilAttachment.view = m_depthTextureView; 
  // initial value of depth buffer representing 'far'
  depthStencilAttachment.depthClearValue = 1.0f; 
  depthStencilAttachment.depthLoadOp = LoadOp::Clear; 
  depthStencilAttachment.depthStoreOp = StoreOp::Store; 
  depthStencilAttachment.depthReadOnly = false; 
  depthStencilAttachment.stencilClearValue = 0;
  depthStencilAttachment.stencilLoadOp = LoadOp::Undefined;
  depthStencilAttachment.stencilStoreOp = StoreOp::Undefined; 
  depthStencilAttachment.stencilReadOnly = true; 
  renderPassDesc.depthStencilAttachment = & depthStencilAttachment;
  renderPassDesc.timestampWriteCount = 0; 
  renderPassDesc.timestampWrites = nullptr; 
  RenderPassEncoder renderPass = encoder.beginRenderPass(renderPassDesc);

  renderPass.setPipeline(m_pipeline); 
  // WARINING: this differs
  renderPass.setVertexBuffer(0, m_vertexBuffer, 0, 
    m_indexCount * sizeof(VertexAttributes));
  renderPass.setBindGroup(0, m_bindGroup, 0, nullptr); 
  renderPass.draw(m_indexCount, 1, 0, 0);

  // update the GUI
  updateGui(renderPass);

  renderPass.end(); 
  nextTexture.release(); 
  // destroy the encoder
  CommandBufferDescriptor cmdBufferDescriptor;
  cmdBufferDescriptor.label = "Command buffer";
  CommandBuffer command = encoder.finish(cmdBufferDescriptor);
  m_queue.submit(command);

  // present the rendered texture
  m_swapChain.present(); 
#ifdef WEBGPU_BACKEND_DAWN
    // Check for pending error callbacks
    m_device.tick();
#endif
}

void Application::onFinish() {
    terminateGui();
    terminateBindGroup();
    terminateUniforms();
    terminateGeometry();
    terminateTexture();
    terminateRenderPipeline();
    terminateDepthBuffer();
    terminateSwapChain();
    terminateWindowAndDevice();
}

bool Application::isRunning() {
  return !glfwWindowShouldClose(m_window);
}

bool Application::initWindowAndDevice() {
  // create instance
  Instance m_instance = createInstance(InstanceDescriptor{});
  if (!m_instance) {
    std::cerr << "Could not initialize WebGPU!" << std::endl;
    return false;
  }
  
  // intialize GLFW
  if (!glfwInit()) {
    std::cerr << "Could not initialize GLFW!" << std::endl;
    return false;
  }
  
  // create window
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
  m_window = glfwCreateWindow(640, 480, "Learn WEBGPU", NULL, NULL);
  if (!m_window) {
    std::cerr << "Could not open window!" << std::endl;
    return false;
  }
  
  // window callbacks
  // set up callback for window resizing
  // Set the user pointer to be "this"
  glfwSetWindowUserPointer(m_window, this);
  // Use a non-capturing lambda as resize callback
  glfwSetFramebufferSizeCallback(m_window, [](GLFWwindow* window, int, int){
      auto that = reinterpret_cast<Application*>(glfwGetWindowUserPointer(window));
      if (that != nullptr) that->onResize();
  });
  glfwSetCursorPosCallback(m_window, [](GLFWwindow* window, double xpos, double ypos){
    auto that = reinterpret_cast<Application*>(glfwGetWindowUserPointer(window));
    if (that != nullptr) that->onMouseMove(xpos, ypos);
  });
  glfwSetMouseButtonCallback(m_window, [](GLFWwindow* window, int button, int action, int mods){
      auto that = reinterpret_cast<Application*>(glfwGetWindowUserPointer(window));
      if (that != nullptr) that->onMouseButton(button, action, mods);
  });
  glfwSetScrollCallback(m_window, [](GLFWwindow* window, double xoffset, double yoffset) {
    auto that = reinterpret_cast<Application*>(glfwGetWindowUserPointer(window));
    if (that != nullptr) that->onScroll(xoffset, yoffset);
  });


  // get adapter
  std::cout << "Requesting adapter..." << std::endl;
  m_surface = glfwGetWGPUSurface(m_instance, m_window);
  RequestAdapterOptions adapterOpts;
  adapterOpts.compatibleSurface = m_surface;
  Adapter adapter = m_instance.requestAdapter(adapterOpts);
  std::cout << "Got adapter: " << adapter << std::endl;
  
  // get limits
  SupportedLimits supportedLimits;
  adapter.getLimits(&supportedLimits);
  
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
  requiredLimits.limits.maxInterStageShaderComponents = 11;
  requiredLimits.limits.maxBindGroups = 2;
  requiredLimits.limits.maxUniformBuffersPerShaderStage = 1;
  requiredLimits.limits.maxUniformBufferBindingSize = 16 * 4 * sizeof(float);
  // requiredLimits.limits.maxDynamicUniformBuffersPerPipelineLayout = 1;
  requiredLimits.limits.maxTextureDimension1D = 2048;
  requiredLimits.limits.maxTextureDimension2D = 2048;
  requiredLimits.limits.maxTextureArrayLayers = 1;
  requiredLimits.limits.maxSampledTexturesPerShaderStage = 1;
  requiredLimits.limits.maxSamplersPerShaderStage = 1;
  
  // get device
  DeviceDescriptor deviceDesc; 
  deviceDesc.label = "My Device"; 
  deviceDesc.requiredFeaturesCount = 0;
  deviceDesc.requiredLimits = &requiredLimits;
  deviceDesc.defaultQueue.label = "The default queue";
  m_device = adapter.requestDevice(deviceDesc);
  std::cout << "Got device: " << m_device << std::endl;
  
  m_queue = m_device.getQueue();

  
  return m_device != nullptr and m_queue != nullptr;
}

void Application::onResize() {
  // Terminate in reverse order 
  terminateDepthBuffer(); 
  terminateSwapChain(); 

  // Reinit 
  initSwapChain(); 
  initDepthBuffer();
  updateProjectionMatrix();
}

void Application::updateProjectionMatrix() {
  int width, height;
  glfwGetFramebufferSize(m_window, &width, &height);

  float ratio = width / (float)height;

  m_uniforms.projectionMatrix = glm::perspective(45 * PI / 180, ratio, 0.01f, 100.0f);
  m_queue.writeBuffer(
      m_uniformBuffer,
      offsetof(MyUniforms, projectionMatrix),
      &m_uniforms.projectionMatrix,
      sizeof(MyUniforms::projectionMatrix)
  );
}

void Application::updateViewMatrix() {
  float cx = cos(m_cameraState.angles.x);
  float sx = sin(m_cameraState.angles.x);
  float cy = cos(m_cameraState.angles.y);
  float sy = sin(m_cameraState.angles.y);

  vec3 position = vec3(cx * cy, sx * cy, sy) * std::exp(-m_cameraState.zoom);
  m_uniforms.viewMatrix = glm::lookAt(position, vec3(0.0f), vec3(0, 0, 1));

  m_queue.writeBuffer(
      m_uniformBuffer,
      offsetof(MyUniforms, viewMatrix),
      &m_uniforms.viewMatrix,
      sizeof(MyUniforms::viewMatrix)
  );


  m_uniforms.cameraWorldPosition = position;
  m_device.getQueue().writeBuffer(
      m_uniformBuffer,
      offsetof(MyUniforms, cameraWorldPosition),
      &m_uniforms.cameraWorldPosition,
      sizeof(MyUniforms::cameraWorldPosition)
  );
}

void Application::onMouseMove(double xpos, double ypos) {
  if (m_drag.active) {
    // why is xpos negated?
    vec2 currentMouse = vec2(-(float)xpos, (float)ypos); 
    vec2 delta = (currentMouse - m_drag.startMouse) * m_drag.sensitivity;
    m_cameraState.angles = m_drag.startCameraState.angles + delta; 
    // clamp to avoid looking too far up/down
    m_cameraState.angles.y = glm::clamp(m_cameraState.angles.y, -PI / 2 + 1e-5f, PI / 2 - 1e-5f);
    updateViewMatrix();
  }
}

void Application::onMouseButton(int button, int action, int /*mods*/) {

  ImGuiIO& io = ImGui::GetIO();
  // dont rotate the camera if ImGui is capturing the mouse interaction
  if (io.WantCaptureMouse) {
    return;
  }
  if (button == GLFW_MOUSE_BUTTON_LEFT) {
    switch(action) {
      case GLFW_PRESS:
        m_drag.active = true; 
        double xpos, ypos; 
        glfwGetCursorPos(m_window, &xpos, &ypos);
        m_drag.startMouse = vec2(-(float)xpos, (float)ypos);
        m_drag.startCameraState = m_cameraState;
        break;
      case GLFW_RELEASE:
        m_drag.active = false; 
        break;
    }
  }
}

void Application::onScroll(double /*xoffset*/, double yoffset) {
  m_cameraState.zoom += m_drag.scrollSensitivity * static_cast<float>(yoffset);
  m_cameraState.zoom = glm::clamp(m_cameraState.zoom, -2.0f, 2.0f);
  updateViewMatrix();
}

void Application::terminateWindowAndDevice(){
  m_queue.release();
  m_device.release();
  m_surface.release();
  m_instance.release(); 
  glfwDestroyWindow(m_window);
  glfwTerminate();
}

bool Application::initSwapChain() {
  std::cout << "Creating swapchain..." << std::endl;

  // width and height of the window in pixels
  int width, height; 
  glfwGetFramebufferSize(m_window, &width, &height);
  
  SwapChainDescriptor swapChainDesc; 
  swapChainDesc.width = static_cast<uint32_t>(width);
  swapChainDesc.height = static_cast<uint32_t>(height);
  swapChainDesc.usage = TextureUsage::RenderAttachment;
  swapChainDesc.format = m_swapChainFormat; 
  swapChainDesc.presentMode = PresentMode::Fifo;
  m_swapChain = m_device.createSwapChain(m_surface, swapChainDesc);
  std::cout << "Swapchain: " << m_swapChain << std::endl;

  return m_swapChain != nullptr;
}

void Application::terminateSwapChain() {
  m_swapChain.release();
}

bool Application::initDepthBuffer() {
  // init depth stencil state
  m_depthTextureFormat = TextureFormat::Depth24Plus;

  int width, height;
  glfwGetFramebufferSize(m_window, &width, &height);

  TextureDescriptor depthTextureDesc; 
  depthTextureDesc.dimension = TextureDimension::_2D;
  depthTextureDesc.format = m_depthTextureFormat;
  depthTextureDesc.mipLevelCount = 1; 
  depthTextureDesc.sampleCount = 1; 
  depthTextureDesc.size = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};
  depthTextureDesc.usage = TextureUsage::RenderAttachment;
  depthTextureDesc.viewFormatCount = 1; 
  depthTextureDesc.viewFormats = (WGPUTextureFormat*)&m_depthTextureFormat; 
  m_depthTexture = m_device.createTexture(depthTextureDesc);

  // Create the view of the depth texture manipulated by the rasterizer
  TextureViewDescriptor depthTextureViewDesc;
  depthTextureViewDesc.aspect = TextureAspect::DepthOnly;
  depthTextureViewDesc.baseArrayLayer = 0;
  depthTextureViewDesc.arrayLayerCount = 1;
  depthTextureViewDesc.baseMipLevel = 0; 
  depthTextureViewDesc.mipLevelCount = 1;
  depthTextureViewDesc.dimension = TextureViewDimension::_2D;
  depthTextureViewDesc.format = m_depthTextureFormat;
  m_depthTextureView = m_depthTexture.createView(depthTextureViewDesc);

  return m_depthTexture != nullptr and m_depthTextureView != nullptr;
}

void Application::terminateDepthBuffer() {
  m_depthTextureView.release();
  m_depthTexture.release();
}

bool Application::initRenderPipeline() {
  std::cout << "Creating render pipeline..." << std::endl;
  RenderPipelineDescriptor pipelineDesc; 

  // create vertex buffer
  // VertexAttribute is a WGPU struct that specifies the properties of an attribute,
  // not to be confused with VertexAttributes, which is a struct listing the attributes
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

  // shader module 
  m_shaderModule = ResourceManager::loadShaderModule
    (RESOURCE_DIR "/shader.wgsl", m_device);
  pipelineDesc.vertex.module = m_shaderModule;
  pipelineDesc.vertex.entryPoint = "vs_main";
  pipelineDesc.vertex.constantCount = 0;
  pipelineDesc.vertex.constants = nullptr;

  // fragment shader setup
  pipelineDesc.primitive.topology = PrimitiveTopology::TriangleList;  
  pipelineDesc.primitive.stripIndexFormat= IndexFormat::Undefined;  
  pipelineDesc.primitive.frontFace = FrontFace::CCW;
  pipelineDesc.primitive.cullMode = CullMode::None;

  FragmentState fragmentState;
  fragmentState.module = m_shaderModule;
  fragmentState.entryPoint = "fs_main";
  fragmentState.constantCount = 0;
  fragmentState.constants = nullptr;
  pipelineDesc.fragment = &fragmentState;

  // color target setup
  ColorTargetState colorTarget;
  {
    // blend state setup
    BlendState blendState;
    blendState.color.srcFactor = BlendFactor::SrcAlpha;
    blendState.color.dstFactor = BlendFactor::OneMinusSrcAlpha;
    blendState.color.operation = BlendOperation::Add;
    blendState.alpha.srcFactor = BlendFactor::Zero;
    blendState.alpha.dstFactor = BlendFactor::One;
    blendState.alpha.operation = BlendOperation::Add;
    
    colorTarget.format = m_swapChainFormat;
    colorTarget.blend = &blendState; 
    colorTarget.writeMask = ColorWriteMask::All;

  }

  fragmentState.targetCount = 1; 
  fragmentState.targets = &colorTarget;
  
  // depth stencil
  DepthStencilState m_depthStencilState = Default; 
  m_depthStencilState.depthCompare = CompareFunction::Less;  
  m_depthStencilState.depthWriteEnabled = true;
  m_depthStencilState.format = m_depthTextureFormat;
  m_depthStencilState.stencilReadMask = 0;
  m_depthStencilState.stencilWriteMask = 0;
  pipelineDesc.depthStencil = &m_depthStencilState;
  
  // multisampling
  pipelineDesc.multisample.count = 1;
  pipelineDesc.multisample.mask = ~0u;
  pipelineDesc.multisample.alphaToCoverageEnabled = false;
  
  // bind groups
  std::vector<BindGroupLayoutEntry> bindingLayoutEntries(4, Default);
  // for uniform buffer
  BindGroupLayoutEntry& uniformBindingLayout = bindingLayoutEntries[0];
  uniformBindingLayout.binding = 0;
  uniformBindingLayout.visibility = ShaderStage::Vertex | ShaderStage::Fragment;
  uniformBindingLayout.buffer.type = BufferBindingType::Uniform;
  uniformBindingLayout.buffer.minBindingSize = sizeof(MyUniforms);
  // for texture 
  BindGroupLayoutEntry& textureBindingLayout = bindingLayoutEntries[1];
  textureBindingLayout.binding = 1;
  textureBindingLayout.visibility = ShaderStage::Fragment; 
  textureBindingLayout.texture.sampleType = TextureSampleType::Float;  
  textureBindingLayout.texture.viewDimension = TextureViewDimension::_2D;  
  // for sampler
  BindGroupLayoutEntry& samplerBindingLayout = bindingLayoutEntries[2];
  samplerBindingLayout.binding = 2; 
  samplerBindingLayout.visibility = ShaderStage::Fragment; 
  samplerBindingLayout.sampler.type = SamplerBindingType::Filtering;
  // for lighting uniform
  BindGroupLayoutEntry& lightingBindingLayout = bindingLayoutEntries[3];
  lightingBindingLayout.binding = 3;
  lightingBindingLayout.visibility = ShaderStage::Fragment;
  lightingBindingLayout.buffer.type = BufferBindingType::Uniform;
  lightingBindingLayout.buffer.minBindingSize = sizeof(LightingUniforms);
  // bind group layout
  BindGroupLayoutDescriptor bindGroupLayoutDesc{}; 
  bindGroupLayoutDesc.entryCount = (uint32_t) bindingLayoutEntries.size();
  bindGroupLayoutDesc.entries = bindingLayoutEntries.data();
  m_bindGroupLayout = m_device.createBindGroupLayout(bindGroupLayoutDesc);
  

  // pipeline layout
  PipelineLayoutDescriptor layoutDesc{}; 
  layoutDesc.bindGroupLayoutCount = 1; 
  layoutDesc.bindGroupLayouts = (WGPUBindGroupLayout*)&m_bindGroupLayout;
  PipelineLayout layout = m_device.createPipelineLayout(layoutDesc);
  pipelineDesc.layout = layout;
  
  // create the pipeline
  m_pipeline = m_device.createRenderPipeline(pipelineDesc);
  std::cout << "Render pipeline: " << m_pipeline << std::endl;

  return m_pipeline != nullptr;
}

void Application::terminateRenderPipeline() {
  m_bindGroupLayout.release();
  m_pipeline.release(); 
}

bool Application::initTexture() {
  std::cout << "initialzing texture..." << std::endl;
  // initialize sampler
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
  m_sampler = m_device.createSampler(samplerDesc);

  std::cout << "loading texture..." << std::endl;
  m_texture = ResourceManager::loadTexture(RESOURCE_DIR "/fourareen2K_albedo.jpg", m_device, &m_textureView);
  if (!m_texture) {
    std::cerr << "Could not load texture!" << std::endl;
    return false;
  }

  std::cout << "Texture: " << m_texture << std::endl;
  std::cout << "Texture View: " << m_textureView << std::endl;

  return m_textureView != nullptr;
}

void Application::terminateTexture() {
  m_textureView.release();
  m_texture.release(); 
  m_sampler.release(); 
}

bool Application::initGeometry() {
  // load vertex data into this vector
  std::vector<VertexAttributes> vertexData;
  bool success = ResourceManager::loadGeometryFromObj(RESOURCE_DIR "/fourareen.obj", vertexData);
  if (!success) {
    std::cerr << "Could not load geometry!" << std::endl;
    return false;
  }

  BufferDescriptor bufferDesc{};
  // create vertex buffer
  bufferDesc.size = vertexData.size() * sizeof(VertexAttributes);
  bufferDesc.usage = BufferUsage::CopyDst | BufferUsage::Vertex;  
  bufferDesc.mappedAtCreation = false; 
  m_vertexBuffer = m_device.createBuffer(bufferDesc);
  if (m_vertexBuffer == nullptr) {
    std::cerr << "Could not create vertex buffer!" << std::endl;
    return false;
  }
  m_queue.writeBuffer(m_vertexBuffer, 0, vertexData.data(), bufferDesc.size);
  
  // get number of vertices
  m_indexCount = static_cast<int>(vertexData.size());

  return true;
}

void Application::terminateGeometry() {
  m_vertexBuffer.release();
}


bool Application::initUniforms() {
  // initialize the buffer
  BufferDescriptor bufferDesc{};
  bufferDesc.size = sizeof(MyUniforms);
  bufferDesc.usage = BufferUsage::CopyDst | BufferUsage::Uniform; 
  bufferDesc.mappedAtCreation = false;
  m_uniformBuffer = m_device.createBuffer(bufferDesc);
  if (m_uniformBuffer == nullptr) {
    std::cerr << "Could not create uniform buffer!" << std::endl;
    return false;
  }


  // initial values
  m_uniforms.time = 1.0f; 
  m_uniforms.color = {0.4f, 0.0f, 1.0f, 1.0f};
  m_uniforms.modelMatrix = mat4x4(1.0);
  m_uniforms.viewMatrix = mat4x4(1.0);
  updateViewMatrix();
  m_uniforms.projectionMatrix = glm::perspective(45 * PI / 180, 640.0f / 480.0f, 0.01f, 100.0f);
  m_queue.writeBuffer(m_uniformBuffer, 0, &m_uniforms, sizeof(MyUniforms));


  return initLightingUniforms();
}



void Application::terminateUniforms() {
  m_uniformBuffer.release();
}

bool Application::initLightingUniforms() {
  // lighting buffer
  BufferDescriptor bufferDesc{};
  bufferDesc.size = sizeof(LightingUniforms); 
  bufferDesc.usage = BufferUsage::CopyDst | BufferUsage::Uniform; 
  bufferDesc.mappedAtCreation = false;
  m_lightingBuffer = m_device.createBuffer(bufferDesc);

  m_lighting.directions = {vec4(0.5, -0.9, 0.1, 0.0), vec4(0.2, 0.4, 0.3, 0.0)};
  m_lighting.colors = {vec4(1.0, 0.9, 0.6, 0.0), vec4(0.6, 0.9, 1.0, 0.0)};
  updateLightingUniforms();
  if (m_lightingBuffer == nullptr) {
    std::cerr << "Could not create lighting buffer!" << std::endl; 
    return false;
  }
  return true;
}

void Application::terminateLightingUniforms() {
  m_lightingBuffer.destroy();
  m_lightingBuffer.release(); 
}

void Application::updateLightingUniforms() {
  m_queue.writeBuffer(m_lightingBuffer, 0, &m_lighting, sizeof(LightingUniforms));
}

bool Application::initBindGroup() {
  std::vector<BindGroupEntry> bindings(4);
  BindGroupEntry binding{};  
  
  // uniform buffer binding
  bindings[0].binding = 0; 
  bindings[0].buffer = m_uniformBuffer;
  bindings[0].offset = 0;
  bindings[0].size = sizeof(MyUniforms);
  
  // texture binding
  bindings[1].binding = 1; 
  bindings[1].textureView= m_textureView;
  
  // sampler binding
  bindings[2].binding = 2; 
  bindings[2].sampler = m_sampler;

  // lighting binding 
  bindings[3].binding = 3; 
  bindings[3].buffer = m_lightingBuffer; 
  bindings[3].offset = 0; 
  bindings[3].size = sizeof(LightingUniforms);
  
  // create the bind group
  BindGroupDescriptor bindGroupDesc{};   
  bindGroupDesc.layout = m_bindGroupLayout;
  bindGroupDesc.entryCount = (uint32_t)bindings.size();
  bindGroupDesc.entries = bindings.data();
  m_bindGroup = m_device.createBindGroup(bindGroupDesc) ;

  return m_bindGroup != nullptr;
}

void Application::terminateBindGroup() {
  m_bindGroup.release();
}

bool Application::initGui() {
  // setup imGui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext(); 
  ImGui::GetIO();

  // setup Platform/renderer backends 
  ImGui_ImplGlfw_InitForOther(m_window, true);
  ImGui_ImplWGPU_Init(m_device, 3, m_swapChainFormat, m_depthTextureFormat);
  return true;
}

void Application::terminateGui() {
  ImGui_ImplGlfw_Shutdown();
  ImGui_ImplWGPU_Shutdown();
}

void Application::updateGui(RenderPassEncoder renderPass) {
  // start new frame 
  ImGui_ImplWGPU_NewFrame();
  ImGui_ImplGlfw_NewFrame(); 
  ImGui::NewFrame(); 

  // build the UI 
  // static variables are remembered across frames
  static float f = 0.0f; 
  static int counter = 0; 
  static bool show_demo_window = true; 
  static bool show_another_window = false; 
  static ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

  ImGui::Begin("Hello World!"); 
  ImGui::Text("This is some useful text"); 
  ImGui::Checkbox("Demo Window", &show_demo_window); 
  ImGui::Checkbox("Another Window", &show_another_window);

  ImGui::SliderFloat("float", &f, 0.0f, 1.0f); 
  ImGui::ColorEdit3("clear color", (float*)&clear_color);

  if (ImGui::Button("Button")) {
    ++counter; 
  }
  ImGui::SameLine(); 
  ImGui::Text("counter = %d", counter); 

  ImGuiIO& io = ImGui::GetIO(); 
  ImGui::Text("Application Average %.3f ms/frame (%.1f FPS)", 1000.0f/io.Framerate, io.Framerate);
  ImGui::End();

  ImGui::Begin("Lighting");
  bool changed = false;
  changed = ImGui::ColorEdit3("Color #0", glm::value_ptr(m_lighting.colors[0])) || changed;
  changed = ImGui::DragDirection("Direction #0", m_lighting.directions[0]) || changed;
  changed = ImGui::ColorEdit3("Color #1", glm::value_ptr(m_lighting.colors[1])) || changed;
  changed = ImGui::DragDirection("Direction #1", m_lighting.directions[1]) || changed;
  changed = ImGui::SliderFloat("Hardness", &m_lighting.hardness, 1.0f, 100.0f) || changed;
  changed = ImGui::SliderFloat("K Diffuse", &m_lighting.diffStr, 0.0f, 1.0f) || changed;
  changed = ImGui::SliderFloat("K Specular", &m_lighting.specStr, 0.0f, 1.0f) || changed;
  if (changed) {
    m_lightingUniformsChanged = true;
  }
  ImGui::End();


  // draw the UI
  ImGui::EndFrame(); 
  // Convert the UI into low level rendering commands 
  ImGui::Render(); 
  // Execute the low level drawing commands on the WGPU backend 
  ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), renderPass);
}

