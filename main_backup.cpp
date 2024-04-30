/**
 * This file is part of the "Learn WebGPU for C++" book.
 *   https://github.com/eliemichel/LearnWebGPU
 *
 * MIT License
 * Copyright (c) 2022 Elie Michel
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

#define WEBGPU_CPP_IMPLEMENTATION
#include <webgpu/webgpu.hpp>

#include <GLFW/glfw3.h>
#include <glfw3webgpu.h>

#include <cassert>
#include <iostream>

#define UNUSED(x) (void)x;

using namespace wgpu;

int main(int, char **) {
  // create instance
  Instance instance = createInstance(InstanceDescriptor{});
  if (!instance) {
    std::cerr << "Could not initialize WebGPU!" << std::endl;
    return 1;
  }

  // initialize glfw
  if (!glfwInit()) {
    std::cerr << "Could not initialize GLFW!" << std::endl;
    return 1;
  }

  // create window
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
  GLFWwindow *window = glfwCreateWindow(640, 480, "Learn WebGPU", NULL, NULL);
  if (!window) {
    std::cerr << "Could not open window!" << std::endl;
    return 1;
  }

  // request adapter
  std::cout << "Requesting adapter..." << std::endl;
  Surface surface = glfwGetWGPUSurface(instance, window);
  RequestAdapterOptions adapterOpts{};
  adapterOpts.compatibleSurface = surface;
  Adapter adapter = instance.requestAdapter(adapterOpts);
  std::cout << "Got adapter: " << adapter << std::endl;

  // create supported limits
  SupportedLimits supportedLimits;
  adapter.getLimits(&supportedLimits);

  // Don't forget to = Default
  RequiredLimits requiredLimits = Default;
  requiredLimits.limits.maxVertexAttributes = 3;
  // We should also tell that we use 2 vertex buffers
  requiredLimits.limits.maxVertexBuffers = 3;
  // Maximum size of a buffer is 6 vertices of 2 float each
  requiredLimits.limits.maxBufferSize = 8 * 3 * sizeof(float);
  // Maximum stride between 2 consecutive vertices in the vertex buffer
  requiredLimits.limits.maxVertexBufferArrayStride = 3 * sizeof(float);
  // This must be set even if we do not use storage buffers for now
  requiredLimits.limits.minStorageBufferOffsetAlignment =
      supportedLimits.limits.minStorageBufferOffsetAlignment;
  // This must be set even if we do not use uniform buffers for now
  requiredLimits.limits.minUniformBufferOffsetAlignment =
      supportedLimits.limits.minUniformBufferOffsetAlignment;
  requiredLimits.limits.maxInterStageShaderComponents = 3;

  // request device
  std::cout << "Requesting device..." << std::endl;
  DeviceDescriptor deviceDesc{};
  deviceDesc.label = "My Device";
  deviceDesc.requiredFeaturesCount = 0;
  deviceDesc.requiredLimits = nullptr;
  deviceDesc.defaultQueue.label = "The default queue";
  deviceDesc.requiredLimits = &requiredLimits;
  Device device = adapter.requestDevice(deviceDesc);
  std::cout << "Got device: " << device << std::endl;

  // print out supported limits
  adapter.getLimits(&supportedLimits);
  std::cout << "adapter.maxVertexAttributes: "
            << supportedLimits.limits.maxVertexAttributes << std::endl;
  device.getLimits(&supportedLimits);
  std::cout << "device.maxVertexAttributes: "
            << supportedLimits.limits.maxVertexAttributes << std::endl;

  // create queue
  Queue queue = device.getQueue();

  // bro idk
  auto onDeviceError = [](WGPUErrorType type, char const *message,
                          void * /* pUserData */) {
    std::cout << "Uncaptured device error: type " << type;
    if (message)
      std::cout << " (" << message << ")";
    std::cout << std::endl;
  };
  wgpuDeviceSetUncapturedErrorCallback(device, onDeviceError,
                                       nullptr /* pUserData */);

  // create swap chain
  std::cout << "Creating swapchain device..." << std::endl;
#ifdef WEBGPU_BACKEND_WGPU
  TextureFormat swapChainFormat = surface.getPreferredFormat(adapter);
#else
  TextureFormat swapChainFormat = TextureFormat::BGRA8Unorm;
#endif
  SwapChainDescriptor swapChainDesc = {};
  swapChainDesc.width = 640;
  swapChainDesc.height = 480;
  swapChainDesc.usage = TextureUsage::RenderAttachment;
  swapChainDesc.format = swapChainFormat;
  swapChainDesc.presentMode = PresentMode::Fifo;
  SwapChain swapChain = device.createSwapChain(surface, swapChainDesc);
  std::cout << "Swapchain: " << swapChain << std::endl;

  // define our shader

  const char *shaderSource = R"(
    struct VertexInput {
        @location(0) position: vec2f,
        @location(1) color: vec3f,
    };

    /**
    * A structure with fields labeled with builtins and locations can also be used
    * as *output* of the vertex shader, which is also the input of the fragment
    * shader.
    */
    struct VertexOutput {
        @builtin(position) position: vec4f,
        // The location here does not refer to a vertex attribute, it just means
        // that this field must be handled by the rasterizer.
        // (It can also refer to another field of another struct that would be used
        // as input to the fragment shader.)
        @location(0) color: vec3f,
    };

    @vertex
    fn vs_main(in: VertexInput) -> VertexOutput {
        var out: VertexOutput;
        out.position = vec4f(in.position, 0.0, 1.0);
        out.color = in.color; // forward to the fragment shader
        return out;
    }

    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4f {
        return vec4f(in.color, 1.0);
    }
  )";

  ShaderModuleDescriptor shaderDesc;
#ifdef WEBGPU_BACKEND_WGPU
  shaderDesc.hintCount = 0;
  shaderDesc.hints = nullptr;
#endif
  ShaderModuleWGSLDescriptor shaderCodeDesc;
  // Set the chained struct's header
  shaderCodeDesc.chain.next = nullptr;
  shaderCodeDesc.chain.sType = SType::ShaderModuleWGSLDescriptor;
  // Connect the chain
  shaderDesc.nextInChain = &shaderCodeDesc.chain;
  shaderCodeDesc.code = shaderSource;
  ShaderModule shaderModule = device.createShaderModule(shaderDesc);

  // create render pipeline
  RenderPipelineDescriptor pipelineDesc;
  // pipelineDesc.vertex is a struct configuring the vertex fetch and vertex
  // shader. We don't need a position buffer for a triangle since we're hard
  // coding them

  // Vertex fetch
  // We now have 2 attributes
  std::vector<VertexBufferLayout> vertexBufferLayouts(2);

  // Position attribute remains untouched
  VertexAttribute positionAttrib;
  positionAttrib.shaderLocation = 0;
  positionAttrib.format = VertexFormat::Float32x2; // size of position
  positionAttrib.offset = 0;

  vertexBufferLayouts[0].attributeCount = 1;
  vertexBufferLayouts[0].attributes = &positionAttrib;
  vertexBufferLayouts[0].arrayStride =
      2 * sizeof(float); // stride = size of position
  vertexBufferLayouts[0].stepMode = VertexStepMode::Vertex;

  // Color attribute
  VertexAttribute colorAttrib;
  colorAttrib.shaderLocation = 1;
  colorAttrib.format = VertexFormat::Float32x3; // size of color
  colorAttrib.offset = 0;

  vertexBufferLayouts[1].attributeCount = 1;
  vertexBufferLayouts[1].attributes = &colorAttrib;
  vertexBufferLayouts[1].arrayStride =
      3 * sizeof(float); // stride = size of color
  vertexBufferLayouts[1].stepMode = VertexStepMode::Vertex;

  pipelineDesc.vertex.bufferCount =
      static_cast<uint32_t>(vertexBufferLayouts.size());
  pipelineDesc.vertex.buffers = vertexBufferLayouts.data();

  // initialize other pipeline vertex values
  pipelineDesc.vertex.module = shaderModule;
  pipelineDesc.vertex.entryPoint = "vs_main";
  pipelineDesc.vertex.constantCount = 0;
  pipelineDesc.vertex.constants = nullptr;

  // Each sequence of 3 vertices is considered as a triangle
  pipelineDesc.primitive.topology = PrimitiveTopology::TriangleList;

  // We'll see later how to specify the order in which vertices should be
  // connected. When not specified, vertices are considered sequentially.
  pipelineDesc.primitive.stripIndexFormat = IndexFormat::Undefined;

  // The face orientation is defined by assuming that when looking
  // from the front of the face, its corner vertices are enumerated
  // in the counter-clockwise (CCW) order.
  pipelineDesc.primitive.frontFace = FrontFace::CCW;

  // But the face orientation does not matter much because we do not
  // cull (i.e. "hide") the faces pointing away from us (which is often
  // used for optimization).
  pipelineDesc.primitive.cullMode = CullMode::None;

  // define fragment shader
  FragmentState fragmentState;
  fragmentState.module = shaderModule;
  fragmentState.entryPoint = "fs_main";
  fragmentState.constantCount = 0;
  fragmentState.constants = nullptr;
  // [...] We'll configure the blend stage here
  pipelineDesc.fragment = &fragmentState;

  pipelineDesc.depthStencil = nullptr;

  // define blend state
  BlendState blendState;
  blendState.color.srcFactor = BlendFactor::SrcAlpha;
  blendState.color.dstFactor = BlendFactor::OneMinusSrcAlpha;
  blendState.color.operation = BlendOperation::Add;

  // leave target alpha untouched
  blendState.alpha.srcFactor = BlendFactor::Zero;
  blendState.alpha.dstFactor = BlendFactor::One;
  blendState.alpha.operation = BlendOperation::Add;

  ColorTargetState colorTarget;
  colorTarget.format = swapChainFormat;
  colorTarget.blend = &blendState;
  // We could write to only some of the color channels.
  colorTarget.writeMask = ColorWriteMask::All;

  // We have only one target because our render pass has only one output color
  // attachment.
  fragmentState.targetCount = 1;
  fragmentState.targets = &colorTarget;

  // Samples per pixel
  pipelineDesc.multisample.count = 1;
  // Default value for the mask, meaning "all bits on"
  pipelineDesc.multisample.mask = ~0u;
  // Default value as well (irrelevant for count = 1 anyways)
  pipelineDesc.multisample.alphaToCoverageEnabled = false;

  // we don't use any resources (buffers/textures)
  pipelineDesc.layout = nullptr;

  RenderPipeline pipeline = device.createRenderPipeline(pipelineDesc);

  // Vertex buffer
  std::vector<float> pointData = {
      -0.5, -0.5,             // A
      +0.5, -0.5, +0.5, +0.5, // C
      -0.5, +0.5,
  };

  // This is a list of indices referencing positions in the pointData
  std::vector<uint16_t> indexData = {
      0, 1, 2, // Triangle #0
      0, 2, 3  // Triangle #1
  };

  // r0,  g0,  b0, r1,  g1,  b1, ...
  std::vector<float> colorData = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                                  0.0, 0.0, 1.0, 1.0, 1.0, 0.0};

  int indexCount = static_cast<int>(indexData.size());
  int pointCount = static_cast<int>(pointData.size() / 2);
  assert(pointCount == static_cast<int>(colorData.size() / 3));

  // Create vertex buffers
  BufferDescriptor bufferDesc;
  bufferDesc.usage = BufferUsage::CopyDst | BufferUsage::Vertex;
  bufferDesc.mappedAtCreation = false;

  // position buffer
  bufferDesc.size = pointData.size() * sizeof(float);
  Buffer positionBuffer = device.createBuffer(bufferDesc);
  queue.writeBuffer(positionBuffer, 0, pointData.data(), bufferDesc.size);

  // Create index buffer
  // (we reuse the bufferDesc initialized for the vertexBuffer)
  bufferDesc.size = indexData.size() * sizeof(uint16_t);
  bufferDesc.usage = BufferUsage::CopyDst | BufferUsage::Index;
  Buffer indexBuffer = device.createBuffer(bufferDesc);
  queue.writeBuffer(indexBuffer, 0, indexData.data(), bufferDesc.size);
  // color buffer
  bufferDesc.size = colorData.size() * sizeof(float);
  Buffer colorBuffer = device.createBuffer(bufferDesc);
  queue.writeBuffer(colorBuffer, 0, colorData.data(), bufferDesc.size);

  // main loop
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    TextureView nextTexture = swapChain.getCurrentTextureView();
    if (!nextTexture) {
      std::cerr << "Cannot acquire next swap chain texture" << std::endl;
      return 1;
    }

    CommandEncoderDescriptor commandEncoderDesc;
    commandEncoderDesc.label = "Command Encoder";
    CommandEncoder encoder = device.createCommandEncoder(commandEncoderDesc);

    RenderPassDescriptor renderPassDesc{};

    WGPURenderPassColorAttachment renderPassColorAttachment = {};
    renderPassColorAttachment.view = nextTexture;
    renderPassColorAttachment.resolveTarget = nullptr;
    renderPassColorAttachment.loadOp = LoadOp::Clear;
    renderPassColorAttachment.storeOp = StoreOp::Store;
    renderPassColorAttachment.clearValue = Color{0.05, 0.05, 0.05, 1.0};
    renderPassDesc.colorAttachmentCount = 1;
    renderPassDesc.colorAttachments = &renderPassColorAttachment;

    renderPassDesc.depthStencilAttachment = nullptr;
    renderPassDesc.timestampWriteCount = 0;
    renderPassDesc.timestampWrites = nullptr;

    RenderPassEncoder renderPass = encoder.beginRenderPass(renderPassDesc);

    // Select which render pipeline to use
    renderPass.setPipeline(pipeline);

    // Set vertex buffers while encoding the render pass
    renderPass.setVertexBuffer(0, positionBuffer, 0,
                               pointData.size() * sizeof(float));

    renderPass.setVertexBuffer(1, colorBuffer, 0,
                               colorData.size() * sizeof(float));
    renderPass.setIndexBuffer(indexBuffer, IndexFormat::Uint16, 0,
                              indexData.size() * sizeof(uint16_t));
    /// Replace `draw()` with `drawIndexed()` and `vertexCount` with
    /// `indexCount`
    // The extra argument is an offset within the index buffer.
    renderPass.drawIndexed(indexCount, 1, 0, 0, 0);
    renderPass.end();
    nextTexture.release();

    CommandBufferDescriptor cmdBufferDescriptor{};
    cmdBufferDescriptor.label = "Command buffer";
    CommandBuffer command = encoder.finish(cmdBufferDescriptor);
    queue.submit(command);

    swapChain.present();
  }

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
