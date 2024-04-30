#pragma once 
#include <glm/glm.hpp> // all types inspired from GLSL
#include <webgpu/webgpu.hpp>
#include <array>

struct GLFWwindow; 

class Application {

  public: 
    // called once at the beginning. returns false if initialization fails
    bool onInit();

    // called once per frame 
    void onFrame(); 

    // called at the very end 
    void onFinish();


    bool isRunning();

    // handle user input
    void onResize();
    void onMouseMove(double xpos, double ypos);
    void onMouseButton(int button, int action, int mods);
    void onScroll(double xoffset, double yoffset);

    // lighting
    struct LightingUniforms {
      std::array<glm::vec4, 2>  directions;
      std::array<glm::vec4, 2>  colors;
      float hardness; 
      float diffStr; 
      float specStr;
      float _pad[1]; 
    };
    static_assert(sizeof(LightingUniforms) % 16 == 0);

  private: 
    // store shader uniforms
    struct MyUniforms {
      glm::mat4x4 projectionMatrix;
      glm::mat4x4 viewMatrix;
      glm::mat4x4 modelMatrix;
      glm::vec4 color; // or float color[4]
      glm::vec3 cameraWorldPosition; 
      float time;
      float _pad[3];
    };



    struct CameraState {
      // angles.x is rotation around global vertical axis
      // angles.y is rotation around local horizontal axis (looking up and down)
      glm::vec2 angles = {0.8f, 0.5f};
      // position of the camera along its local forward axis
      float zoom = -1.2f;
    };

    struct DragState {
      // whether the mouse is currently dragging the view 
      bool active = false; 
      // position of mouse at start of dragging 
      glm::vec2 startMouse; 
      // camera state at the start of dragging 
      CameraState startCameraState; 

      // constant settings
      float sensitivity = 0.01f;
      float scrollSensitivity = 0.1f;
    };


    // everything that is initialized on onInit() and needed in onFrame()
    // these must be initialized to nullptr since they have no default constructor

    // for initWindowAndDevice
    wgpu::Instance m_instance = nullptr; 
    wgpu::Surface m_surface = nullptr; 
    GLFWwindow* m_window = nullptr; 
    wgpu::Device m_device = nullptr;
    wgpu::Queue m_queue = nullptr;

    // for initSwapChain()
    wgpu::SwapChain m_swapChain = nullptr;
    wgpu::TextureFormat m_swapChainFormat = wgpu::TextureFormat::BGRA8Unorm;

    // for initDepthBuffer()
    wgpu::TextureFormat m_depthTextureFormat = wgpu::TextureFormat::Depth24Plus;
    wgpu::Texture m_depthTexture = nullptr;
    wgpu::TextureView m_depthTextureView = nullptr;

    // for initRenderPipeline()
    wgpu::ShaderModule m_shaderModule = nullptr;
    wgpu::RenderPipeline m_pipeline = nullptr;
    wgpu::BindGroupLayout m_bindGroupLayout  = nullptr;

    // for initTexture()
    wgpu::Sampler m_sampler = nullptr;
    wgpu::TextureView m_textureView = nullptr;
    wgpu::Texture m_texture = nullptr; 

    // for initGeometry()
    wgpu::Buffer m_vertexBuffer = nullptr;
    int m_indexCount = 0;
   
    // for initUniforms()
    MyUniforms m_uniforms; 
    CameraState m_cameraState;
    wgpu::Buffer m_uniformBuffer = nullptr;
    LightingUniforms m_lighting; 
    wgpu::Buffer m_lightingBuffer = nullptr;

    // for initBindGroup()
    wgpu::BindGroup m_bindGroup = nullptr;
    
    // initialization helper functions
    bool initWindowAndDevice(); 
    bool initSwapChain(); 
    bool initDepthBuffer(); 
    bool initRenderPipeline(); 
    bool initTexture(); 
    bool initGeometry(); 
    bool initUniforms(); 
    bool initBindGroup();
    bool initGui();
    
    // onFinish() helper functions
    void terminateBindGroup();
    void terminateUniforms();
    void terminateGeometry();
    void terminateTexture();
    void terminateRenderPipeline();
    void terminateDepthBuffer();
    void terminateSwapChain();
    void terminateWindowAndDevice();
    void terminateGui();

    // Keep the error callback alive
    std::unique_ptr<wgpu::ErrorCallback> m_errorCallbackHandle;
    // bool m_terminate;

    // handle user input
    void updateProjectionMatrix();
    void updateViewMatrix();
    DragState m_drag;

    // gui 
    void updateGui(wgpu::RenderPassEncoder renderPass);

    // lighting
    bool initLightingUniforms(); 
    void terminateLightingUniforms(); 
    void updateLightingUniforms();
    bool m_lightingUniformsChanged = false;
};
