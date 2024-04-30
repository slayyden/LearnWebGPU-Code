constexpr float PI = 3.14159265358979323846f;

// model matrix
float angle1 = 2.0f; // arbitrary time
mat4x4 S = glm::scale(mat4x4(1.0), vec3(0.3f));
mat4x4 T1 = glm::translate(mat4x4(1.0), vec3(0.5, 0, 0));
mat4x4 R1 = glm::rotate(mat4x4(1.0), angle1, vec3(0.0, 0.0, 1.0));
uniforms.modelMatrix = R1 * T1 * S;

// view matrix
vec3 focalPoint(0.0, 0.0, -2.0);
float angle2 = 3.0 * PI / 4.0;
mat4x4 R2 = glm::rotate(mat4x4(1.0), -angle2, vec3(1.0, 0.0, 0.0)); 
mat4x4 T2 = glm:: translate(mat4x4(1.0), -focalPoint);
uniforms.viewMatrix = T2 * R2;

// projection matrix
float ratio = 640.0f / 480.0f;
float focalLength = 2.0;
float fov = 2 * glm::atan(1 / focalLength);
float near = 0.01f;
float far = 100.0f;
uniforms.projectionMatrix  = glm::perspective(fov, ratio, near, far);