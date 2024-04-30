
	float ratio = 640.0f / 480.0f;
	float focalLength = 2.0;
	float near = 0.01f;
	float far = 100.0f;
	float divider = 1 / (focalLength * (far - near));	
   	vec3 focalPoint(0.0, 0.0, -2.0); 
    float angle1 = 2.0f; // arbitrary time
    
    S = glm::scale(mat4x4(1.0), vec3(0.3f));
	T1 = glm::translate(mat4x4(1.0), vec3(0.5, 0.0, 0.0));
	R1 = glm::rotate(mat4x4(1.0), angle1, vec3(0.0, 0.0, 1.0));
	uniforms.modelMatrix = R1 * T1 * S;

	R2 = glm::rotate(mat4x4(1.0), -angle2, vec3(1.0, 0.0, 0.0));
	T2 = glm::translate(mat4x4(1.0), -focalPoint);
	uniforms.viewMatrix = T2 * R2;

    float fov = 2 * glm::atan(1 / focalLength);
	uniforms.projectionMatrix = glm::perspective(fov, ratio, near, far);