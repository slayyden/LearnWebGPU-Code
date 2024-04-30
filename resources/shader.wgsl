struct VertexInput {
	@location(0) position: vec3f,
	@location(1) normal: vec3f,
	@location(2) color: vec3f,
	@location(3) uv: vec2f,
};

struct VertexOutput {
	@builtin(position) position: vec4f,
	@location(0) color: vec3f,
	@location(1) normal: vec3f,
	@location(2) uv: vec2f,
	@location(3) viewDirection: vec3f,
};

// A structure holding the value of our uniforms
struct MyUniforms {
	projectionMatrix: mat4x4f, 
	viewMatrix: mat4x4f, 
	modelMatrix: mat4x4f,
	color: vec4f,
	cameraWorldPosition: vec3f,
	time: f32,
};

// struct holding lighting data 
struct LightingUniforms {
	directions: array<vec4f, 2>,
	colors: array<vec4f, 2>,
	hardness: f32, 
	diffStr: f32, 
	specStr: f32,
};

// Instead of the simple uTime variable, our uniform variable is a struct
@group(0) @binding(0) var<uniform> uMyUniforms: MyUniforms;
@group(0) @binding(1) var baseColorTexture: texture_2d<f32>; 
@group(0) @binding(2) var textureSampler: sampler; 
@group(0) @binding(3) var<uniform> uLighting: LightingUniforms;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
	var out: VertexOutput;
	let worldPosition = uMyUniforms.modelMatrix * vec4f(in.position, 1.0);	
    out.position = uMyUniforms.projectionMatrix * uMyUniforms.viewMatrix * worldPosition; 
	out.normal = (uMyUniforms.modelMatrix * vec4f(in.normal, 1.0)).xyz;
	out.color = in.color;
	out.uv = in.uv;
	let cameraWorldPosition = uMyUniforms.cameraWorldPosition;
	out.viewDirection = cameraWorldPosition - worldPosition.xyz; 

	return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
	// let texelCoords = vec2i(in.uv * vec2f(textureDimensions(gradientTexture)));

	// interpolate and normalize normal and view direction from vtx shader
	let normal = normalize(in.normal);
	let V = normalize(in.viewDirection);

	let baseColor = textureSample(baseColorTexture, textureSampler, in.uv).rgb;
	let hardness = uLighting.hardness;
	let diffStr = uLighting.diffStr; 
	let specStr = uLighting.specStr;

	var shading = vec3f(0.0);
	for (var i : i32 = 0; i < 2; i++) {
		let direction = normalize(uLighting.directions[i].xyz);
		let color = uLighting.colors[i].rgb; 
		let diffuse = max(0.0, dot(direction, normal)) * color;

		let L = direction; 
		let N = normal; 
		let R = reflect(-L, N); // reflect the vector -L over N
		let RoV = max(0.0, dot(R, V));
		let specular = pow(RoV, hardness);
		shading += baseColor * diffuse * diffStr + specular * specStr; 
	}


	let color = baseColor * shading;
	// Gamma-correction
	//let corrected_color = pow(color, vec3f(2.2));
	//return vec4f(corrected_color, uMyUniforms.color.a);
	return vec4f(color, uMyUniforms.color.a);
}