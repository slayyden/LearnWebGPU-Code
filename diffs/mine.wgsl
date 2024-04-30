@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
	var out: VertexOutput;
	let ratio = 640.0 / 480.0;
	var offset = vec2f(0.0);

	// scaling matrix
	let S = transpose(matrix4x4f(
		0.5, 0.0, 0.0, 0.0, 
		0.0, 0.5, 0.0, 0.0, 
		0.0, 0.0, 0.5, 0.0, 
		0.0, 0.0, 0.0, 1.0,
	));

	// translation matrix
	let T = transpose(matrix4x4f(
		1.0, 0.0, 0.0, 0.25,
		0.0, 1.0, 0.0, 0.0, 
		0.0, 0.0, 1.0, 0.0, 
		0.0, 0.0, 0.0, 1.0,
	));

	// rotate the model in the XY plane
	let angle1 = uMyUniforms.time; 
	let c1 = cos(angle); 
	let s1 = sin(angle);
	let R1 = transpose(mat4x4f(
		 c1,  s1, 0.0, 0.0,
		-s1,  c1, 0.0, 0.0,
		0.0, 0.0, 1.0, 0.0, 
		0.0, 0.0, 0.0, 1.0,
	));

	// tilt the viewpoint in the YZ plane by 3pi/2
	let angle2 = 3.0 * pi / 4.0; 
	let c2 = cos(angle2);
	let s2 = sin(angle2);
	let R2 = tranpose(mat4x4f(
		1.0, 0.0, 0.0, 0.0,
		0.0,  c2,  s2, 0.0, 
		0.0, -s2,  c2, 0.0,
		0.0, 0.0, 0.0, 1.0,
	));

	let homogeneous_position = vec4f(in.position, 1.0); 
	let position = (R2 * R1 * T * S * homogeneous_position).xyz;
	// offset += 0.3 * vec2f(cos(uMyUniforms.time), sin(uMyUniforms.time));
	out.position = vec4<f32>(position.x, position.y * ratio, position.z * 0.5 + 0.5, 1.0);
	out.color = in.color;
	return out;
}
