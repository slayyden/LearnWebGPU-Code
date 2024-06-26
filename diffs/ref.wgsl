@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
	var out: VertexOutput;
	let ratio = 640.0 / 480.0;
	var offset = vec2f(0.0);

	// Scale the object
	let S = transpose(mat4x4f(
		0.3,  0.0, 0.0, 0.0,
		0.0,  0.3, 0.0, 0.0,
		0.0,  0.0, 0.3, 0.0,
		0.0,  0.0, 0.0, 1.0,
	));

	// Translate the object
	let T = transpose(mat4x4f(
		1.0,  0.0, 0.0, 0.5,
		0.0,  1.0, 0.0, 0.0,
		0.0,  0.0, 1.0, 0.0,
		0.0,  0.0, 0.0, 1.0,
	));

	// Rotate the model in the XY plane
	let angle1 = uMyUniforms.time;
	let c1 = cos(angle1);
	let s1 = sin(angle1);
	let R1 = transpose(mat4x4f(
		 c1,  s1, 0.0, 0.0,
		-s1,  c1, 0.0, 0.0,
		0.0, 0.0, 1.0, 0.0,
		0.0,  0.0, 0.0, 1.0,
	));

	// Tilt the view point in the YZ plane
	// by three 8th of turn (1 turn = 2 pi)
	let angle2 = 3.0 * pi / 4.0;
	let c2 = cos(angle2);
	let s2 = sin(angle2);
	let R2 = transpose(mat4x4f(
		1.0, 0.0, 0.0, 0.0,
		0.0,  c2,  s2, 0.0,
		0.0, -s2,  c2, 0.0,
		0.0,  0.0, 0.0, 1.0,
	));

	// Compose and apply rotations
	// (S then T then R1 then R2, remember this reads backwards)
	let homogeneous_position = vec4f(in.position, 1.0);
	let position = (R2 * R1 * T * S * homogeneous_position).xyz;

	out.position = vec4<f32>(position.x, position.y * ratio, position.z * 0.5 + 0.5, 1.0);

	out.color = in.color;
	return out;
}
