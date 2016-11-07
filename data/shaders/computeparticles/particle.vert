#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (location = 0) in vec2 inPos;
layout (location = 1) in vec2 inVel;

layout (location = 0) out vec2 outColor;

out gl_PerVertex
{
	vec4 gl_Position;
	float gl_PointSize;
};

void main ()
{
  gl_PointSize = 8.0;
  outColor = inVel;
  gl_Position = vec4(inPos.xy, 1.0, 1.0);
}
