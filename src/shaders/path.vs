#version 330
uniform mat4 mvp;
in vec3 position;
in vec3 color;
out vec3 frag_color;

void main() {
    gl_Position = mvp * vec4(position, 1.0);
    if (isnan(position.r))
        frag_color = vec3(0.0);
    else
        frag_color = color;
}