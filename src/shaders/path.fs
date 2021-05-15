#version 330
in vec3 frag_color;
out vec4 out_color;

void main() {
    if (frag_color == vec3(0.0))
        discard;
    out_color = vec4(frag_color, 1.0);
}