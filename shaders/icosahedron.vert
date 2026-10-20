#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragWorldPos;

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 viewProj;
    vec4 cameraPos;
    vec4 ambient;
    vec4 dirDirection;
    vec4 dirColor;
    uint pointLightCount;
    uint spotLightCount;
} scene;

layout(set = 1, binding = 0) uniform ModelUBO {
    mat4 model;
    vec4 albedo;
    vec4 specular;
} model;

void main() {
    vec4 worldPos = model.model * vec4(inPosition, 1.0);

    gl_Position = scene.viewProj * worldPos;
    fragWorldPos = worldPos.xyz;

    mat3 normalMatrix = mat3(transpose(inverse(model.model)));
    fragNormal = normalMatrix * inNormal;

    fragColor = inColor;
}