#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragWorldPos;

layout(location = 0) out vec4 outColor;

struct PointLight {
    vec4 position;
    vec4 color;
};

struct SpotLight {
    vec4 position;
    vec4 direction;
    vec4 color;
    vec4 params;
};

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 viewProj;
    vec4 cameraPos;
    vec4 ambient;
    vec4 dirDirection;
    vec4 dirColor;
    uint pointLightCount;
    uint spotLightCount;
} scene;

layout(set = 0, binding = 1) readonly buffer PointLights {
    PointLight lights[];
} pointLights;

layout(set = 0, binding = 2) readonly buffer SpotLights {
    SpotLight lights[];
} spotLights;

layout(set = 1, binding = 0) uniform ModelUBO {
    mat4 model;
    vec4 albedo;
    vec4 specular;
} model;

void main() {
    vec3 N = normalize(fragNormal);
    vec3 V = normalize(scene.cameraPos.xyz - fragWorldPos);
    vec3 baseColor = model.albedo.rgb * fragColor;

    vec3 color = scene.ambient.rgb * scene.ambient.a * baseColor;

    {
        vec3 L = normalize(-scene.dirDirection.xyz);
        float diff = max(dot(N, L), 0.0);
        vec3 H = normalize(L + V);
        float spec = pow(max(dot(N, H), 0.0), model.specular.a);

        vec3 lightColor = scene.dirColor.rgb * scene.dirDirection.w;
        color += (diff * baseColor + spec * model.specular.rgb) * lightColor;
    }

    for (uint i = 0u; i < scene.pointLightCount; ++i) {
        vec3 lightPos = pointLights.lights[i].position.xyz;
        vec3 lightVec = lightPos - fragWorldPos;
        float dist = length(lightVec);

        if (dist > 0.0001) {
            vec3 L = lightVec / dist;
            float att = 1.0 / (dist * dist);

            vec3 lightColor = pointLights.lights[i].color.rgb * pointLights.lights[i].color.a;
            float diff = max(dot(N, L), 0.0);
            vec3 H = normalize(L + V);
            float spec = pow(max(dot(N, H), 0.0), model.specular.a);

            color += (diff * baseColor + spec * model.specular.rgb) * lightColor * att;
        }
    }

    for (uint i = 0u; i < scene.spotLightCount; ++i) {
        vec3 lightPos = spotLights.lights[i].position.xyz;
        vec3 lightVec = lightPos - fragWorldPos;
        float dist = length(lightVec);

        if (dist > 0.0001) {
            vec3 L = lightVec / dist;
            vec3 spotDir = normalize(spotLights.lights[i].direction.xyz);

            float cosTheta = dot(-L, spotDir);
            float innerCos = spotLights.lights[i].params.x;
            float outerCos = spotLights.lights[i].params.y;
            float range = spotLights.lights[i].params.z;

            float spotFactor = clamp(
                (cosTheta - outerCos) / max(innerCos - outerCos, 0.0001),
                0.0, 1.0);

            float att = 1.0 / (dist * dist);
            float rangeFactor = clamp(1.0 - dist / max(range, 0.0001), 0.0, 1.0);
            att *= rangeFactor * rangeFactor;

            vec3 lightColor = spotLights.lights[i].color.rgb * spotLights.lights[i].color.a;
            float diff = max(dot(N, L), 0.0);
            vec3 H = normalize(L + V);
            float spec = pow(max(dot(N, H), 0.0), model.specular.a);

            color += (diff * baseColor + spec * model.specular.rgb)
                   * lightColor * att * spotFactor;
        }
    }

    color = color / (color + vec3(1.0));
    outColor = vec4(color, 1.0);
}