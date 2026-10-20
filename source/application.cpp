#include "application.hpp"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <imgui.h>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

namespace application {

namespace {

constexpr float PI = 3.14159265f;
constexpr uint32_t MAX_POINT_LIGHTS = 16;
constexpr uint32_t MAX_SPOT_LIGHTS  = 16;

struct Mat4 { float m[16]; };

Mat4 identity() {
    Mat4 r{};
    r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
    return r;
}

Mat4 multiply(const Mat4& a, const Mat4& b) {
    Mat4 r{};
    for (int c = 0; c < 4; ++c)
        for (int row = 0; row < 4; ++row) {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k)
                s += a.m[k * 4 + row] * b.m[c * 4 + k];
            r.m[c * 4 + row] = s;
        }
    return r;
}

Mat4 perspective(float fovyRad, float aspect, float zn, float zf) {
    Mat4 r{};
    const float t = 1.0f / std::tan(fovyRad * 0.5f);
    r.m[0]  = t / aspect;
    r.m[5]  = -t;
    r.m[10] = zf / (zn - zf);
    r.m[11] = -1.0f;
    r.m[14] = zn * zf / (zn - zf);
    return r;
}

Mat4 ortho(float l, float r_, float b, float t, float n, float f) {
    Mat4 m = identity();
    m.m[0]  =  2.0f / (r_ - l);
    m.m[5]  = -2.0f / (t - b);
    m.m[10] =  1.0f / (n - f);
    m.m[12] = -(r_ + l) / (r_ - l);
    m.m[13] = -(t + b) / (t - b);
    m.m[14] =  n / (n - f);
    return m;
}

Mat4 lookAt(const float eye[3], const float center[3], const float up[3]) {
    float f[3] = { center[0]-eye[0], center[1]-eye[1], center[2]-eye[2] };
    auto norm = [](float v[3]) {
        const float l = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
        if (l > 1e-6f) { v[0]/=l; v[1]/=l; v[2]/=l; }
    };
    norm(f);
    float s[3] = { f[1]*up[2] - f[2]*up[1], f[2]*up[0] - f[0]*up[2], f[0]*up[1] - f[1]*up[0] };
    norm(s);
    float u[3] = { s[1]*f[2] - s[2]*f[1], s[2]*f[0] - s[0]*f[2], s[0]*f[1] - s[1]*f[0] };

    Mat4 r = identity();
    r.m[0]=s[0]; r.m[4]=s[1]; r.m[8]=s[2];    r.m[12]=-(s[0]*eye[0]+s[1]*eye[1]+s[2]*eye[2]);
    r.m[1]=u[0]; r.m[5]=u[1]; r.m[9]=u[2];    r.m[13]=-(u[0]*eye[0]+u[1]*eye[1]+u[2]*eye[2]);
    r.m[2]=-f[0];r.m[6]=-f[1];r.m[10]=-f[2];  r.m[14]= (f[0]*eye[0]+f[1]*eye[1]+f[2]*eye[2]);
    return r;
}

Mat4 translate(float x, float y, float z) {
    Mat4 r = identity();
    r.m[12]=x; r.m[13]=y; r.m[14]=z;
    return r;
}

Mat4 scaleM(float x, float y, float z) {
    Mat4 r = identity();
    r.m[0]=x; r.m[5]=y; r.m[10]=z;
    return r;
}

Mat4 rotateEuler(float rx, float ry, float rz) {
    const float cx=std::cos(rx), sx=std::sin(rx);
    const float cy=std::cos(ry), sy=std::sin(ry);
    const float cz=std::cos(rz), sz=std::sin(rz);
    Mat4 r = identity();
    r.m[0]  =  cy*cz;
    r.m[1]  =  cx*sz + sx*sy*cz;
    r.m[2]  =  sx*sz - cx*sy*cz;
    r.m[4]  = -cy*sz;
    r.m[5]  =  cx*cz - sx*sy*sz;
    r.m[6]  =  sx*cz + cx*sy*sz;
    r.m[8]  =  sy;
    r.m[9]  = -sx*cy;
    r.m[10] =  cx*cy;
    return r;
}

struct Vertex {
    float pos[3];
    float normal[3];
    float color[3];
};

constexpr float PHI = 1.6180339887498949f;

const float icoVerts[12][3] = {
    {-1.0f,  PHI, 0.0f}, { 1.0f,  PHI, 0.0f},
    {-1.0f, -PHI, 0.0f}, { 1.0f, -PHI, 0.0f},
    { 0.0f, -1.0f,  PHI}, { 0.0f,  1.0f,  PHI},
    { 0.0f, -1.0f, -PHI}, { 0.0f,  1.0f, -PHI},
    { PHI, 0.0f, -1.0f}, { PHI, 0.0f,  1.0f},
    {-PHI, 0.0f, -1.0f}, {-PHI, 0.0f,  1.0f},
};

const int icoFaces[20][3] = {
    { 0, 11,  5}, { 0,  5,  1}, { 0,  1,  7}, { 0,  7, 10}, { 0, 10, 11},
    { 1,  5,  9}, { 5, 11,  4}, {11, 10,  2}, {10,  7,  6}, { 7,  1,  8},
    { 3,  9,  4}, { 3,  4,  2}, { 3,  2,  6}, { 3,  6,  8}, { 3,  8,  9},
    { 4,  9,  5}, { 2,  4, 11}, { 6,  2, 10}, { 8,  6,  7}, { 9,  8,  1},
};

const float icoFaceColors[20][3] = {
    {1.00f, 0.25f, 0.25f}, {1.00f, 0.55f, 0.10f}, {1.00f, 0.95f, 0.20f},
    {0.65f, 1.00f, 0.30f}, {0.15f, 0.90f, 0.35f}, {0.10f, 0.85f, 0.65f},
    {0.10f, 0.70f, 1.00f}, {0.15f, 0.40f, 0.95f}, {0.35f, 0.15f, 0.95f},
    {0.65f, 0.20f, 0.90f}, {0.95f, 0.20f, 0.85f}, {1.00f, 0.40f, 0.70f},
    {1.00f, 0.75f, 0.55f}, {0.80f, 1.00f, 0.40f}, {0.40f, 1.00f, 0.80f},
    {0.40f, 0.80f, 1.00f}, {0.60f, 0.60f, 1.00f}, {0.90f, 0.60f, 0.95f},
    {1.00f, 0.55f, 0.55f}, {0.60f, 1.00f, 0.65f},
};

std::vector<Vertex> buildIcosahedronVertices() {
    std::vector<Vertex> out;
    out.reserve(60);
    constexpr float s = 0.5f;

    for (int f = 0; f < 20; ++f) {
        const int i0 = icoFaces[f][0];
        const int i1 = icoFaces[f][1];
        const int i2 = icoFaces[f][2];

        float v0[3] = { icoVerts[i0][0]*s, icoVerts[i0][1]*s, icoVerts[i0][2]*s };
        float v1[3] = { icoVerts[i1][0]*s, icoVerts[i1][1]*s, icoVerts[i1][2]*s };
        float v2[3] = { icoVerts[i2][0]*s, icoVerts[i2][1]*s, icoVerts[i2][2]*s };

        float e1[3] = { v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2] };
        float e2[3] = { v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2] };

        float n[3] = {
            e1[1]*e2[2] - e1[2]*e2[1],
            e1[2]*e2[0] - e1[0]*e2[2],
            e1[0]*e2[1] - e1[1]*e2[0],
        };
        float len = std::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
        if (len > 1e-6f) {
            n[0] /= len; n[1] /= len; n[2] /= len;
        }

        for (int k = 0; k < 3; ++k) {
            const int vi = icoFaces[f][k];
            Vertex v{};
            v.pos[0] = icoVerts[vi][0] * s;
            v.pos[1] = icoVerts[vi][1] * s;
            v.pos[2] = icoVerts[vi][2] * s;
            v.normal[0] = n[0];
            v.normal[1] = n[1];
            v.normal[2] = n[2];
            v.color[0] = icoFaceColors[f][0];
            v.color[1] = icoFaceColors[f][1];
            v.color[2] = icoFaceColors[f][2];
            out.push_back(v);
        }
    }
    return out;
}

struct PointLightGPU {
    float position[4];
    float color[4];
};

struct SpotLightGPU {
    float position[4];
    float direction[4];
    float color[4];
    float params[4];
};

struct SceneUBO {
    float viewProj[16];
    float cameraPos[4];
    float ambient[4];
    float dirDirection[4];
    float dirColor[4];
    uint32_t pointLightCount;
    uint32_t spotLightCount;
    float pad[2];
};

struct ModelUBO {
    float model[16];
    float albedo[4];
    float specular[4];
};

struct PointLightData {
    float position[3] = { 0.0f, 0.0f, 0.0f };
    float color[3]    = { 1.0f, 1.0f, 1.0f };
    float intensity   = 1.0f;
};

struct SpotLightData {
    float position[3] = { 0.0f, 0.0f, 0.0f };
    float direction[3] = { 0.0f, -1.0f, 0.0f };
    float color[3]    = { 1.0f, 1.0f, 1.0f };
    float intensity   = 1.0f;
    float innerAngle  = 20.0f;
    float outerAngle  = 30.0f;
    float range       = 10.0f;
};

VkBuffer vertexBuffer = VK_NULL_HANDLE; VmaAllocation vertexAlloc = VK_NULL_HANDLE;
VkBuffer sceneUBO = VK_NULL_HANDLE; VmaAllocation sceneUBOAlloc = VK_NULL_HANDLE;
VkBuffer modelUBO = VK_NULL_HANDLE; VmaAllocation modelUBOAlloc = VK_NULL_HANDLE;
VkBuffer pointLightBuffer = VK_NULL_HANDLE; VmaAllocation pointLightAlloc = VK_NULL_HANDLE;
VkBuffer spotLightBuffer = VK_NULL_HANDLE; VmaAllocation spotLightAlloc = VK_NULL_HANDLE;

VkDescriptorSetLayout sceneSetLayout = VK_NULL_HANDLE;
VkDescriptorSetLayout modelSetLayout = VK_NULL_HANDLE;
VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
VkDescriptorSet sceneSet = VK_NULL_HANDLE;
VkDescriptorSet modelSet = VK_NULL_HANDLE;

VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;

uint32_t vertexCount = 0;

bool usePerspective = true;
float cameraPos[3] = { 2.0f, 2.0f, 2.0f };
float cameraYaw = -45.0f;
float cameraPitch = -35.26f;
float cameraFov = 60.0f;

float objPosition[3] = { 0.0f, 0.0f, 0.0f };
float objRotation[3] = { 0.0f, 0.0f, 0.0f };
float objScale[3]    = { 1.0f, 1.0f, 1.0f };

float albedo[3] = { 0.8f, 0.8f, 0.8f };
float specularColor[3] = { 1.0f, 1.0f, 1.0f };
float shininess = 32.0f;

float ambientColor[3] = { 0.1f, 0.1f, 0.1f };
float ambientIntensity = 0.2f;

float dirLightDirection[3] = { -1.0f, -1.0f, -1.0f };
float dirLightColor[3] = { 1.0f, 1.0f, 1.0f };
float dirLightIntensity = 1.0f;

std::vector<PointLightData> pointLights;
std::vector<SpotLightData> spotLights;

std::vector<char> readFile(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { std::cerr << "Problem with open " << path << "\n"; return {}; }
    const size_t n = size_t(f.tellg());
    std::vector<char> buf(n);
    f.seekg(0);
    f.read(buf.data(), n);
    return buf;
}

VkShaderModule createShaderModule(VkDevice dev, const std::vector<char>& code) {
    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = code.size();
    ci.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule m = VK_NULL_HANDLE;
    vkCreateShaderModule(dev, &ci, nullptr, &m);
    return m;
}

bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                  VkBuffer& outBuf, VmaAllocation& outAlloc) {
    VkBufferCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ci.size = size;
    ci.usage = usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
              | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    return vmaCreateBuffer(graphics::internal::context.allocator, &ci, &aci,
                           &outBuf, &outAlloc, nullptr) == VK_SUCCESS;
}

void uploadBuffer(VkBuffer, VmaAllocation alloc, const void* data, size_t size) {
    void* mapped = nullptr;
    vmaMapMemory(graphics::internal::context.allocator, alloc, &mapped);
    std::memcpy(mapped, data, size);
    vmaUnmapMemory(graphics::internal::context.allocator, alloc);
}

} // namespace

bool initialize() {
    auto& ctx = graphics::internal::context;
    VkDevice dev = ctx.device;

    std::vector<Vertex> vertices = buildIcosahedronVertices();
    vertexCount = uint32_t(vertices.size());

    if (!createBuffer(vertices.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                      vertexBuffer, vertexAlloc)) return false;
    uploadBuffer(vertexBuffer, vertexAlloc, vertices.data(), vertices.size() * sizeof(Vertex));

    if (!createBuffer(sizeof(SceneUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                      sceneUBO, sceneUBOAlloc)) return false;

    if (!createBuffer(sizeof(ModelUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                      modelUBO, modelUBOAlloc)) return false;

    if (!createBuffer(MAX_POINT_LIGHTS * sizeof(PointLightGPU),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      pointLightBuffer, pointLightAlloc)) return false;

    if (!createBuffer(MAX_SPOT_LIGHTS * sizeof(SpotLightGPU),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      spotLightBuffer, spotLightAlloc)) return false;

    {
        VkDescriptorSetLayoutBinding bindings[3]{};
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        bindings[2].binding = 2;
        bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[2].descriptorCount = 1;
        bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = 3;
        ci.pBindings = bindings;
        if (vkCreateDescriptorSetLayout(dev, &ci, nullptr, &sceneSetLayout) != VK_SUCCESS)
            return false;
    }

    {
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = 0;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = 1;
        ci.pBindings = &binding;
        if (vkCreateDescriptorSetLayout(dev, &ci, nullptr, &modelSetLayout) != VK_SUCCESS)
            return false;
    }

    {
        VkDescriptorPoolSize sizes[2] = {
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 },
        };

        VkDescriptorPoolCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        ci.maxSets = 2;
        ci.poolSizeCount = 2;
        ci.pPoolSizes = sizes;
        if (vkCreateDescriptorPool(dev, &ci, nullptr, &descriptorPool) != VK_SUCCESS)
            return false;
    }

    {
        VkDescriptorSetLayout layouts[2] = { sceneSetLayout, modelSetLayout };
        VkDescriptorSet sets[2] = {};

        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = descriptorPool;
        ai.descriptorSetCount = 2;
        ai.pSetLayouts = layouts;
        if (vkAllocateDescriptorSets(dev, &ai, sets) != VK_SUCCESS)
            return false;

        sceneSet = sets[0];
        modelSet = sets[1];
    }

    {
        VkDescriptorBufferInfo sceneInfo{ sceneUBO, 0, sizeof(SceneUBO) };
        VkDescriptorBufferInfo pointInfo{ pointLightBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo spotInfo{ spotLightBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo modelInfo{ modelUBO, 0, sizeof(ModelUBO) };

        VkWriteDescriptorSet writes[4]{};

        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = sceneSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].descriptorCount = 1;
        writes[0].pBufferInfo = &sceneInfo;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = sceneSet;
        writes[1].dstBinding = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].descriptorCount = 1;
        writes[1].pBufferInfo = &pointInfo;

        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = sceneSet;
        writes[2].dstBinding = 2;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[2].descriptorCount = 1;
        writes[2].pBufferInfo = &spotInfo;

        writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[3].dstSet = modelSet;
        writes[3].dstBinding = 0;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[3].descriptorCount = 1;
        writes[3].pBufferInfo = &modelInfo;

        vkUpdateDescriptorSets(dev, 4, writes, 0, nullptr);
    }

    {
        VkDescriptorSetLayout layouts[2] = { sceneSetLayout, modelSetLayout };
        VkPipelineLayoutCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        ci.setLayoutCount = 2;
        ci.pSetLayouts = layouts;
        if (vkCreatePipelineLayout(dev, &ci, nullptr, &pipelineLayout) != VK_SUCCESS)
            return false;
    }

    auto vertCode = readFile("shaders/icosahedron.vert.spv");
    auto fragCode = readFile("shaders/icosahedron.frag.spv");
    if (vertCode.empty() || fragCode.empty()) return false;

    VkShaderModule vertMod = createShaderModule(dev, vertCode);
    VkShaderModule fragMod = createShaderModule(dev, fragCode);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertMod;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragMod;
    stages[1].pName = "main";

    VkVertexInputBindingDescription vib{ 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX };

    VkVertexInputAttributeDescription via[3]{};
    via[0] = { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos) };
    via[1] = { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) };
    via[2] = { 2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) };

    VkPipelineVertexInputStateCreateInfo visci{};
    visci.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    visci.vertexBindingDescriptionCount = 1;
    visci.pVertexBindingDescriptions = &vib;
    visci.vertexAttributeDescriptionCount = 3;
    visci.pVertexAttributeDescriptions = via;

    VkPipelineInputAssemblyStateCreateInfo iasci{};
    iasci.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    iasci.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vsci{};
    vsci.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vsci.viewportCount = 1;
    vsci.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rsci{};
    rsci.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rsci.polygonMode = VK_POLYGON_MODE_FILL;
    rsci.cullMode = VK_CULL_MODE_NONE;
    rsci.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rsci.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo msci{};
    msci.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    msci.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo dsci{};
    dsci.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    dsci.depthTestEnable = VK_TRUE;
    dsci.depthWriteEnable = VK_TRUE;
    dsci.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendAttachmentState cba{};
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                       | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    cba.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo cbci{};
    cbci.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cbci.attachmentCount = 1;
    cbci.pAttachments = &cba;

    VkDynamicState dynStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dstateci{};
    dstateci.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dstateci.dynamicStateCount = 2;
    dstateci.pDynamicStates = dynStates;

    VkGraphicsPipelineCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pci.stageCount = 2;
    pci.pStages = stages;
    pci.pVertexInputState = &visci;
    pci.pInputAssemblyState = &iasci;
    pci.pViewportState = &vsci;
    pci.pRasterizationState = &rsci;
    pci.pMultisampleState = &msci;
    pci.pDepthStencilState = &dsci;
    pci.pColorBlendState = &cbci;
    pci.pDynamicState = &dstateci;
    pci.layout = pipelineLayout;
    pci.renderPass = ctx.render_pass;
    pci.subpass = 0;

    const VkResult pr = vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &pci, nullptr, &pipeline);
    vkDestroyShaderModule(dev, vertMod, nullptr);
    vkDestroyShaderModule(dev, fragMod, nullptr);
    if (pr != VK_SUCCESS) { std::cerr << "Problem with pipeline\n"; return false; }

    return true;
}

void shutdown() {
    auto& ctx = graphics::internal::context;
    vkDeviceWaitIdle(ctx.device);
    VkDevice dev = ctx.device;

    if (pipeline)              vkDestroyPipeline(dev, pipeline, nullptr);
    if (pipelineLayout)        vkDestroyPipelineLayout(dev, pipelineLayout, nullptr);
    if (descriptorPool)        vkDestroyDescriptorPool(dev, descriptorPool, nullptr);
    if (modelSetLayout)        vkDestroyDescriptorSetLayout(dev, modelSetLayout, nullptr);
    if (sceneSetLayout)        vkDestroyDescriptorSetLayout(dev, sceneSetLayout, nullptr);

    if (vertexBuffer)     vmaDestroyBuffer(ctx.allocator, vertexBuffer, vertexAlloc);
    if (sceneUBO)         vmaDestroyBuffer(ctx.allocator, sceneUBO, sceneUBOAlloc);
    if (modelUBO)         vmaDestroyBuffer(ctx.allocator, modelUBO, modelUBOAlloc);
    if (pointLightBuffer) vmaDestroyBuffer(ctx.allocator, pointLightBuffer, pointLightAlloc);
    if (spotLightBuffer)  vmaDestroyBuffer(ctx.allocator, spotLightBuffer, spotLightAlloc);
}

void update([[maybe_unused]] double time) {
    ImGui::SetNextWindowSize(ImVec2(420, 0), ImGuiCond_FirstUseEver);
    ImGui::Begin("Lab 2: Camera & Lighting");

    if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat3("Position", cameraPos, -10.0f, 10.0f);
        ImGui::SliderFloat("Yaw", &cameraYaw, -180.0f, 180.0f);
        ImGui::SliderFloat("Pitch", &cameraPitch, -89.0f, 89.0f);
        ImGui::SliderFloat("FOV", &cameraFov, 30.0f, 120.0f);

        if (ImGui::RadioButton("Perspective", usePerspective)) usePerspective = true;
        ImGui::SameLine();
        if (ImGui::RadioButton("Orthographic", !usePerspective)) usePerspective = false;
    }

    if (ImGui::CollapsingHeader("Model", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat3("Position##model", objPosition, -3.0f, 3.0f);
        ImGui::SliderFloat3("Rotation##model", objRotation, -180.0f, 180.0f);
        ImGui::SliderFloat3("Scale##model", objScale, 0.1f, 3.0f);

        if (ImGui::Button("Reset model")) {
            objPosition[0] = objPosition[1] = objPosition[2] = 0.0f;
            objRotation[0] = objRotation[1] = objRotation[2] = 0.0f;
            objScale[0] = objScale[1] = objScale[2] = 1.0f;
        }
    }

    if (ImGui::CollapsingHeader("Material", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::ColorEdit3("Albedo", albedo);
        ImGui::ColorEdit3("Specular", specularColor);
        ImGui::SliderFloat("Shininess", &shininess, 1.0f, 256.0f);
    }

    if (ImGui::CollapsingHeader("Ambient & Directional", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::ColorEdit3("Ambient color", ambientColor);
        ImGui::SliderFloat("Ambient intensity", &ambientIntensity, 0.0f, 1.0f);

        ImGui::Separator();
        ImGui::SliderFloat3("Dir direction", dirLightDirection, -1.0f, 1.0f);
        ImGui::ColorEdit3("Dir color", dirLightColor);
        ImGui::SliderFloat("Dir intensity", &dirLightIntensity, 0.0f, 5.0f);
    }

    if (ImGui::CollapsingHeader("Point lights", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::Button("Add point") && pointLights.size() < MAX_POINT_LIGHTS) {
            pointLights.push_back(PointLightData{});
        }

        for (int i = 0; i < (int)pointLights.size(); ++i) {
            ImGui::PushID(i);
            ImGui::Separator();
            ImGui::Text("Point %d", i);
            ImGui::SameLine();
            if (ImGui::Button("Remove")) {
                pointLights.erase(pointLights.begin() + i);
                ImGui::PopID();
                --i;
                continue;
            }

            ImGui::SliderFloat3("Position##point", pointLights[i].position, -10.0f, 10.0f);
            ImGui::ColorEdit3("Color##point", pointLights[i].color);
            ImGui::SliderFloat("Intensity##point", &pointLights[i].intensity, 0.0f, 10.0f);
            ImGui::PopID();
        }
    }

    if (ImGui::CollapsingHeader("Spot lights", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::Button("Add spot") && spotLights.size() < MAX_SPOT_LIGHTS) {
            spotLights.push_back(SpotLightData{});
        }

        for (int i = 0; i < (int)spotLights.size(); ++i) {
            ImGui::PushID(1000 + i);
            ImGui::Separator();
            ImGui::Text("Spot %d", i);
            ImGui::SameLine();
            if (ImGui::Button("Remove")) {
                spotLights.erase(spotLights.begin() + i);
                ImGui::PopID();
                --i;
                continue;
            }

            ImGui::SliderFloat3("Position##spot", spotLights[i].position, -10.0f, 10.0f);
            ImGui::SliderFloat3("Direction##spot", spotLights[i].direction, -1.0f, 1.0f);
            ImGui::ColorEdit3("Color##spot", spotLights[i].color);
            ImGui::SliderFloat("Intensity##spot", &spotLights[i].intensity, 0.0f, 10.0f);
            ImGui::SliderFloat("Inner angle##spot", &spotLights[i].innerAngle, 0.0f, 89.0f);
            ImGui::SliderFloat("Outer angle##spot", &spotLights[i].outerAngle, 0.0f, 89.0f);
            ImGui::SliderFloat("Range##spot", &spotLights[i].range, 0.1f, 50.0f);
            ImGui::PopID();
        }
    }

    ImGui::End();
}

void render(const graphics::internal::FrameData& fd) {
    auto& ctx = graphics::internal::context;

    const float aspect = float(ctx.swapchain_extent.width) / float(ctx.swapchain_extent.height);

    Mat4 proj;
    if (usePerspective) {
        proj = perspective(cameraFov * PI / 180.0f, aspect, 0.1f, 100.0f);
    } else {
        const float halfH = 1.5f;
        const float halfW = halfH * aspect;
        proj = ortho(-halfW, halfW, -halfH, halfH, 0.1f, 100.0f);
    }

    const float yaw = cameraYaw * PI / 180.0f;
    const float pitch = cameraPitch * PI / 180.0f;
    const float forward[3] = {
        std::cos(pitch) * std::sin(yaw),
        std::sin(pitch),
        -std::cos(pitch) * std::cos(yaw),
    };
    const float target[3] = {
        cameraPos[0] + forward[0],
        cameraPos[1] + forward[1],
        cameraPos[2] + forward[2],
    };
    const float up[3] = { 0.0f, 1.0f, 0.0f };
    const Mat4 view = lookAt(cameraPos, target, up);

    const float rx = objRotation[0] * PI / 180.0f;
    const float ry = objRotation[1] * PI / 180.0f;
    const float rz = objRotation[2] * PI / 180.0f;

    const Mat4 modelMat = multiply(
        translate(objPosition[0], objPosition[1], objPosition[2]),
        multiply(rotateEuler(rx, ry, rz),
                 scaleM(objScale[0], objScale[1], objScale[2])));

    const Mat4 viewProj = multiply(proj, view);

    SceneUBO scene{};
    std::memcpy(scene.viewProj, viewProj.m, sizeof(scene.viewProj));
    scene.cameraPos[0] = cameraPos[0];
    scene.cameraPos[1] = cameraPos[1];
    scene.cameraPos[2] = cameraPos[2];
    scene.cameraPos[3] = 1.0f;

    scene.ambient[0] = ambientColor[0];
    scene.ambient[1] = ambientColor[1];
    scene.ambient[2] = ambientColor[2];
    scene.ambient[3] = ambientIntensity;

    float dir[3] = { dirLightDirection[0], dirLightDirection[1], dirLightDirection[2] };
    float dirLen = std::sqrt(dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2]);
    if (dirLen > 1e-6f) {
        dir[0] /= dirLen; dir[1] /= dirLen; dir[2] /= dirLen;
    } else {
        dir[0] = 0.0f; dir[1] = -1.0f; dir[2] = 0.0f;
    }
    scene.dirDirection[0] = dir[0];
    scene.dirDirection[1] = dir[1];
    scene.dirDirection[2] = dir[2];
    scene.dirDirection[3] = dirLightIntensity;

    scene.dirColor[0] = dirLightColor[0];
    scene.dirColor[1] = dirLightColor[1];
    scene.dirColor[2] = dirLightColor[2];
    scene.dirColor[3] = 1.0f;

    scene.pointLightCount = uint32_t(pointLights.size());
    scene.spotLightCount  = uint32_t(spotLights.size());

    ModelUBO model{};
    std::memcpy(model.model, modelMat.m, sizeof(model.model));
    model.albedo[0] = albedo[0];
    model.albedo[1] = albedo[1];
    model.albedo[2] = albedo[2];
    model.albedo[3] = 1.0f;
    model.specular[0] = specularColor[0];
    model.specular[1] = specularColor[1];
    model.specular[2] = specularColor[2];
    model.specular[3] = shininess;

    uploadBuffer(sceneUBO, sceneUBOAlloc, &scene, sizeof(scene));
    uploadBuffer(modelUBO, modelUBOAlloc, &model, sizeof(model));

    std::vector<PointLightGPU> pointGPU(pointLights.size());
    for (size_t i = 0; i < pointLights.size(); ++i) {
        pointGPU[i].position[0] = pointLights[i].position[0];
        pointGPU[i].position[1] = pointLights[i].position[1];
        pointGPU[i].position[2] = pointLights[i].position[2];
        pointGPU[i].position[3] = 1.0f;

        pointGPU[i].color[0] = pointLights[i].color[0];
        pointGPU[i].color[1] = pointLights[i].color[1];
        pointGPU[i].color[2] = pointLights[i].color[2];
        pointGPU[i].color[3] = pointLights[i].intensity;
    }
    if (!pointGPU.empty()) {
        uploadBuffer(pointLightBuffer, pointLightAlloc,
                     pointGPU.data(), pointGPU.size() * sizeof(PointLightGPU));
    }

    std::vector<SpotLightGPU> spotGPU(spotLights.size());
    for (size_t i = 0; i < spotLights.size(); ++i) {
        spotGPU[i].position[0] = spotLights[i].position[0];
        spotGPU[i].position[1] = spotLights[i].position[1];
        spotGPU[i].position[2] = spotLights[i].position[2];
        spotGPU[i].position[3] = 1.0f;

        float d[3] = { spotLights[i].direction[0], spotLights[i].direction[1], spotLights[i].direction[2] };
        float dl = std::sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
        if (dl > 1e-6f) { d[0]/=dl; d[1]/=dl; d[2]/=dl; }
        else { d[0]=0.0f; d[1]=-1.0f; d[2]=0.0f; }

        spotGPU[i].direction[0] = d[0];
        spotGPU[i].direction[1] = d[1];
        spotGPU[i].direction[2] = d[2];
        spotGPU[i].direction[3] = 0.0f;

        spotGPU[i].color[0] = spotLights[i].color[0];
        spotGPU[i].color[1] = spotLights[i].color[1];
        spotGPU[i].color[2] = spotLights[i].color[2];
        spotGPU[i].color[3] = spotLights[i].intensity;

        const float inner = spotLights[i].innerAngle * PI / 180.0f;
        const float outer = spotLights[i].outerAngle * PI / 180.0f;
        spotGPU[i].params[0] = std::cos(inner);
        spotGPU[i].params[1] = std::cos(outer);
        spotGPU[i].params[2] = spotLights[i].range;
        spotGPU[i].params[3] = 0.0f;
    }
    if (!spotGPU.empty()) {
        uploadBuffer(spotLightBuffer, spotLightAlloc,
                     spotGPU.data(), spotGPU.size() * sizeof(SpotLightGPU));
    }

    VkCommandBuffer cb = fd.command_buffer;
    vkResetCommandBuffer(cb, 0);

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);

    VkClearValue clears[2]{};
    clears[0].color = { { 0.1f, 0.1f, 0.15f, 1.0f } };
    clears[1].depthStencil = { 1.0f, 0 };

    VkRenderPassBeginInfo rpbi{};
    rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbi.renderPass = ctx.render_pass;
    rpbi.framebuffer = fd.framebuffer;
    rpbi.renderArea = { { 0, 0 }, ctx.swapchain_extent };
    rpbi.clearValueCount = 2;
    rpbi.pClearValues = clears;

    vkCmdBeginRenderPass(cb, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport vp{};
    vp.x = 0.0f;
    vp.y = 0.0f;
    vp.width  = float(ctx.swapchain_extent.width);
    vp.height = float(ctx.swapchain_extent.height);
    vp.minDepth = 0.0f;
    vp.maxDepth = 1.0f;
    vkCmdSetViewport(cb, 0, 1, &vp);

    VkRect2D scissor{};
    scissor.extent = ctx.swapchain_extent;
    vkCmdSetScissor(cb, 0, 1, &scissor);

    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    VkDescriptorSet sets[2] = { sceneSet, modelSet };
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
                            0, 2, sets, 0, nullptr);

    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cb, 0, 1, &vertexBuffer, &offset);

    vkCmdDraw(cb, vertexCount, 1, 0, 0);

    vkCmdEndRenderPass(cb);
    vkEndCommandBuffer(cb);
}

} // namespace application