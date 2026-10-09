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
		for (int k = 0; k < 3; ++k) {
			const int vi = icoFaces[f][k];
			Vertex v{};
			v.pos[0] = icoVerts[vi][0] * s;
			v.pos[1] = icoVerts[vi][1] * s;
			v.pos[2] = icoVerts[vi][2] * s;
			v.color[0] = icoFaceColors[f][0];
			v.color[1] = icoFaceColors[f][1];
			v.color[2] = icoFaceColors[f][2];
			out.push_back(v);
		}
	}
	return out;
}

VkBuffer vertexBuffer = VK_NULL_HANDLE; VmaAllocation vertexAlloc = VK_NULL_HANDLE;
VkBuffer uniformBuffer= VK_NULL_HANDLE; VmaAllocation uniformAlloc= VK_NULL_HANDLE;

VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
VkDescriptorSet descriptorSet = VK_NULL_HANDLE;

VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;

uint32_t vertexCount = 0;

bool  usePerspective = true;
float objPosition[3] = { 0.0f, 0.0f, 0.0f };
float objRotation[3] = { 0.0f, 0.0f, 0.0f };
float objScale[3]    = { 1.0f, 1.0f, 1.0f };

struct UBO { float mvp[16]; };

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

} // namespace

bool initialize() {
	auto& ctx = graphics::internal::context;
	VkDevice dev = ctx.device;

	std::vector<Vertex> vertices = buildIcosahedronVertices();
	vertexCount = uint32_t(vertices.size());

	if (!createBuffer(vertices.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
	                  vertexBuffer, vertexAlloc)) return false;
	{
		void* p = nullptr;
		vmaMapMemory(ctx.allocator, vertexAlloc, &p);
		std::memcpy(p, vertices.data(), vertices.size() * sizeof(Vertex));
		vmaUnmapMemory(ctx.allocator, vertexAlloc);
	}

	if (!createBuffer(sizeof(UBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
	                  uniformBuffer, uniformAlloc)) return false;

	VkDescriptorSetLayoutBinding b{};
	b.binding = 0;
	b.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	b.descriptorCount = 1;
	b.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

	VkDescriptorSetLayoutCreateInfo dslci{};
	dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	dslci.bindingCount = 1;
	dslci.pBindings = &b;
	if (vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &descriptorSetLayout) != VK_SUCCESS)
		return false;

	VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 };
	VkDescriptorPoolCreateInfo dpci{};
	dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	dpci.maxSets = 1;
	dpci.poolSizeCount = 1;
	dpci.pPoolSizes = &ps;
	if (vkCreateDescriptorPool(dev, &dpci, nullptr, &descriptorPool) != VK_SUCCESS)
		return false;

	VkDescriptorSetAllocateInfo dsai{};
	dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	dsai.descriptorPool = descriptorPool;
	dsai.descriptorSetCount = 1;
	dsai.pSetLayouts = &descriptorSetLayout;
	if (vkAllocateDescriptorSets(dev, &dsai, &descriptorSet) != VK_SUCCESS)
		return false;

	VkDescriptorBufferInfo bi{ uniformBuffer, 0, sizeof(UBO) };
	VkWriteDescriptorSet w{};
	w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	w.dstSet = descriptorSet;
	w.dstBinding = 0;
	w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	w.descriptorCount = 1;
	w.pBufferInfo = &bi;
	vkUpdateDescriptorSets(dev, 1, &w, 0, nullptr);

	VkPipelineLayoutCreateInfo plci{};
	plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	plci.setLayoutCount = 1;
	plci.pSetLayouts = &descriptorSetLayout;
	if (vkCreatePipelineLayout(dev, &plci, nullptr, &pipelineLayout) != VK_SUCCESS)
		return false;

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

	VkVertexInputAttributeDescription via[2]{};
	via[0] = { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos) };
	via[1] = { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) };

	VkPipelineVertexInputStateCreateInfo visci{};
	visci.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	visci.vertexBindingDescriptionCount = 1;
	visci.pVertexBindingDescriptions = &vib;
	visci.vertexAttributeDescriptionCount = 2;
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
	if (descriptorSetLayout)   vkDestroyDescriptorSetLayout(dev, descriptorSetLayout, nullptr);

	if (vertexBuffer)  vmaDestroyBuffer(ctx.allocator, vertexBuffer, vertexAlloc);
	if (uniformBuffer) vmaDestroyBuffer(ctx.allocator, uniformBuffer, uniformAlloc);
}

void update([[maybe_unused]] double time) {
	ImGui::SetNextWindowSize(ImVec2(360, 0), ImGuiCond_FirstUseEver);
	ImGui::Begin("Icosahedron controls");

	ImGui::Text("Projection");
	if (ImGui::RadioButton("Perspective", usePerspective)) usePerspective = true;
	ImGui::SameLine();
	if (ImGui::RadioButton("Orthographic", !usePerspective)) usePerspective = false;

	ImGui::Separator();

	ImGui::Text("Position");
	ImGui::SliderFloat3("##pos", objPosition, -3.0f, 3.0f);

	ImGui::Text("Rotation (deg)");
	ImGui::SliderFloat3("##rot", objRotation, -180.0f, 180.0f);

	ImGui::Text("Scale");
	ImGui::SliderFloat3("##scl", objScale, 0.1f, 3.0f);

	ImGui::Separator();

	if (ImGui::Button("Reset")) {
		objPosition[0] = objPosition[1] = objPosition[2] = 0.0f;
		objRotation[0] = objRotation[1] = objRotation[2] = 0.0f;
		objScale[0] = objScale[1] = objScale[2] = 1.0f;
	}

	ImGui::End();

	ImGui::ShowDemoWindow();
}

void render(const graphics::internal::FrameData& fd) {
	auto& ctx = graphics::internal::context;

	const float aspect = float(ctx.swapchain_extent.width) / float(ctx.swapchain_extent.height);

	Mat4 proj;
	if (usePerspective) {
		proj = perspective(60.0f * 3.14159265f / 180.0f, aspect, 0.1f, 100.0f);
	} else {
		const float halfH = 1.5f;
		const float halfW = halfH * aspect;
		proj = ortho(-halfW, halfW, -halfH, halfH, 0.1f, 100.0f);
	}

	const float eye[3]    = { 2.0f, 2.0f, 2.0f };
	const float center[3] = { 0.0f, 0.0f, 0.0f };
	const float up[3]     = { 0.0f, 1.0f, 0.0f };
	const Mat4 view = lookAt(eye, center, up);

	const float rx = objRotation[0] * 3.14159265f / 180.0f;
	const float ry = objRotation[1] * 3.14159265f / 180.0f;
	const float rz = objRotation[2] * 3.14159265f / 180.0f;

	const Mat4 model = multiply(
		translate(objPosition[0], objPosition[1], objPosition[2]),
		multiply(rotateEuler(rx, ry, rz),
		         scaleM(objScale[0], objScale[1], objScale[2])));

	const Mat4 mvp = multiply(proj, multiply(view, model));

	UBO ubo{};
	std::memcpy(ubo.mvp, mvp.m, sizeof(mvp.m));

	void* mapped = nullptr;
	vmaMapMemory(ctx.allocator, uniformAlloc, &mapped);
	std::memcpy(mapped, &ubo, sizeof(ubo));
	vmaUnmapMemory(ctx.allocator, uniformAlloc);

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
	vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
	                        0, 1, &descriptorSet, 0, nullptr);

	VkDeviceSize offset = 0;
	vkCmdBindVertexBuffers(cb, 0, 1, &vertexBuffer, &offset);

	vkCmdDraw(cb, vertexCount, 1, 0, 0);

	vkCmdEndRenderPass(cb);
	vkEndCommandBuffer(cb);
}

} // namespace application