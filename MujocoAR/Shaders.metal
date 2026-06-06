//
//  Shaders.metal
//  MujocoAR
//
//  Created by Tatsuya Ogawa on 2026/05/29.
//

#include <metal_stdlib>
#include <simd/simd.h>

#import "ShaderTypes.h"

using namespace metal;

struct CameraVertexOut {
    float4 position [[position]];
    float2 texCoord;
};

struct MeshVertexOut {
    float4 position [[position]];
};

struct SolidVertexOut {
    float4 position [[position]];
    float3 normal;
};

vertex CameraVertexOut cameraVertexShader(
    uint vertexID [[vertex_id]],
    constant FrameUniforms &uniforms [[buffer(BufferIndexFrameUniforms)]]
) {
    constexpr float4 positions[6] = {
        float4(-1.0, -1.0, 0.0, 1.0),
        float4( 1.0, -1.0, 0.0, 1.0),
        float4(-1.0,  1.0, 0.0, 1.0),
        float4(-1.0,  1.0, 0.0, 1.0),
        float4( 1.0, -1.0, 0.0, 1.0),
        float4( 1.0,  1.0, 0.0, 1.0)
    };

    CameraVertexOut out;
    out.position = positions[vertexID];

    float2 viewCoord = out.position.xy * 0.5 + 0.5;
    viewCoord.y = 1.0 - viewCoord.y;

    float3 imageCoord = uniforms.viewToImageTransform * float3(viewCoord, 1.0);
    out.texCoord = imageCoord.xy;
    return out;
}

fragment float4 cameraFragmentShader(
    CameraVertexOut in [[stage_in]],
    texture2d<float, access::sample> textureY [[texture(TextureIndexCameraY)]],
    texture2d<float, access::sample> textureCbCr [[texture(TextureIndexCameraCbCr)]]
) {
    constexpr sampler cameraSampler(mip_filter::linear, mag_filter::linear, min_filter::linear);

    float y = textureY.sample(cameraSampler, in.texCoord).r;
    float2 uv = textureCbCr.sample(cameraSampler, in.texCoord).rg - float2(0.5, 0.5);

    float r = y + 1.5748 * uv.y;
    float g = y - 0.1873 * uv.x - 0.4681 * uv.y;
    float b = y + 1.8556 * uv.x;

    return float4(clamp(float3(r, g, b), 0.0, 1.0), 1.0);
}

vertex MeshVertexOut meshVertexShader(
    uint vertexID [[vertex_id]],
    constant MeshUniforms &uniforms [[buffer(BufferIndexMeshUniforms)]],
    const device float3 *positions [[buffer(BufferIndexMeshPositions)]]
) {
    MeshVertexOut out;
    float4 worldPosition = uniforms.modelMatrix * float4(positions[vertexID], 1.0);
    out.position = uniforms.projectionMatrix * uniforms.viewMatrix * worldPosition;
    return out;
}

vertex SolidVertexOut solidVertexShader(
    uint vertexID [[vertex_id]],
    constant MeshUniforms &uniforms [[buffer(BufferIndexMeshUniforms)]],
    const device float3 *positions [[buffer(BufferIndexMeshPositions)]],
    const device float3 *normals [[buffer(BufferIndexMeshNormals)]]
) {
    SolidVertexOut out;
    float4 worldPosition = uniforms.modelMatrix * float4(positions[vertexID], 1.0);
    out.position = uniforms.projectionMatrix * uniforms.viewMatrix * worldPosition;
    out.normal = normalize((uniforms.modelMatrix * float4(normals[vertexID], 0.0)).xyz);
    return out;
}

fragment float4 solidFragmentShader(
    SolidVertexOut in [[stage_in]],
    constant MeshUniforms &uniforms [[buffer(BufferIndexMeshUniforms)]]
) {
    float3 normal = normalize(in.normal);
    float3 lightDirection = normalize(float3(0.35, -0.45, 0.82));
    float diffuse = max(dot(normal, lightDirection), 0.0);
    float shade = 0.42 + 0.58 * diffuse;
    return float4(uniforms.color.rgb * shade, uniforms.color.a);
}

fragment float4 meshWireFragmentShader(
    MeshVertexOut in [[stage_in]],
    float3 barycentricCoord [[barycentric_coord]],
    constant FrameUniforms &uniforms [[buffer(BufferIndexFrameUniforms)]]
) {
    float3 derivative = fwidth(barycentricCoord);
    float3 smoothed = smoothstep(
        float3(0.0),
        derivative * max(uniforms.wireWidth, 0.001),
        barycentricCoord
    );
    float edgeFactor = min(smoothed.x, min(smoothed.y, smoothed.z));
    edgeFactor = powr(edgeFactor, max(uniforms.wireSoftness, 0.0001));

    float alpha = (1.0 - edgeFactor) * uniforms.wireColor.a;
    if (alpha < 0.01) {
        discard_fragment();
    }

    return float4(uniforms.wireColor.rgb, alpha);
}
