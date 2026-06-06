//
//  ShaderTypes.h
//  MujocoAR
//
//  Created by Tatsuya Ogawa on 2026/05/29.
//

#ifndef ShaderTypes_h
#define ShaderTypes_h

#ifdef __METAL_VERSION__
#define NS_ENUM(_type, _name) enum _name : _type _name; enum _name : _type
typedef metal::int32_t EnumBackingType;
#else
#import <Foundation/Foundation.h>
typedef NSInteger EnumBackingType;
#endif

#include <simd/simd.h>

typedef NS_ENUM(EnumBackingType, BufferIndex)
{
    BufferIndexFrameUniforms = 0,
    BufferIndexMeshUniforms  = 0,
    BufferIndexMeshPositions = 1,
    BufferIndexMeshNormals   = 2,
};

typedef NS_ENUM(EnumBackingType, TextureIndex)
{
    TextureIndexCameraY    = 0,
    TextureIndexCameraCbCr = 1,
};

typedef struct
{
    matrix_float3x3 viewToImageTransform;
    vector_float4 wireColor;
    float wireWidth;
    float wireSoftness;
} FrameUniforms;

typedef struct
{
    matrix_float4x4 modelMatrix;
    matrix_float4x4 viewMatrix;
    matrix_float4x4 projectionMatrix;
    vector_float4 color;
} MeshUniforms;

#endif /* ShaderTypes_h */
