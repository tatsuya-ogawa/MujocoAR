//
//  Renderer.swift
//  MujocoAR
//
//  Created by Tatsuya Ogawa on 2026/05/29.
//

import ARKit
import Metal
import MetalKit
import UIKit
import simd

private let maxBuffersInFlight = 3
private let maxMeshAnchorsPerFrame = 768

private let alignedFrameUniformsSize = alignTo256(MemoryLayout<FrameUniforms>.stride)
private let alignedMeshUniformsSize = alignTo256(MemoryLayout<MeshUniforms>.stride)
private let perFrameUniformsSize = alignedFrameUniformsSize + alignedMeshUniformsSize * maxMeshAnchorsPerFrame

nonisolated enum RendererError: Error {
    case missingMetal4Support
    case missingBarycentricSupport
    case pipelineCreationFailed
}

@MainActor
final class Renderer: NSObject, MTKViewDelegate {
    enum DebugCameraMode {
        case orbit
        case thirdPerson
    }

    let device: MTLDevice

#if !targetEnvironment(simulator)
    private let commandQueue: MTL4CommandQueue
    private let commandBuffer: MTL4CommandBuffer
    private let commandAllocators: [MTL4CommandAllocator]
    private let vertexArgumentTable: MTL4ArgumentTable
    private let fragmentArgumentTable: MTL4ArgumentTable
    private let transientResidencySets: [MTLResidencySet]
#endif

    private let endFrameEvent: MTLSharedEvent
    private var frameIndex = maxBuffersInFlight
    private var uniformBuffer: MTLBuffer
    private var frameSlot = 0

    private var cameraPipelineState: MTLRenderPipelineState
    private var meshPipelineState: MTLRenderPipelineState
    private var solidPipelineState: MTLRenderPipelineState
    private var meshDepthState: MTLDepthStencilState

    private var cameraTextureCache: CVMetalTextureCache?
    private var pendingFrame: ARFrame?
    private var pendingMuJoCoScene: MuJoCoRenderScene?
    private var meshAnchors: [UUID: MeshBuffers] = [:]
    private var sphereMeshBuffers: RenderMeshBuffers
    private var boxMeshBuffers: RenderMeshBuffers
    private var cylinderMeshBuffers: RenderMeshBuffers
    private var mujocoEnvironmentBuffers: RenderMeshBuffers?
    private var mujocoNavigationDebugBuffers: RenderMeshBuffers?
    private var mujocoVisualMeshBuffers: [Int: RenderMeshBuffers] = [:]
    private var retainedFrameResources = Array<FrameResources?>(repeating: nil, count: maxBuffersInFlight)
    private var debugCameraTarget = SIMD3<Float>(0, 0, 0.25)
    private var debugCameraDistance: Float = 4.64
    private var debugCameraYaw: Float = -0.85
    private var debugCameraPitch: Float = 0.41
    private var debugCameraMode: DebugCameraMode = .orbit
    private var thirdPersonDistance: Float = 2.7
    private var cameraBackgroundEnabled = true
    private var arMeshWireframeEnabled = true
    private var heightFieldEnabled = true

    init?(metalKitView: MTKView) {
#if targetEnvironment(simulator)
        return nil
#else
        guard let device = metalKitView.device else {
            return nil
        }
        guard device.supportsFamily(.metal4) else {
            print("Metal 4 is not supported on this device")
            return nil
        }
        guard device.supportsShaderBarycentricCoordinates else {
            print("Shader barycentric coordinates are not supported on this device")
            return nil
        }

        self.device = device
        guard let queue = device.makeMTL4CommandQueue(),
              let reusableCommandBuffer = device.makeCommandBuffer() else {
            return nil
        }
        self.commandQueue = queue
        self.commandBuffer = reusableCommandBuffer
        self.commandAllocators = (0..<maxBuffersInFlight).compactMap { _ in device.makeCommandAllocator() }
        guard commandAllocators.count == maxBuffersInFlight else {
            return nil
        }

        let vertexArgumentTableDescriptor = MTL4ArgumentTableDescriptor()
        vertexArgumentTableDescriptor.label = "Vertex Arguments"
        vertexArgumentTableDescriptor.maxBufferBindCount = 3
        guard let vertexArgumentTable = try? device.makeArgumentTable(descriptor: vertexArgumentTableDescriptor) else {
            return nil
        }
        self.vertexArgumentTable = vertexArgumentTable

        let fragmentArgumentTableDescriptor = MTL4ArgumentTableDescriptor()
        fragmentArgumentTableDescriptor.label = "Fragment Arguments"
        fragmentArgumentTableDescriptor.maxBufferBindCount = 1
        fragmentArgumentTableDescriptor.maxTextureBindCount = 2
        guard let fragmentArgumentTable = try? device.makeArgumentTable(descriptor: fragmentArgumentTableDescriptor) else {
            return nil
        }
        self.fragmentArgumentTable = fragmentArgumentTable

        guard let endFrameEvent = device.makeSharedEvent() else {
            return nil
        }
        self.endFrameEvent = endFrameEvent
        self.endFrameEvent.signaledValue = UInt64(frameIndex - 1)

        let uniformBufferSize = perFrameUniformsSize * maxBuffersInFlight
        guard let uniformBuffer = device.makeBuffer(length: uniformBufferSize, options: .storageModeShared) else {
            return nil
        }
        uniformBuffer.label = "AR Frame Uniforms"
        self.uniformBuffer = uniformBuffer

        metalKitView.colorPixelFormat = .bgra8Unorm
        metalKitView.depthStencilPixelFormat = .depth32Float
        metalKitView.sampleCount = 1
        metalKitView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

        do {
            self.cameraPipelineState = try Renderer.makeCameraPipelineState(device: device, view: metalKitView)
            self.meshPipelineState = try Renderer.makeMeshPipelineState(device: device, view: metalKitView)
            self.solidPipelineState = try Renderer.makeSolidPipelineState(device: device, view: metalKitView)
        } catch {
            print("Unable to create render pipeline state: \(error)")
            return nil
        }

        guard let sphereMeshBuffers = RenderMeshBuffers(device: device, mesh: Renderer.makeUnitSphereMesh()) else {
            return nil
        }
        self.sphereMeshBuffers = sphereMeshBuffers
        guard let boxMeshBuffers = RenderMeshBuffers(device: device, mesh: Renderer.makeUnitBoxMesh()) else {
            return nil
        }
        self.boxMeshBuffers = boxMeshBuffers
        guard let cylinderMeshBuffers = RenderMeshBuffers(device: device, mesh: Renderer.makeUnitCylinderMesh()) else {
            return nil
        }
        self.cylinderMeshBuffers = cylinderMeshBuffers

        let depthDescriptor = MTLDepthStencilDescriptor()
        depthDescriptor.depthCompareFunction = .lessEqual
        depthDescriptor.isDepthWriteEnabled = true
        guard let meshDepthState = device.makeDepthStencilState(descriptor: depthDescriptor) else {
            return nil
        }
        self.meshDepthState = meshDepthState

        let transientResidencySets = (0..<maxBuffersInFlight).compactMap { index -> MTLResidencySet? in
            let descriptor = MTLResidencySetDescriptor()
            descriptor.label = "AR Transient Residency \(index)"
            descriptor.initialCapacity = 512
            return try? device.makeResidencySet(descriptor: descriptor)
        }
        guard transientResidencySets.count == maxBuffersInFlight else {
            return nil
        }
        self.transientResidencySets = transientResidencySets

        super.init()

        var textureCache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(nil, nil, device, nil, &textureCache)
        cameraTextureCache = textureCache
#endif
    }

    func updateFrame(_ frame: ARFrame) {
        pendingFrame = frame
    }

    func updateMuJoCoScene(_ scene: MuJoCoRenderScene) {
        pendingMuJoCoScene = scene
    }

    func setCameraBackgroundEnabled(_ enabled: Bool) {
        cameraBackgroundEnabled = enabled
    }

    func setARMeshWireframeEnabled(_ enabled: Bool) {
        arMeshWireframeEnabled = enabled
    }

    func setHeightFieldEnabled(_ enabled: Bool) {
        heightFieldEnabled = enabled
    }

    func setDebugCameraMode(_ mode: DebugCameraMode) {
        debugCameraMode = mode
    }

    func orbitDebugCamera(delta: CGPoint) {
        let sensitivity: Float = 0.006
        debugCameraYaw -= Float(delta.x) * sensitivity
        debugCameraPitch += Float(delta.y) * sensitivity
        debugCameraPitch = min(max(debugCameraPitch, -1.25), 1.25)
    }

    func panDebugCamera(delta: CGPoint) {
        let basis = debugCameraBasis()
        let scale = debugCameraDistance * 0.0012
        debugCameraTarget += (-basis.right * Float(delta.x) + basis.up * Float(delta.y)) * scale
    }

    func zoomDebugCamera(scale: CGFloat) {
        let safeScale = max(Float(scale), 0.05)
        switch debugCameraMode {
        case .orbit:
            debugCameraDistance = min(max(debugCameraDistance / safeScale, 0.6), 40)
        case .thirdPerson:
            thirdPersonDistance = min(max(thirdPersonDistance / safeScale, 1.2), 8)
        }
    }

    func resetDebugCamera() {
        debugCameraTarget = SIMD3<Float>(0, 0, 0.25)
        debugCameraDistance = 4.64
        debugCameraYaw = -0.85
        debugCameraPitch = 0.41
        thirdPersonDistance = 2.7
    }

    func debugGroundPoint(for point: CGPoint, in view: MTKView, scene: MuJoCoRenderScene) -> SIMD3<Float>? {
        let matrices = debugCameraMatrices(for: view.drawableSize, scene: scene)
        let drawableWidth = max(Float(view.drawableSize.width), 1)
        let drawableHeight = max(Float(view.drawableSize.height), 1)
        let scale = Float(view.contentScaleFactor)
        let x = (Float(point.x) * scale / drawableWidth) * 2 - 1
        let y = 1 - (Float(point.y) * scale / drawableHeight) * 2
        let inverseViewProjection = simd_inverse(matrices.projection * matrices.view)
        let near4 = inverseViewProjection * SIMD4<Float>(x, y, 0, 1)
        let far4 = inverseViewProjection * SIMD4<Float>(x, y, 1, 1)
        guard abs(near4.w) > 0.000001, abs(far4.w) > 0.000001 else {
            return nil
        }
        let near = SIMD3<Float>(near4.x, near4.y, near4.z) / near4.w
        let far = SIMD3<Float>(far4.x, far4.y, far4.z) / far4.w
        let direction = far - near
        guard abs(direction.z) > 0.000001 else {
            return nil
        }
        let t = -near.z / direction.z
        guard t >= 0 else {
            return nil
        }
        return near + direction * t
    }

    @available(*, deprecated, message: "Temporary collision probe shooter support; no UI is wired.")
    func debugLaunchRay(scene: MuJoCoRenderScene?) -> (origin: SIMD3<Float>, direction: SIMD3<Float>) {
        let eye: SIMD3<Float>
        let target: SIMD3<Float>
        switch debugCameraMode {
        case .orbit:
            eye = debugCameraEye()
            target = debugCameraTarget
        case .thirdPerson:
            if let robotPose = scene?.robotPose {
                let heading = SIMD3<Float>(cosf(robotPose.yaw), sinf(robotPose.yaw), 0)
                target = robotPose.position + heading * 0.45 + SIMD3<Float>(0, 0, 0.32)
                eye = robotPose.position - heading * thirdPersonDistance + SIMD3<Float>(0, 0, max(1.05, thirdPersonDistance * 0.42))
            } else {
                eye = debugCameraEye()
                target = debugCameraTarget
            }
        }

        return (eye, simd_normalize(target - eye))
    }

    func updateMesh(for anchor: ARMeshAnchor) {
        guard let mesh = MeshBuffers(device: device, meshAnchor: anchor) else {
            return
        }
        meshAnchors[anchor.identifier] = mesh
    }

    func removeMesh(for anchor: ARMeshAnchor) {
        meshAnchors.removeValue(forKey: anchor.identifier)
    }

    func draw(in view: MTKView) {
#if !targetEnvironment(simulator)
        let frame = pendingFrame
        pendingFrame = nil
        let mujocoScene = pendingMuJoCoScene

        guard frame != nil || mujocoScene != nil else {
            return
        }

        guard let drawable = view.currentDrawable,
              let renderPassDescriptor = view.currentMTL4RenderPassDescriptor else {
            return
        }

        frameSlot = frameIndex % maxBuffersInFlight
        let previousValueToWaitFor = frameIndex - maxBuffersInFlight
        endFrameEvent.wait(untilSignaledValue: UInt64(previousValueToWaitFor), timeoutMS: 10)
        retainedFrameResources[frameSlot] = nil

        let commandAllocator = commandAllocators[frameSlot]
        commandAllocator.reset()
        commandBuffer.beginCommandBuffer(allocator: commandAllocator)

        var cameraTextures: CameraTextures?
        var arViewMatrix: simd_float4x4?
        var arProjectionMatrix: simd_float4x4?

        if let frame {
            let orientation = currentInterfaceOrientation(for: view)
            if cameraBackgroundEnabled {
                cameraTextures = makeCameraTextures(from: frame.capturedImage)
            }
            writeFrameUniforms(for: frame, orientation: orientation, viewportSize: view.drawableSize)
            arViewMatrix = frame.camera.viewMatrix(for: orientation)
            arProjectionMatrix = frame.camera.projectionMatrix(
                for: orientation,
                viewportSize: view.drawableSize,
                zNear: 0.001,
                zFar: 1000
            )
        } else {
            writeDefaultFrameUniforms()
        }

        updateMuJoCoEnvironmentBuffersIfNeeded(mujocoScene)
        refreshResidencySet(cameraTextures: cameraTextures)
        commandBuffer.useResidencySet(transientResidencySets[frameSlot])
        commandBuffer.useResidencySet((view.layer as! CAMetalLayer).residencySet)

        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            commandBuffer.endCommandBuffer()
            return
        }
        renderEncoder.label = "AR Camera + Mesh Wire Encoder"

        let frameUniformsAddress = uniformBuffer.gpuAddress + UInt64(frameUniformsOffset)
        vertexArgumentTable.setAddress(frameUniformsAddress, index: BufferIndex.frameUniforms.rawValue)
        fragmentArgumentTable.setAddress(frameUniformsAddress, index: BufferIndex.frameUniforms.rawValue)

        if let cameraTextures {
            fragmentArgumentTable.setTexture(cameraTextures.luma.texture.gpuResourceID, index: TextureIndex.cameraY.rawValue)
            fragmentArgumentTable.setTexture(cameraTextures.chroma.texture.gpuResourceID, index: TextureIndex.cameraCbCr.rawValue)

            renderEncoder.pushDebugGroup("Camera Texture")
            renderEncoder.setRenderPipelineState(cameraPipelineState)
            renderEncoder.setArgumentTable(vertexArgumentTable, stages: .vertex)
            renderEncoder.setArgumentTable(fragmentArgumentTable, stages: .fragment)
            renderEncoder.drawPrimitives(primitiveType: .triangle, vertexStart: 0, vertexCount: 6)
            renderEncoder.popDebugGroup()
        }

        let debugCamera = debugCameraMatrices(for: view.drawableSize, scene: mujocoScene)
        let sceneViewMatrix = arViewMatrix ?? debugCamera.view
        let sceneProjectionMatrix = arProjectionMatrix ?? debugCamera.projection
        var nextMeshUniformIndex = 0

        if let mujocoScene {
            nextMeshUniformIndex = drawMuJoCoScene(
                mujocoScene,
                renderEncoder: renderEncoder,
                viewMatrix: sceneViewMatrix,
                projectionMatrix: sceneProjectionMatrix,
                firstUniformIndex: nextMeshUniformIndex
            )
        }

        guard let frame else {
            renderEncoder.endEncoding()
            commandBuffer.endCommandBuffer()

            retainedFrameResources[frameSlot] = nil
            commandQueue.waitForDrawable(drawable)
            commandQueue.commit([commandBuffer])
            commandQueue.signalDrawable(drawable)
            commandQueue.signalEvent(endFrameEvent, value: UInt64(frameIndex))
            frameIndex += 1
            drawable.present()
            return
        }

        guard arMeshWireframeEnabled else {
            renderEncoder.endEncoding()
            commandBuffer.endCommandBuffer()

            if let cameraTextures {
                retainedFrameResources[frameSlot] = FrameResources(frame: frame, cameraTextures: cameraTextures)
            } else {
                retainedFrameResources[frameSlot] = FrameResources(frame: frame, cameraTextures: nil)
            }

            commandQueue.waitForDrawable(drawable)
            commandQueue.commit([commandBuffer])
            commandQueue.signalDrawable(drawable)
            commandQueue.signalEvent(endFrameEvent, value: UInt64(frameIndex))
            frameIndex += 1
            drawable.present()
            return
        }

        renderEncoder.pushDebugGroup("AR Mesh Wireframe")
        renderEncoder.setRenderPipelineState(meshPipelineState)
        renderEncoder.setDepthStencilState(meshDepthState)
        renderEncoder.setCullMode(.back)
        renderEncoder.setFrontFacing(.counterClockwise)
        renderEncoder.setArgumentTable(vertexArgumentTable, stages: .vertex)
        renderEncoder.setArgumentTable(fragmentArgumentTable, stages: .fragment)
        fragmentArgumentTable.setAddress(frameUniformsAddress, index: BufferIndex.frameUniforms.rawValue)

        for (index, mesh) in meshAnchors.values.prefix(maxMeshAnchorsPerFrame).enumerated() {
            let uniformIndex = nextMeshUniformIndex + index
            guard uniformIndex < maxMeshAnchorsPerFrame else {
                break
            }
            writeMeshUniforms(
                modelMatrix: mesh.transform,
                viewMatrix: sceneViewMatrix,
                projectionMatrix: sceneProjectionMatrix,
                color: SIMD4<Float>(0.05, 0.95, 1.0, 0.9),
                at: uniformIndex
            )

            vertexArgumentTable.setAddress(uniformBuffer.gpuAddress + UInt64(meshUniformsOffset(at: uniformIndex)), index: BufferIndex.meshUniforms.rawValue)
            vertexArgumentTable.setAddress(mesh.vertexBuffer.gpuAddress, index: BufferIndex.meshPositions.rawValue)

            renderEncoder.drawIndexedPrimitives(
                primitiveType: .triangle,
                indexCount: mesh.indexCount,
                indexType: .uint32,
                indexBuffer: mesh.indexBuffer.gpuAddress,
                indexBufferLength: mesh.indexBuffer.length
            )
        }
        renderEncoder.popDebugGroup()

        renderEncoder.endEncoding()
        commandBuffer.endCommandBuffer()

        if let cameraTextures {
            retainedFrameResources[frameSlot] = FrameResources(frame: frame, cameraTextures: cameraTextures)
        } else {
            retainedFrameResources[frameSlot] = FrameResources(frame: frame, cameraTextures: nil)
        }

        commandQueue.waitForDrawable(drawable)
        commandQueue.commit([commandBuffer])
        commandQueue.signalDrawable(drawable)
        commandQueue.signalEvent(endFrameEvent, value: UInt64(frameIndex))
        frameIndex += 1
        drawable.present()
#endif
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
    }

#if !targetEnvironment(simulator)
    private static func makeCameraPipelineState(device: MTLDevice, view: MTKView) throws -> MTLRenderPipelineState {
        let pipelineDescriptor = MTL4RenderPipelineDescriptor()
        pipelineDescriptor.label = "Camera Texture Pipeline"
        pipelineDescriptor.rasterSampleCount = view.sampleCount
        pipelineDescriptor.vertexFunctionDescriptor = functionDescriptor(device: device, name: "cameraVertexShader")
        pipelineDescriptor.fragmentFunctionDescriptor = functionDescriptor(device: device, name: "cameraFragmentShader")
        pipelineDescriptor.inputPrimitiveTopology = .triangle
        pipelineDescriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat
        return try makeCompiler(device: device).makeRenderPipelineState(descriptor: pipelineDescriptor)
    }

    private static func makeMeshPipelineState(device: MTLDevice, view: MTKView) throws -> MTLRenderPipelineState {
        let pipelineDescriptor = MTL4RenderPipelineDescriptor()
        pipelineDescriptor.label = "AR Mesh Barycentric Wire Pipeline"
        pipelineDescriptor.rasterSampleCount = view.sampleCount
        pipelineDescriptor.vertexFunctionDescriptor = functionDescriptor(device: device, name: "meshVertexShader")
        pipelineDescriptor.fragmentFunctionDescriptor = functionDescriptor(device: device, name: "meshWireFragmentShader")
        pipelineDescriptor.inputPrimitiveTopology = .triangle
        pipelineDescriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat
        pipelineDescriptor.colorAttachments[0].blendingState = .enabled
        pipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        pipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        pipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
        pipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        pipelineDescriptor.colorAttachments[0].rgbBlendOperation = .add
        pipelineDescriptor.colorAttachments[0].alphaBlendOperation = .add
        return try makeCompiler(device: device).makeRenderPipelineState(descriptor: pipelineDescriptor)
    }

    private static func makeSolidPipelineState(device: MTLDevice, view: MTKView) throws -> MTLRenderPipelineState {
        let pipelineDescriptor = MTL4RenderPipelineDescriptor()
        pipelineDescriptor.label = "MuJoCo Solid Mesh Pipeline"
        pipelineDescriptor.rasterSampleCount = view.sampleCount
        pipelineDescriptor.vertexFunctionDescriptor = functionDescriptor(device: device, name: "solidVertexShader")
        pipelineDescriptor.fragmentFunctionDescriptor = functionDescriptor(device: device, name: "solidFragmentShader")
        pipelineDescriptor.inputPrimitiveTopology = .triangle
        pipelineDescriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat
        pipelineDescriptor.colorAttachments[0].blendingState = .enabled
        pipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        pipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        pipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
        pipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        pipelineDescriptor.colorAttachments[0].rgbBlendOperation = .add
        pipelineDescriptor.colorAttachments[0].alphaBlendOperation = .add
        return try makeCompiler(device: device).makeRenderPipelineState(descriptor: pipelineDescriptor)
    }

    private static func functionDescriptor(device: MTLDevice, name: String) -> MTL4LibraryFunctionDescriptor {
        let descriptor = MTL4LibraryFunctionDescriptor()
        descriptor.library = device.makeDefaultLibrary()
        descriptor.name = name
        return descriptor
    }

    private static func makeCompiler(device: MTLDevice) throws -> MTL4Compiler {
        try device.makeCompiler(descriptor: MTL4CompilerDescriptor())
    }

    private static func makeUnitSphereMesh() -> RenderMeshDescriptor {
        let latitudeSegments = 12
        let longitudeSegments = 18

        var vertices: [SIMD3<Float>] = []
        var normals: [SIMD3<Float>] = []
        var indices: [UInt32] = []

        for lat in 0...latitudeSegments {
            let theta = Float(lat) / Float(latitudeSegments) * .pi
            let z = cosf(theta)
            let ringRadius = sinf(theta)
            for lon in 0...longitudeSegments {
                let phi = Float(lon) / Float(longitudeSegments) * 2 * .pi
                let normal = SIMD3<Float>(ringRadius * cosf(phi), ringRadius * sinf(phi), z)
                vertices.append(normal)
                normals.append(normalize(normal))
            }
        }

        let rowStride = longitudeSegments + 1
        for lat in 0..<latitudeSegments {
            for lon in 0..<longitudeSegments {
                let a = UInt32(lat * rowStride + lon)
                let b = UInt32((lat + 1) * rowStride + lon)
                let c = UInt32(lat * rowStride + lon + 1)
                let d = UInt32((lat + 1) * rowStride + lon + 1)
                indices.append(contentsOf: [a, b, c, c, b, d])
            }
        }

        return RenderMeshDescriptor(
            vertices: vertices,
            normals: normals,
            indices: indices,
            color: SIMD4<Float>(1, 1, 1, 1),
            revision: -1
        )
    }

    private static func makeUnitBoxMesh() -> RenderMeshDescriptor {
        let faces: [(normal: SIMD3<Float>, corners: [SIMD3<Float>])] = [
            (SIMD3<Float>(0, 0, 1), [
                SIMD3<Float>(-1, -1, 1), SIMD3<Float>(1, -1, 1),
                SIMD3<Float>(-1, 1, 1), SIMD3<Float>(1, 1, 1),
            ]),
            (SIMD3<Float>(0, 0, -1), [
                SIMD3<Float>(1, -1, -1), SIMD3<Float>(-1, -1, -1),
                SIMD3<Float>(1, 1, -1), SIMD3<Float>(-1, 1, -1),
            ]),
            (SIMD3<Float>(1, 0, 0), [
                SIMD3<Float>(1, -1, 1), SIMD3<Float>(1, -1, -1),
                SIMD3<Float>(1, 1, 1), SIMD3<Float>(1, 1, -1),
            ]),
            (SIMD3<Float>(-1, 0, 0), [
                SIMD3<Float>(-1, -1, -1), SIMD3<Float>(-1, -1, 1),
                SIMD3<Float>(-1, 1, -1), SIMD3<Float>(-1, 1, 1),
            ]),
            (SIMD3<Float>(0, 1, 0), [
                SIMD3<Float>(-1, 1, 1), SIMD3<Float>(1, 1, 1),
                SIMD3<Float>(-1, 1, -1), SIMD3<Float>(1, 1, -1),
            ]),
            (SIMD3<Float>(0, -1, 0), [
                SIMD3<Float>(-1, -1, -1), SIMD3<Float>(1, -1, -1),
                SIMD3<Float>(-1, -1, 1), SIMD3<Float>(1, -1, 1),
            ]),
        ]

        var vertices: [SIMD3<Float>] = []
        var normals: [SIMD3<Float>] = []
        var indices: [UInt32] = []
        for face in faces {
            let base = UInt32(vertices.count)
            vertices.append(contentsOf: face.corners)
            normals.append(contentsOf: Array(repeating: face.normal, count: 4))
            indices.append(contentsOf: [base, base + 1, base + 2, base + 2, base + 1, base + 3])
        }

        return RenderMeshDescriptor(
            vertices: vertices,
            normals: normals,
            indices: indices,
            color: SIMD4<Float>(1, 1, 1, 1),
            revision: -2
        )
    }

    private static func makeUnitCylinderMesh() -> RenderMeshDescriptor {
        let segments = 24
        var vertices: [SIMD3<Float>] = []
        var normals: [SIMD3<Float>] = []
        var indices: [UInt32] = []

        for i in 0...segments {
            let angle = Float(i) / Float(segments) * 2 * .pi
            let normal = SIMD3<Float>(cosf(angle), sinf(angle), 0)
            vertices.append(SIMD3<Float>(normal.x, normal.y, -1))
            normals.append(normal)
            vertices.append(SIMD3<Float>(normal.x, normal.y, 1))
            normals.append(normal)
        }
        for i in 0..<segments {
            let base = UInt32(i * 2)
            indices.append(contentsOf: [base, base + 1, base + 2, base + 2, base + 1, base + 3])
        }

        let bottomCenter = UInt32(vertices.count)
        vertices.append(SIMD3<Float>(0, 0, -1))
        normals.append(SIMD3<Float>(0, 0, -1))
        let topCenter = UInt32(vertices.count)
        vertices.append(SIMD3<Float>(0, 0, 1))
        normals.append(SIMD3<Float>(0, 0, 1))

        for i in 0...segments {
            let angle = Float(i) / Float(segments) * 2 * .pi
            vertices.append(SIMD3<Float>(cosf(angle), sinf(angle), -1))
            normals.append(SIMD3<Float>(0, 0, -1))
            vertices.append(SIMD3<Float>(cosf(angle), sinf(angle), 1))
            normals.append(SIMD3<Float>(0, 0, 1))
        }

        let capStart = topCenter + 1
        for i in 0..<segments {
            let a = capStart + UInt32(i * 2)
            let b = capStart + UInt32((i + 1) * 2)
            indices.append(contentsOf: [bottomCenter, b, a])

            let ta = capStart + UInt32(i * 2 + 1)
            let tb = capStart + UInt32((i + 1) * 2 + 1)
            indices.append(contentsOf: [topCenter, ta, tb])
        }

        return RenderMeshDescriptor(
            vertices: vertices,
            normals: normals,
            indices: indices,
            color: SIMD4<Float>(1, 1, 1, 1),
            revision: -3
        )
    }
#endif

    private var frameUniformsOffset: Int {
        frameSlot * perFrameUniformsSize
    }

    private func meshUniformsOffset(at index: Int) -> Int {
        frameUniformsOffset + alignedFrameUniformsSize + alignedMeshUniformsSize * index
    }

    private func writeFrameUniforms(for frame: ARFrame, orientation: UIInterfaceOrientation, viewportSize: CGSize) {
        let transform = frame.displayTransform(for: orientation, viewportSize: viewportSize).inverted()
        let viewToImageTransform = simd_float3x3(
            SIMD3<Float>(Float(transform.a), Float(transform.b), 0),
            SIMD3<Float>(Float(transform.c), Float(transform.d), 0),
            SIMD3<Float>(Float(transform.tx), Float(transform.ty), 1)
        )

        let pointer = uniformBuffer.contents()
            .advanced(by: frameUniformsOffset)
            .bindMemory(to: FrameUniforms.self, capacity: 1)
        pointer.pointee.viewToImageTransform = viewToImageTransform
        pointer.pointee.wireColor = SIMD4<Float>(0.05, 0.95, 1.0, 0.9)
        pointer.pointee.wireWidth = 1.35
        pointer.pointee.wireSoftness = 1.0
    }

    private func writeDefaultFrameUniforms() {
        let pointer = uniformBuffer.contents()
            .advanced(by: frameUniformsOffset)
            .bindMemory(to: FrameUniforms.self, capacity: 1)
        pointer.pointee.viewToImageTransform = matrix_identity_float3x3
        pointer.pointee.wireColor = SIMD4<Float>(0.05, 0.95, 1.0, 0.9)
        pointer.pointee.wireWidth = 1.35
        pointer.pointee.wireSoftness = 1.0
    }

    private func writeFrameWireUniforms(
        color: SIMD4<Float>,
        wireWidth: Float,
        wireSoftness: Float,
        at index: Int
    ) {
        let pointer = uniformBuffer.contents()
            .advanced(by: meshUniformsOffset(at: index))
            .bindMemory(to: FrameUniforms.self, capacity: 1)
        pointer.pointee.viewToImageTransform = matrix_identity_float3x3
        pointer.pointee.wireColor = color
        pointer.pointee.wireWidth = wireWidth
        pointer.pointee.wireSoftness = wireSoftness
    }

    private func writeMeshUniforms(
        modelMatrix: simd_float4x4,
        viewMatrix: simd_float4x4,
        projectionMatrix: simd_float4x4,
        color: SIMD4<Float>,
        at index: Int
    ) {
        let pointer = uniformBuffer.contents()
            .advanced(by: meshUniformsOffset(at: index))
            .bindMemory(to: MeshUniforms.self, capacity: 1)
        pointer.pointee.modelMatrix = modelMatrix
        pointer.pointee.viewMatrix = viewMatrix
        pointer.pointee.projectionMatrix = projectionMatrix
        pointer.pointee.color = color
    }

    private func currentInterfaceOrientation(for view: MTKView) -> UIInterfaceOrientation {
        if let orientation = view.window?.windowScene?.effectiveGeometry.interfaceOrientation, orientation != .unknown {
            return orientation
        }
        return .portrait
    }

    private func debugCameraMatrices(
        for drawableSize: CGSize,
        scene: MuJoCoRenderScene?
    ) -> (view: simd_float4x4, projection: simd_float4x4) {
        let aspect = Float(drawableSize.width / max(drawableSize.height, 1))
        let viewMatrix: simd_float4x4
        switch debugCameraMode {
        case .orbit:
            viewMatrix = matrix4x4_lookAt(
                eye: debugCameraEye(),
                target: debugCameraTarget,
                up: SIMD3<Float>(0, 0, 1)
            )
        case .thirdPerson:
            viewMatrix = thirdPersonCameraMatrix(robotPose: scene?.robotPose)
        }
        let projection = matrix_perspective_right_hand(
            fovyRadians: radians_from_degrees(58),
            aspectRatio: aspect,
            nearZ: 0.01,
            farZ: 80
        )
        return (viewMatrix, projection)
    }

    private func thirdPersonCameraMatrix(robotPose: RobotPose?) -> simd_float4x4 {
        guard let robotPose else {
            return matrix4x4_lookAt(
                eye: debugCameraEye(),
                target: debugCameraTarget,
                up: SIMD3<Float>(0, 0, 1)
            )
        }

        let heading = SIMD3<Float>(cosf(robotPose.yaw), sinf(robotPose.yaw), 0)
        let target = robotPose.position + heading * 0.45 + SIMD3<Float>(0, 0, 0.32)
        let eye = robotPose.position - heading * thirdPersonDistance + SIMD3<Float>(0, 0, max(1.05, thirdPersonDistance * 0.42))
        return matrix4x4_lookAt(
            eye: eye,
            target: target,
            up: SIMD3<Float>(0, 0, 1)
        )
    }

    private func debugCameraEye() -> SIMD3<Float> {
        let horizontalDistance = debugCameraDistance * cosf(debugCameraPitch)
        let offset = SIMD3<Float>(
            horizontalDistance * cosf(debugCameraYaw),
            horizontalDistance * sinf(debugCameraYaw),
            debugCameraDistance * sinf(debugCameraPitch)
        )
        return debugCameraTarget + offset
    }

    private func debugCameraBasis() -> (right: SIMD3<Float>, up: SIMD3<Float>) {
        let eye = debugCameraEye()
        let forward = simd_normalize(debugCameraTarget - eye)
        let worldUp = SIMD3<Float>(0, 0, 1)
        let right = simd_normalize(simd_cross(forward, worldUp))
        let up = simd_normalize(simd_cross(right, forward))
        return (right, up)
    }

#if !targetEnvironment(simulator)
    private func updateMuJoCoEnvironmentBuffersIfNeeded(_ scene: MuJoCoRenderScene?) {
        updateMuJoCoNavigationDebugBuffersIfNeeded(scene)
        updateMuJoCoVisualMeshBuffersIfNeeded(scene)

        guard let mesh = scene?.environmentMesh else {
            mujocoEnvironmentBuffers = nil
            return
        }
        if mujocoEnvironmentBuffers?.revision == mesh.revision {
            return
        }
        mujocoEnvironmentBuffers = RenderMeshBuffers(device: device, mesh: mesh)
    }

    private func updateMuJoCoNavigationDebugBuffersIfNeeded(_ scene: MuJoCoRenderScene?) {
        guard let mesh = scene?.navigationDebugMesh else {
            mujocoNavigationDebugBuffers = nil
            return
        }
        if mujocoNavigationDebugBuffers?.revision == mesh.revision {
            return
        }
        mujocoNavigationDebugBuffers = RenderMeshBuffers(device: device, mesh: mesh)
    }

    private func updateMuJoCoVisualMeshBuffersIfNeeded(_ scene: MuJoCoRenderScene?) {
        guard let scene else {
            mujocoVisualMeshBuffers.removeAll()
            return
        }

        let validMeshIDs = Set(scene.visualMeshes.keys)
        mujocoVisualMeshBuffers = mujocoVisualMeshBuffers.filter { validMeshIDs.contains($0.key) }

        for (meshID, mesh) in scene.visualMeshes {
            if mujocoVisualMeshBuffers[meshID]?.revision == mesh.revision {
                continue
            }
            mujocoVisualMeshBuffers[meshID] = RenderMeshBuffers(device: device, mesh: mesh)
        }
    }

    private func drawMuJoCoScene(
        _ scene: MuJoCoRenderScene,
        renderEncoder: MTL4RenderCommandEncoder,
        viewMatrix: simd_float4x4,
        projectionMatrix: simd_float4x4,
        firstUniformIndex: Int
    ) -> Int {
        var uniformIndex = firstUniformIndex

        renderEncoder.pushDebugGroup("MuJoCo Scene")
        renderEncoder.setRenderPipelineState(solidPipelineState)
        renderEncoder.setDepthStencilState(meshDepthState)
        renderEncoder.setCullMode(.back)
        renderEncoder.setFrontFacing(.counterClockwise)
        renderEncoder.setArgumentTable(vertexArgumentTable, stages: .vertex)
        renderEncoder.setArgumentTable(fragmentArgumentTable, stages: .fragment)

        if heightFieldEnabled,
           let environmentMesh = scene.environmentMesh,
           let environmentBuffers = mujocoEnvironmentBuffers,
           uniformIndex < maxMeshAnchorsPerFrame {
            let meshUniformIndex = uniformIndex
            writeMeshUniforms(
                modelMatrix: scene.worldTransform,
                viewMatrix: viewMatrix,
                projectionMatrix: projectionMatrix,
                color: environmentMesh.color,
                at: meshUniformIndex
            )
            drawSolidMesh(environmentBuffers, renderEncoder: renderEncoder, uniformIndex: meshUniformIndex)
            if meshUniformIndex + 1 < maxMeshAnchorsPerFrame {
                let wireUniformIndex = meshUniformIndex + 1
                renderEncoder.setRenderPipelineState(meshPipelineState)
                renderEncoder.setCullMode(.none)
                writeFrameWireUniforms(
                    color: SIMD4<Float>(1.0, 0.72, 0.08, 0.95),
                    wireWidth: 1.8,
                    wireSoftness: 0.85,
                    at: wireUniformIndex
                )
                vertexArgumentTable.setAddress(
                    uniformBuffer.gpuAddress + UInt64(meshUniformsOffset(at: meshUniformIndex)),
                    index: BufferIndex.meshUniforms.rawValue
                )
                vertexArgumentTable.setAddress(
                    environmentBuffers.vertexBuffer.gpuAddress,
                    index: BufferIndex.meshPositions.rawValue
                )
                fragmentArgumentTable.setAddress(
                    uniformBuffer.gpuAddress + UInt64(meshUniformsOffset(at: wireUniformIndex)),
                    index: BufferIndex.frameUniforms.rawValue
                )
                renderEncoder.drawIndexedPrimitives(
                    primitiveType: .triangle,
                    indexCount: environmentBuffers.indexCount,
                    indexType: .uint32,
                    indexBuffer: environmentBuffers.indexBuffer.gpuAddress,
                    indexBufferLength: environmentBuffers.indexBuffer.length
                )
                renderEncoder.setRenderPipelineState(solidPipelineState)
                renderEncoder.setCullMode(.back)
                uniformIndex = wireUniformIndex + 1
            } else {
                uniformIndex = meshUniformIndex + 1
            }
        }

        if let navigationDebugMesh = scene.navigationDebugMesh,
           let navigationDebugBuffers = mujocoNavigationDebugBuffers,
           uniformIndex < maxMeshAnchorsPerFrame {
            renderEncoder.setCullMode(.none)
            writeMeshUniforms(
                modelMatrix: scene.worldTransform,
                viewMatrix: viewMatrix,
                projectionMatrix: projectionMatrix,
                color: navigationDebugMesh.color,
                at: uniformIndex
            )
            drawSolidMesh(navigationDebugBuffers, renderEncoder: renderEncoder, uniformIndex: uniformIndex)
            uniformIndex += 1

            if uniformIndex + 1 < maxMeshAnchorsPerFrame {
                let meshUniformIndex = uniformIndex
                let wireUniformIndex = uniformIndex + 1
                renderEncoder.setRenderPipelineState(meshPipelineState)
                renderEncoder.setDepthStencilState(meshDepthState)

                writeFrameWireUniforms(
                    color: SIMD4<Float>(0.90, 1.0, 0.12, 0.95),
                    wireWidth: 2.4,
                    wireSoftness: 0.8,
                    at: wireUniformIndex
                )
                writeMeshUniforms(
                    modelMatrix: scene.worldTransform,
                    viewMatrix: viewMatrix,
                    projectionMatrix: projectionMatrix,
                    color: SIMD4<Float>(0, 0, 0, 0),
                    at: meshUniformIndex
                )

                vertexArgumentTable.setAddress(
                    uniformBuffer.gpuAddress + UInt64(meshUniformsOffset(at: meshUniformIndex)),
                    index: BufferIndex.meshUniforms.rawValue
                )
                vertexArgumentTable.setAddress(
                    navigationDebugBuffers.vertexBuffer.gpuAddress,
                    index: BufferIndex.meshPositions.rawValue
                )
                fragmentArgumentTable.setAddress(
                    uniformBuffer.gpuAddress + UInt64(meshUniformsOffset(at: wireUniformIndex)),
                    index: BufferIndex.frameUniforms.rawValue
                )

                renderEncoder.drawIndexedPrimitives(
                    primitiveType: .triangle,
                    indexCount: navigationDebugBuffers.indexCount,
                    indexType: .uint32,
                    indexBuffer: navigationDebugBuffers.indexBuffer.gpuAddress,
                    indexBufferLength: navigationDebugBuffers.indexBuffer.length
                )
                uniformIndex = wireUniformIndex + 1

                renderEncoder.setRenderPipelineState(solidPipelineState)
            }

            renderEncoder.setCullMode(.back)
        }

        if !scene.meshInstances.isEmpty {
            renderEncoder.setCullMode(.none)
            for meshInstance in scene.meshInstances {
                guard uniformIndex < maxMeshAnchorsPerFrame,
                      let meshBuffers = mujocoVisualMeshBuffers[meshInstance.meshID] else {
                    break
                }

                writeMeshUniforms(
                    modelMatrix: scene.worldTransform * meshInstance.modelMatrix,
                    viewMatrix: viewMatrix,
                    projectionMatrix: projectionMatrix,
                    color: meshInstance.color,
                    at: uniformIndex
                )
                drawSolidMesh(meshBuffers, renderEncoder: renderEncoder, uniformIndex: uniformIndex)
                uniformIndex += 1
            }
            renderEncoder.setCullMode(.back)
        }

        for sphere in scene.spheres {
            guard uniformIndex < maxMeshAnchorsPerFrame else {
                break
            }

            let modelMatrix = scene.worldTransform
                * matrix4x4_translation(sphere.position)
                * matrix4x4_scale(SIMD3<Float>(repeating: sphere.radius))
            writeMeshUniforms(
                modelMatrix: modelMatrix,
                viewMatrix: viewMatrix,
                projectionMatrix: projectionMatrix,
                color: sphere.color,
                at: uniformIndex
            )
            drawSolidMesh(sphereMeshBuffers, renderEncoder: renderEncoder, uniformIndex: uniformIndex)
            uniformIndex += 1
        }

        for primitive in scene.primitives {
            guard uniformIndex < maxMeshAnchorsPerFrame else {
                break
            }

            let buffers: RenderMeshBuffers
            switch primitive.kind {
            case .sphere:
                buffers = sphereMeshBuffers
            case .box:
                buffers = boxMeshBuffers
            case .cylinder:
                buffers = cylinderMeshBuffers
            }
            writeMeshUniforms(
                modelMatrix: scene.worldTransform * primitive.modelMatrix,
                viewMatrix: viewMatrix,
                projectionMatrix: projectionMatrix,
                color: primitive.color,
                at: uniformIndex
            )
            drawSolidMesh(buffers, renderEncoder: renderEncoder, uniformIndex: uniformIndex)
            uniformIndex += 1
        }

        renderEncoder.popDebugGroup()
        return uniformIndex
    }

    private func drawSolidMesh(
        _ mesh: RenderMeshBuffers,
        renderEncoder: MTL4RenderCommandEncoder,
        uniformIndex: Int
    ) {
        let uniformsAddress = uniformBuffer.gpuAddress + UInt64(meshUniformsOffset(at: uniformIndex))
        vertexArgumentTable.setAddress(uniformsAddress, index: BufferIndex.meshUniforms.rawValue)
        fragmentArgumentTable.setAddress(uniformsAddress, index: BufferIndex.meshUniforms.rawValue)
        vertexArgumentTable.setAddress(mesh.vertexBuffer.gpuAddress, index: BufferIndex.meshPositions.rawValue)
        vertexArgumentTable.setAddress(mesh.normalBuffer.gpuAddress, index: BufferIndex.meshNormals.rawValue)
        renderEncoder.drawIndexedPrimitives(
            primitiveType: .triangle,
            indexCount: mesh.indexCount,
            indexType: .uint32,
            indexBuffer: mesh.indexBuffer.gpuAddress,
            indexBufferLength: mesh.indexBuffer.length
        )
    }
#endif

    private func makeCameraTextures(from pixelBuffer: CVPixelBuffer) -> CameraTextures? {
        guard let luma = makeCameraPlaneTexture(from: pixelBuffer, pixelFormat: .r8Unorm, planeIndex: 0),
              let chroma = makeCameraPlaneTexture(from: pixelBuffer, pixelFormat: .rg8Unorm, planeIndex: 1) else {
            return nil
        }
        return CameraTextures(luma: luma, chroma: chroma)
    }

    private func makeCameraPlaneTexture(
        from pixelBuffer: CVPixelBuffer,
        pixelFormat: MTLPixelFormat,
        planeIndex: Int
    ) -> CameraPlaneTexture? {
        guard let cameraTextureCache else {
            return nil
        }

        let width = CVPixelBufferGetWidthOfPlane(pixelBuffer, planeIndex)
        let height = CVPixelBufferGetHeightOfPlane(pixelBuffer, planeIndex)

        var cvTexture: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(
            nil,
            cameraTextureCache,
            pixelBuffer,
            nil,
            pixelFormat,
            width,
            height,
            planeIndex,
            &cvTexture
        )

        guard status == kCVReturnSuccess,
              let cvTexture,
              let texture = CVMetalTextureGetTexture(cvTexture) else {
            return nil
        }

        return CameraPlaneTexture(cvTexture: cvTexture, texture: texture)
    }

#if !targetEnvironment(simulator)
    private func refreshResidencySet(cameraTextures: CameraTextures?) {
        let transientResidencySet = transientResidencySets[frameSlot]
        transientResidencySet.removeAllAllocations()

        var allocations: [MTLAllocation] = [uniformBuffer]
        if let cameraTextures {
            allocations.append(cameraTextures.luma.texture)
            allocations.append(cameraTextures.chroma.texture)
        }
        allocations.append(sphereMeshBuffers.vertexBuffer)
        allocations.append(sphereMeshBuffers.normalBuffer)
        allocations.append(sphereMeshBuffers.indexBuffer)
        allocations.append(boxMeshBuffers.vertexBuffer)
        allocations.append(boxMeshBuffers.normalBuffer)
        allocations.append(boxMeshBuffers.indexBuffer)
        allocations.append(cylinderMeshBuffers.vertexBuffer)
        allocations.append(cylinderMeshBuffers.normalBuffer)
        allocations.append(cylinderMeshBuffers.indexBuffer)
        if let mujocoEnvironmentBuffers {
            allocations.append(mujocoEnvironmentBuffers.vertexBuffer)
            allocations.append(mujocoEnvironmentBuffers.normalBuffer)
            allocations.append(mujocoEnvironmentBuffers.indexBuffer)
        }
        if let mujocoNavigationDebugBuffers {
            allocations.append(mujocoNavigationDebugBuffers.vertexBuffer)
            allocations.append(mujocoNavigationDebugBuffers.normalBuffer)
            allocations.append(mujocoNavigationDebugBuffers.indexBuffer)
        }
        for meshBuffers in mujocoVisualMeshBuffers.values {
            allocations.append(meshBuffers.vertexBuffer)
            allocations.append(meshBuffers.normalBuffer)
            allocations.append(meshBuffers.indexBuffer)
        }
        for mesh in meshAnchors.values.prefix(maxMeshAnchorsPerFrame) {
            allocations.append(mesh.vertexBuffer)
            allocations.append(mesh.indexBuffer)
        }

        transientResidencySet.addAllocations(allocations)
        transientResidencySet.commit()
    }
#endif
}

private struct MeshBuffers {
    let vertexBuffer: MTLBuffer
    let indexBuffer: MTLBuffer
    let indexCount: Int
    let transform: simd_float4x4

    init?(device: MTLDevice, meshAnchor: ARMeshAnchor) {
        let geometry = meshAnchor.geometry

        let vertices = (0..<geometry.vertices.count).map { geometry.vertex(at: $0) }
        guard let vertexBuffer = device.makeBuffer(
            bytes: vertices,
            length: vertices.count * MemoryLayout<SIMD3<Float>>.stride,
            options: .storageModeShared
        ) else {
            return nil
        }
        vertexBuffer.label = "AR Mesh Positions"

        let indices = geometry.faces.triangleIndices()
        guard !indices.isEmpty,
              let indexBuffer = device.makeBuffer(
                bytes: indices,
                length: indices.count * MemoryLayout<UInt32>.stride,
                options: .storageModeShared
              ) else {
            return nil
        }
        indexBuffer.label = "AR Mesh Indices"

        self.vertexBuffer = vertexBuffer
        self.indexBuffer = indexBuffer
        self.indexCount = indices.count
        self.transform = meshAnchor.transform
    }
}

private struct RenderMeshBuffers {
    let vertexBuffer: MTLBuffer
    let normalBuffer: MTLBuffer
    let indexBuffer: MTLBuffer
    let indexCount: Int
    let revision: Int

    init?(device: MTLDevice, mesh: RenderMeshDescriptor) {
        guard !mesh.vertices.isEmpty,
              mesh.vertices.count == mesh.normals.count,
              !mesh.indices.isEmpty,
              let vertexBuffer = device.makeBuffer(
                bytes: mesh.vertices,
                length: mesh.vertices.count * MemoryLayout<SIMD3<Float>>.stride,
                options: .storageModeShared
              ),
              let normalBuffer = device.makeBuffer(
                bytes: mesh.normals,
                length: mesh.normals.count * MemoryLayout<SIMD3<Float>>.stride,
                options: .storageModeShared
              ),
              let indexBuffer = device.makeBuffer(
                bytes: mesh.indices,
                length: mesh.indices.count * MemoryLayout<UInt32>.stride,
                options: .storageModeShared
              ) else {
            return nil
        }

        vertexBuffer.label = "Render Mesh Positions"
        normalBuffer.label = "Render Mesh Normals"
        indexBuffer.label = "Render Mesh Indices"

        self.vertexBuffer = vertexBuffer
        self.normalBuffer = normalBuffer
        self.indexBuffer = indexBuffer
        self.indexCount = mesh.indices.count
        self.revision = mesh.revision
    }
}

private struct CameraPlaneTexture {
    let cvTexture: CVMetalTexture
    let texture: MTLTexture
}

private struct CameraTextures {
    let luma: CameraPlaneTexture
    let chroma: CameraPlaneTexture
}

private struct FrameResources {
    let frame: ARFrame
    let cameraTextures: CameraTextures?
}

private extension ARGeometrySource {
    func vector3(at index: Int) -> SIMD3<Float> {
        let pointer = buffer.contents().advanced(by: offset + stride * index)
        let floats = pointer.assumingMemoryBound(to: Float.self)
        return SIMD3<Float>(floats[0], floats[1], floats[2])
    }
}

private extension ARMeshGeometry {
    func vertex(at index: Int) -> SIMD3<Float> {
        vertices.vector3(at: index)
    }
}

private extension ARGeometryElement {
    func triangleIndices() -> [UInt32] {
        let indexCount = count * indexCountPerPrimitive
        let source = buffer.contents()

        if bytesPerIndex == MemoryLayout<UInt32>.stride {
            let indices = source.assumingMemoryBound(to: UInt32.self)
            return (0..<indexCount).map { indices[$0] }
        }

        if bytesPerIndex == MemoryLayout<UInt16>.stride {
            let indices = source.assumingMemoryBound(to: UInt16.self)
            return (0..<indexCount).map { UInt32(indices[$0]) }
        }

        return []
    }
}

nonisolated private func alignTo256(_ value: Int) -> Int {
    (value + 0xFF) & -0x100
}

nonisolated func matrix4x4_translation(_ translation: SIMD3<Float>) -> matrix_float4x4 {
    matrix_float4x4(columns: (
        SIMD4<Float>(1, 0, 0, 0),
        SIMD4<Float>(0, 1, 0, 0),
        SIMD4<Float>(0, 0, 1, 0),
        SIMD4<Float>(translation.x, translation.y, translation.z, 1)
    ))
}

nonisolated func matrix4x4_scale(_ scale: SIMD3<Float>) -> matrix_float4x4 {
    matrix_float4x4(columns: (
        SIMD4<Float>(scale.x, 0, 0, 0),
        SIMD4<Float>(0, scale.y, 0, 0),
        SIMD4<Float>(0, 0, scale.z, 0),
        SIMD4<Float>(0, 0, 0, 1)
    ))
}

nonisolated func matrix4x4_lookAt(eye: SIMD3<Float>, target: SIMD3<Float>, up: SIMD3<Float>) -> matrix_float4x4 {
    let zAxis = normalize(eye - target)
    let xAxis = normalize(cross(up, zAxis))
    let yAxis = cross(zAxis, xAxis)

    return matrix_float4x4(columns: (
        SIMD4<Float>(xAxis.x, yAxis.x, zAxis.x, 0),
        SIMD4<Float>(xAxis.y, yAxis.y, zAxis.y, 0),
        SIMD4<Float>(xAxis.z, yAxis.z, zAxis.z, 0),
        SIMD4<Float>(-dot(xAxis, eye), -dot(yAxis, eye), -dot(zAxis, eye), 1)
    ))
}

nonisolated func matrix_perspective_right_hand(
    fovyRadians fovy: Float,
    aspectRatio: Float,
    nearZ: Float,
    farZ: Float
) -> matrix_float4x4 {
    let ys = 1 / tanf(fovy * 0.5)
    let xs = ys / aspectRatio
    let zs = farZ / (nearZ - farZ)
    return matrix_float4x4(columns: (
        SIMD4<Float>(xs, 0, 0, 0),
        SIMD4<Float>(0, ys, 0, 0),
        SIMD4<Float>(0, 0, zs, -1),
        SIMD4<Float>(0, 0, zs * nearZ, 0)
    ))
}

nonisolated func matrix4x4_arFromMuJoCo() -> matrix_float4x4 {
    matrix_float4x4(columns: (
        SIMD4<Float>(1, 0, 0, 0),
        SIMD4<Float>(0, 0, -1, 0),
        SIMD4<Float>(0, 1, 0, 0),
        SIMD4<Float>(0, 0, 0, 1)
    ))
}

nonisolated func matrix4x4_muJoCoPosition(fromARPosition position: SIMD3<Float>) -> SIMD3<Float> {
    SIMD3<Float>(position.x, -position.z, position.y)
}

nonisolated func matrix4x4_muJoCoDirection(fromARDirection direction: SIMD3<Float>) -> SIMD3<Float> {
    SIMD3<Float>(direction.x, -direction.z, direction.y)
}

nonisolated func radians_from_degrees(_ degrees: Float) -> Float {
    (degrees / 180) * .pi
}
