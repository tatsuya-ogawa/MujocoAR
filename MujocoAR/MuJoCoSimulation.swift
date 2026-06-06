//
//  MuJoCoSimulation.swift
//  MujocoAR
//
//  Created by Tatsuya Ogawa on 2026/05/29.
//

import Foundation
import MuJoCo
import simd

nonisolated struct RenderMeshDescriptor: Sendable {
    let vertices: [SIMD3<Float>]
    let normals: [SIMD3<Float>]
    let indices: [UInt32]
    let color: SIMD4<Float>
    let revision: Int
}

nonisolated struct RenderSphereDescriptor: Sendable {
    let position: SIMD3<Float>
    let radius: Float
    let color: SIMD4<Float>
}

nonisolated enum RenderPrimitiveKind: Sendable {
    case sphere
    case box
    case cylinder
}

nonisolated struct RenderPrimitiveDescriptor: Sendable {
    let kind: RenderPrimitiveKind
    let modelMatrix: simd_float4x4
    let color: SIMD4<Float>
}

nonisolated struct RenderMeshInstanceDescriptor: Sendable {
    let meshID: Int
    let modelMatrix: simd_float4x4
    let color: SIMD4<Float>
}

nonisolated struct NavigationMeshDescriptor: Sendable {
    let vertices: [SIMD3<Float>]
    let indices: [UInt32]
    let revision: Int
}

nonisolated struct EnvironmentMeshChunkDescriptor: Sendable {
    let identifier: UUID
    let vertices: [SIMD3<Float>]
    let indices: [UInt32]
    let floorFaceMask: [Bool]?
    let revision: Int
}

nonisolated struct RobotPose: Sendable {
    let position: SIMD3<Float>
    let yaw: Float
}

nonisolated struct MuJoCoRenderScene: Sendable {
    let environmentMesh: RenderMeshDescriptor?
    let navigationDebugMesh: RenderMeshDescriptor?
    let visualMeshes: [Int: RenderMeshDescriptor]
    let spheres: [RenderSphereDescriptor]
    let primitives: [RenderPrimitiveDescriptor]
    let meshInstances: [RenderMeshInstanceDescriptor]
    let navigationMesh: NavigationMeshDescriptor?
    let robotPose: RobotPose?
    let worldTransform: simd_float4x4

    static let empty = MuJoCoRenderScene(
        environmentMesh: nil,
        navigationDebugMesh: nil,
        visualMeshes: [:],
        spheres: [],
        primitives: [],
        meshInstances: [],
        navigationMesh: nil,
        robotPose: nil,
        worldTransform: matrix_identity_float4x4
    )
}

actor MuJoCoSimulation {
    enum Environment: Sendable {
        case debugBumpyTerrain
        case debugDynamicTerrain(phase: Double)
        case arMesh(vertices: [SIMD3<Float>], indices: [UInt32], coveragePoints: [SIMD2<Float>] = [])
        case arMeshChunks([EnvironmentMeshChunkDescriptor], coveragePoints: [SIMD2<Float>] = [])
    }

    enum LoadPhase: Sendable {
        case generatingHeightField(HeightFieldProgress)
        case applyingMuJoCo
    }

    struct HeightFieldProgress: Sendable {
        let processedCount: Int
        let totalCount: Int
    }

    struct ARCollisionHFieldBuildPlan: Sendable {
        let requests: [ARCollisionHFieldBuildRequest]
        let initialProgress: HeightFieldProgress
    }

    struct ARCollisionHFieldBuildRequest: Sendable {
        fileprivate let chunk: EnvironmentMeshChunkDescriptor
        fileprivate let coveragePoints: [SIMD2<Float>]
        fileprivate let coverageSignature: Int
        fileprivate let detailEnabled: Bool
        fileprivate let revision: Int
    }

    struct ARCollisionHFieldBuildResult: Sendable {
        fileprivate let identifier: UUID
        fileprivate let chunkRevision: Int
        fileprivate let vertexCount: Int
        fileprivate let indexCount: Int
        fileprivate let coverageSignature: Int
        fileprivate let detailEnabled: Bool
        fileprivate let geometry: ARCollisionGeometry?
    }

    private var model: UnsafeMutablePointer<mjModel>?
    private var data: UnsafeMutablePointer<mjData>?
    private var environmentMesh: RenderMeshDescriptor?
    private var visualMeshes: [Int: RenderMeshDescriptor] = [:]
    private var visualMeshParts: [VisualMeshPart] = []
    private var navigationMesh: NavigationMeshDescriptor?
    private var arCollisionHFieldCache: [UUID: CachedARCollisionHField] = [:]
    private var arCollisionHFieldDetailEnabled = false
    private var environmentRevision = 0
    private var worldTransform = matrix_identity_float4x4
    private var locomotionController: Go1PolicyController?
    private var robotKind: LocomotionRobotKind = .go1
    private var robotScale: Float = 1.0
    private var robotSpawned = true
    private var robotGeomCollisionMasks: [GeomCollisionMask] = []
    private var collisionProbeSlots: [CollisionProbeSlot] = []
    private var collisionProbeActive: [Bool] = []
    private var nextCollisionProbeSlot = 0
    private var terrainBodyID: Int?
    private var renderTerrainPrimitives = false
    private var rootStartPosition = SIMD3<Double>(0, 0, 0.55)
    private var lastStepTime: CFTimeInterval?
    private let fixedStep = 1.0 / 240.0
    private let maxStepsPerFrame = 8
    private static let navigationObstacleMargin: Float = 0.30
    private static let collisionProbeSlotCount = 12
    private static let collisionProbeRadius: Float = 0.045
    private static let collisionProbeSpeed: Float = 3.8
    private static let maxARCollisionVertexCount = 8000
    private static let maxARCollisionIndexCount = 36000

    init() {}

    deinit {
        if let data {
            mj_deleteData(data)
        }
        if let model {
            mj_deleteModel(model)
        }
    }

    func load(
        environment: Environment,
        preservingState: Bool = false,
        progress: ((LoadPhase) -> Void)? = nil
    ) {
        let stateSnapshot = preservingState ? makeStateSnapshot() : nil
        clearModel()

        let sceneDefinition: SceneDefinition
        switch environment {
        case .debugBumpyTerrain:
            arCollisionHFieldCache.removeAll()
            sceneDefinition = Self.makeDebugScene(robot: robotKind, robotScale: robotScale, revision: environmentRevision + 1)
            worldTransform = matrix_identity_float4x4
        case .debugDynamicTerrain(let phase):
            arCollisionHFieldCache.removeAll()
            sceneDefinition = Self.makeDebugScene(
                robot: robotKind,
                robotScale: robotScale,
                revision: environmentRevision + 1,
                variationPhase: phase
            )
            worldTransform = matrix_identity_float4x4
        case .arMesh(let vertices, let indices, let coveragePoints):
            progress?(.generatingHeightField(HeightFieldProgress(processedCount: 0, totalCount: 1)))
            sceneDefinition = makeARScene(
                robot: robotKind,
                robotScale: robotScale,
                vertices: vertices,
                indices: indices,
                coveragePoints: coveragePoints,
                revision: environmentRevision + 1
            )
            progress?(.generatingHeightField(HeightFieldProgress(processedCount: 1, totalCount: 1)))
            worldTransform = matrix4x4_arFromMuJoCo()
        case .arMeshChunks(let chunks, let coveragePoints):
            sceneDefinition = makeARScene(
                robot: robotKind,
                robotScale: robotScale,
                chunks: chunks,
                coveragePoints: coveragePoints,
                revision: environmentRevision + 1,
                progress: progress
            )
            worldTransform = matrix4x4_arFromMuJoCo()
        }

        progress?(.applyingMuJoCo)
        let xmlPath = NSTemporaryDirectory() + "mujoco_ar_scene.xml"
        do {
            try sceneDefinition.xml.write(toFile: xmlPath, atomically: true, encoding: .utf8)
        } catch {
            print("Failed to write MuJoCo XML: \(error)")
            return
        }

        var errorBuffer = [CChar](repeating: 0, count: 2048)
        let loadedModel = mj_loadXML(xmlPath, nil, &errorBuffer, Int32(errorBuffer.count))
        guard let loadedModel else {
            print("Failed to load MuJoCo model: \(String(cString: errorBuffer))")
            return
        }

        guard let loadedData = mj_makeData(loadedModel) else {
            mj_deleteModel(loadedModel)
            print("Failed to allocate MuJoCo data")
            return
        }

        model = loadedModel
        data = loadedData
        loadedModel.pointee.opt.timestep = 0.005
        environmentRevision += 1
        environmentMesh = sceneDefinition.environmentMesh
        rootStartPosition = sceneDefinition.rootStartPosition
        robotSpawned = stateSnapshot?.robotSpawned ?? sceneDefinition.robotInitiallySpawned
        renderTerrainPrimitives = sceneDefinition.renderTerrainPrimitives
        terrainBodyID = Self.objectID(loadedModel, type: mjOBJ_BODY, name: "terrain")
        robotGeomCollisionMasks = Self.makeRobotGeomCollisionMasks(model: loadedModel)
        applyRobotCollisionEnabled(robotSpawned, model: loadedModel)
        collisionProbeSlots = Self.makeCollisionProbeSlots(model: loadedModel)
        collisionProbeActive = Array(repeating: false, count: collisionProbeSlots.count)
        nextCollisionProbeSlot = 0
        let visualAssets = Self.loadRobotVisualAssets(
            robot: robotKind,
            model: loadedModel,
            revision: environmentRevision,
            scale: robotScale
        )
        visualMeshes = visualAssets.meshes
        visualMeshParts = visualAssets.parts
        mj_resetDataKeyframe(loadedModel, loadedData, 0)
        mj_forward(loadedModel, loadedData)
        if let heightAboveTerrain = sceneDefinition.rootHeightAboveTerrain,
           let snappedRootPosition = Self.rootPositionAboveTerrain(
               x: rootStartPosition.x,
               y: rootStartPosition.y,
               heightAboveTerrain: heightAboveTerrain,
               model: loadedModel,
               data: loadedData,
               robot: robotKind
           ) {
            rootStartPosition = snappedRootPosition
        }
        locomotionController = Go1PolicyController(
            model: loadedModel,
            data: loadedData,
            robot: robotKind,
            robotScale: robotScale
        )

        if let stateSnapshot,
           Self.restoreSimulationState(stateSnapshot, model: loadedModel, data: loadedData) {
            rootStartPosition = stateSnapshot.rootStartPosition
            robotSpawned = stateSnapshot.robotSpawned
            if stateSnapshot.collisionProbeActive.count == collisionProbeSlots.count {
                collisionProbeActive = stateSnapshot.collisionProbeActive
            }
            if !collisionProbeSlots.isEmpty {
                nextCollisionProbeSlot = stateSnapshot.nextCollisionProbeSlot % collisionProbeSlots.count
            }
            if let controllerState = stateSnapshot.controllerState {
                locomotionController?.restore(controllerState)
            }
        } else if robotSpawned {
            parkAllCollisionProbes(model: loadedModel, data: loadedData)
            locomotionController?.reset(rootPosition: rootStartPosition)
        } else {
            parkAllCollisionProbes(model: loadedModel, data: loadedData)
        }

        mj_forward(loadedModel, loadedData)
        if let explicitNavigationMesh = sceneDefinition.navigationMesh {
            navigationMesh = explicitNavigationMesh
        } else if sceneDefinition.buildNavigationMeshFromModel {
            navigationMesh = Self.makeNavigationMesh(
                model: loadedModel,
                data: loadedData,
                terrainBodyID: terrainBodyID,
                revision: environmentRevision
            )
        } else if let environmentMesh = sceneDefinition.environmentMesh {
            navigationMesh = NavigationMeshDescriptor(
                vertices: environmentMesh.vertices,
                indices: environmentMesh.indices,
                revision: environmentMesh.revision
            )
        } else {
            navigationMesh = Self.makeNavigationMesh(
                model: loadedModel,
                data: loadedData,
                terrainBodyID: terrainBodyID,
                revision: environmentRevision
            )
        }
        lastStepTime = nil
    }

    func clear() {
        clearModel()
        robotSpawned = false
    }

    func setRobotKind(_ robotKind: LocomotionRobotKind) {
        self.robotKind = robotKind
    }

    func setRobotScale(_ scale: Float) {
        robotScale = min(max(scale, 0.5), 3.0)
    }

    func setARCollisionHFieldDetailEnabled(_ enabled: Bool) {
        guard arCollisionHFieldDetailEnabled != enabled else {
            return
        }
        arCollisionHFieldDetailEnabled = enabled
        arCollisionHFieldCache.removeAll()
    }

    func estimatedARCollisionHFieldProgress(
        chunks: [EnvironmentMeshChunkDescriptor],
        coveragePoints: [SIMD2<Float>]
    ) -> HeightFieldProgress {
        let includedChunks = Self.clippedARCollisionChunks(from: chunks)
        guard !includedChunks.isEmpty else {
            return HeightFieldProgress(processedCount: 0, totalCount: 0)
        }

        let coverageByChunk = Self.coveragePointsByChunk(chunks: includedChunks, coveragePoints: coveragePoints)
        let cachedCount = cachedARCollisionHFieldCount(
            chunks: includedChunks,
            coverageByChunk: coverageByChunk
        )
        return HeightFieldProgress(processedCount: cachedCount, totalCount: includedChunks.count)
    }

    func makeARCollisionHFieldBuildPlan(
        chunks: [EnvironmentMeshChunkDescriptor],
        coveragePoints: [SIMD2<Float>]
    ) -> ARCollisionHFieldBuildPlan {
        let includedChunks = Self.clippedARCollisionChunks(from: chunks)
        guard !includedChunks.isEmpty else {
            return ARCollisionHFieldBuildPlan(
                requests: [],
                initialProgress: HeightFieldProgress(processedCount: 0, totalCount: 0)
            )
        }

        let coverageByChunk = Self.coveragePointsByChunk(chunks: includedChunks, coveragePoints: coveragePoints)
        let cachedCount = cachedARCollisionHFieldCount(
            chunks: includedChunks,
            coverageByChunk: coverageByChunk
        )
        let requests = includedChunks.compactMap { chunk -> ARCollisionHFieldBuildRequest? in
            let chunkCoveragePoints = coverageByChunk[chunk.identifier] ?? []
            let coverageSignature = Self.coverageSignature(for: chunkCoveragePoints)
            if let cached = arCollisionHFieldCache[chunk.identifier],
               cached.revision == chunk.revision,
               cached.vertexCount == chunk.vertices.count,
               cached.indexCount == chunk.indices.count,
               cached.coverageSignature == coverageSignature,
               cached.detailEnabled == arCollisionHFieldDetailEnabled {
                return nil
            }

            return ARCollisionHFieldBuildRequest(
                chunk: chunk,
                coveragePoints: chunkCoveragePoints,
                coverageSignature: coverageSignature,
                detailEnabled: arCollisionHFieldDetailEnabled,
                revision: environmentRevision + 1
            )
        }

        return ARCollisionHFieldBuildPlan(
            requests: requests,
            initialProgress: HeightFieldProgress(
                processedCount: cachedCount,
                totalCount: includedChunks.count
            )
        )
    }

    nonisolated static func buildARCollisionHField(
        _ request: ARCollisionHFieldBuildRequest
    ) -> ARCollisionHFieldBuildResult {
        let geometry = makeARCollisionHeightField(
            name: arCollisionHFieldName(for: request.chunk.identifier),
            vertices: request.chunk.vertices,
            indices: request.chunk.indices,
            floorFaceMask: request.chunk.floorFaceMask,
            coveragePoints: request.coveragePoints,
            detailEnabled: request.detailEnabled,
            revision: request.revision
        )
        return ARCollisionHFieldBuildResult(
            identifier: request.chunk.identifier,
            chunkRevision: request.chunk.revision,
            vertexCount: request.chunk.vertices.count,
            indexCount: request.chunk.indices.count,
            coverageSignature: request.coverageSignature,
            detailEnabled: request.detailEnabled,
            geometry: geometry
        )
    }

    func installARCollisionHFieldBuildResults(_ results: [ARCollisionHFieldBuildResult]) {
        for result in results {
            guard result.detailEnabled == arCollisionHFieldDetailEnabled else {
                continue
            }

            guard let geometry = result.geometry else {
                arCollisionHFieldCache.removeValue(forKey: result.identifier)
                continue
            }

            arCollisionHFieldCache[result.identifier] = CachedARCollisionHField(
                revision: result.chunkRevision,
                vertexCount: result.vertexCount,
                indexCount: result.indexCount,
                coverageSignature: result.coverageSignature,
                detailEnabled: result.detailEnabled,
                geometry: geometry
            )
        }
    }

    func reset() {
        guard let model, let data else {
            return
        }
        mj_resetData(model, data)
        locomotionController?.reset(rootPosition: rootStartPosition)
        mj_forward(model, data)
        lastStepTime = nil
    }

    func setRobotVelocityCommand(forward: Float, lateral: Float = 0, yawRate: Float = 0) {
        locomotionController?.setVelocityCommand(forward: forward, lateral: lateral, yawRate: yawRate)
    }

    func setRobotNavigationTarget(_ target: SIMD3<Float>?) {
        locomotionController?.setNavigationTarget(target)
    }

    func setRobotNavigationPath(_ path: [SIMD3<Float>]) {
        locomotionController?.setNavigationPath(path)
    }

    func clearRobotNavigationTarget() {
        locomotionController?.setNavigationTarget(nil)
    }

    func spawnRobot(at surfacePoint: SIMD3<Float>) {
        guard let model, let data else {
            return
        }
        robotSpawned = true
        applyRobotCollisionEnabled(true, model: model)
        let rootPosition = SIMD3<Double>(
            Double(surfacePoint.x),
            Double(surfacePoint.y),
            Double(surfacePoint.z + robotKind.spawnHeightAboveSurface * robotScale)
        )
        rootStartPosition = rootPosition
        locomotionController?.setNavigationTarget(nil)
        locomotionController?.setVelocityCommand(forward: 0, lateral: 0, yawRate: 0)
        locomotionController?.reset(rootPosition: rootPosition)
        mj_forward(model, data)
        lastStepTime = nil
    }

    @available(*, deprecated, message: "Temporary collision probe shooter for AR collision debugging; no UI is wired.")
    func launchCollisionProbe(from origin: SIMD3<Float>, direction: SIMD3<Float>) {
        guard let model, let data, !collisionProbeSlots.isEmpty else {
            return
        }

        let length = simd_length(direction)
        guard length > 0.0001 else {
            return
        }

        let slotIndex = nextCollisionProbeSlot % collisionProbeSlots.count
        nextCollisionProbeSlot = (slotIndex + 1) % collisionProbeSlots.count
        collisionProbeActive[slotIndex] = true

        let slot = collisionProbeSlots[slotIndex]
        let launchDirection = direction / length
        let start = origin + launchDirection * 0.18
        let velocity = launchDirection * Self.collisionProbeSpeed
        let spinAxis = simd_cross(SIMD3<Float>(0, 0, 1), launchDirection)
        let spin = simd_length(spinAxis) > 0.0001
            ? simd_normalize(spinAxis) * (Self.collisionProbeSpeed / Self.collisionProbeRadius)
            : SIMD3<Float>(0, 0, 0)

        data.pointee.qpos[slot.qposAddress + 0] = Double(start.x)
        data.pointee.qpos[slot.qposAddress + 1] = Double(start.y)
        data.pointee.qpos[slot.qposAddress + 2] = Double(start.z)
        data.pointee.qpos[slot.qposAddress + 3] = 1
        data.pointee.qpos[slot.qposAddress + 4] = 0
        data.pointee.qpos[slot.qposAddress + 5] = 0
        data.pointee.qpos[slot.qposAddress + 6] = 0
        data.pointee.qvel[slot.qvelAddress + 0] = Double(velocity.x)
        data.pointee.qvel[slot.qvelAddress + 1] = Double(velocity.y)
        data.pointee.qvel[slot.qvelAddress + 2] = Double(velocity.z)
        data.pointee.qvel[slot.qvelAddress + 3] = Double(spin.x)
        data.pointee.qvel[slot.qvelAddress + 4] = Double(spin.y)
        data.pointee.qvel[slot.qvelAddress + 5] = Double(spin.z)

        mj_forward(model, data)
        lastStepTime = nil
    }

    func setGo1VelocityCommand(forward: Float, lateral: Float = 0, yawRate: Float = 0) {
        setRobotVelocityCommand(forward: forward, lateral: lateral, yawRate: yawRate)
    }

    func setGo1NavigationTarget(_ target: SIMD3<Float>?) {
        setRobotNavigationTarget(target)
    }

    func setGo1NavigationPath(_ path: [SIMD3<Float>]) {
        setRobotNavigationPath(path)
    }

    func clearGo1NavigationTarget() {
        clearRobotNavigationTarget()
    }

    func spawnGo1(at surfacePoint: SIMD3<Float>) {
        spawnRobot(at: surfacePoint)
    }

    func step(at timestamp: CFTimeInterval) {
        guard let model, let data else {
            return
        }

        let previous = lastStepTime ?? timestamp
        lastStepTime = timestamp
        let elapsed = min(max(timestamp - previous, fixedStep), fixedStep * Double(maxStepsPerFrame))
        if !robotSpawned && !hasActiveCollisionProbes {
            return
        }
        if robotSpawned, let locomotionController {
            locomotionController.step(elapsed: elapsed)
            deactivateFallenCollisionProbes(model: model, data: data)
            return
        }

        let stepCount = max(1, min(maxStepsPerFrame, Int((elapsed / fixedStep).rounded(.up))))

        for _ in 0..<stepCount {
            mj_step(model, data)
        }
        deactivateFallenCollisionProbes(model: model, data: data)

        if robotSpawned && data.pointee.time > 7.0 {
            reset()
        }
    }

    func stepAndRenderScene(at timestamp: CFTimeInterval) -> MuJoCoRenderScene {
        step(at: timestamp)
        return renderScene()
    }

    func renderScene() -> MuJoCoRenderScene {
        guard let model, let data else {
            return MuJoCoRenderScene(
                environmentMesh: environmentMesh,
                navigationDebugMesh: nil,
                visualMeshes: visualMeshes,
                spheres: [],
                primitives: [],
                meshInstances: [],
                navigationMesh: navigationMesh,
                robotPose: nil,
                worldTransform: worldTransform
            )
        }

        let sphereType = 2
        var spheres: [RenderSphereDescriptor] = []
        var primitives: [RenderPrimitiveDescriptor] = []
        var meshInstances: [RenderMeshInstanceDescriptor] = []
        let hasVisualMeshes = !visualMeshes.isEmpty
        spheres.reserveCapacity(32)
        primitives.reserveCapacity(renderTerrainPrimitives ? 384 : 64)
        meshInstances.reserveCapacity(48)

        if robotSpawned {
            for part in visualMeshParts {
                guard part.bodyID >= 0, part.bodyID < Int(model.pointee.nbody) else {
                    continue
                }
                let bodyMatrix = matrix4x4_fromMuJoCoPose(
                    position: SIMD3<Float>(
                        Float(data.pointee.xpos[3 * part.bodyID + 0]),
                        Float(data.pointee.xpos[3 * part.bodyID + 1]),
                        Float(data.pointee.xpos[3 * part.bodyID + 2])
                    ),
                    xmat: data.pointee.xmat.advanced(by: 9 * part.bodyID)
                )
                meshInstances.append(RenderMeshInstanceDescriptor(
                    meshID: part.meshID,
                    modelMatrix: bodyMatrix * part.localTransform,
                    color: part.color
                ))
            }
        }

        for geomID in 0..<Int(model.pointee.ngeom) {
            let bodyID = Int(model.pointee.geom_bodyid[geomID])
            guard bodyID > 0 else {
                continue
            }
            let collisionProbeSlot = collisionProbeSlotIndex(forBodyID: bodyID)
            let isActiveCollisionProbe = collisionProbeSlot.map { collisionProbeActive[$0] } ?? false
            if collisionProbeSlot != nil && !isActiveCollisionProbe {
                continue
            }
            if !robotSpawned && bodyID != terrainBodyID && !isActiveCollisionProbe {
                continue
            }
            if bodyID == terrainBodyID && !renderTerrainPrimitives {
                continue
            }
            let geomGroup = Int(model.pointee.geom_group[geomID])
            if hasVisualMeshes && geomGroup == 3 {
                continue
            }

            let position = SIMD3<Float>(
                Float(data.pointee.geom_xpos[3 * geomID + 0]),
                Float(data.pointee.geom_xpos[3 * geomID + 1]),
                Float(data.pointee.geom_xpos[3 * geomID + 2])
            )
            let radius = Float(model.pointee.geom_size[3 * geomID])
            let color = SIMD4<Float>(
                model.pointee.geom_rgba[4 * geomID + 0],
                model.pointee.geom_rgba[4 * geomID + 1],
                model.pointee.geom_rgba[4 * geomID + 2],
                model.pointee.geom_rgba[4 * geomID + 3]
            )

            let geomMatrix = matrix4x4_fromMuJoCoPose(
                position: position,
                xmat: data.pointee.geom_xmat.advanced(by: 9 * geomID)
            )
            let geomType = Int(model.pointee.geom_type[geomID])
            let visibleColor = color.w > 0.001 ? color : SIMD4<Float>(0.62, 0.66, 0.70, 1)

            switch geomType {
            case sphereType:
                spheres.append(RenderSphereDescriptor(position: position, radius: radius, color: visibleColor))
            case 3:
                let halfLength = Float(model.pointee.geom_size[3 * geomID + 1])
                primitives.append(RenderPrimitiveDescriptor(
                    kind: .cylinder,
                    modelMatrix: geomMatrix * matrix4x4_scale(SIMD3<Float>(radius, radius, max(halfLength, 0.001))),
                    color: visibleColor
                ))
                let zAxis = SIMD3<Float>(
                    Float(data.pointee.geom_xmat[9 * geomID + 2]),
                    Float(data.pointee.geom_xmat[9 * geomID + 5]),
                    Float(data.pointee.geom_xmat[9 * geomID + 8])
                )
                let capOffset = zAxis * halfLength
                spheres.append(RenderSphereDescriptor(position: position + capOffset, radius: radius, color: visibleColor))
                spheres.append(RenderSphereDescriptor(position: position - capOffset, radius: radius, color: visibleColor))
            case 5:
                let halfLength = Float(model.pointee.geom_size[3 * geomID + 1])
                primitives.append(RenderPrimitiveDescriptor(
                    kind: .cylinder,
                    modelMatrix: geomMatrix * matrix4x4_scale(SIMD3<Float>(radius, radius, max(halfLength, 0.001))),
                    color: visibleColor
                ))
            case 6:
                let size = SIMD3<Float>(
                    Float(model.pointee.geom_size[3 * geomID + 0]),
                    Float(model.pointee.geom_size[3 * geomID + 1]),
                    Float(model.pointee.geom_size[3 * geomID + 2])
                )
                primitives.append(RenderPrimitiveDescriptor(
                    kind: .box,
                    modelMatrix: geomMatrix * matrix4x4_scale(size),
                    color: visibleColor
                ))
            default:
                continue
            }
        }

        if robotSpawned, let path = locomotionController?.navigationPath, !path.isEmpty {
            let visiblePath = visibleNavigationPath(from: path)
            for (index, point) in visiblePath.enumerated() {
                let isFinal = index == visiblePath.count - 1
                let radius: Float = isFinal ? 0.08 : 0.045
                let color = isFinal
                    ? SIMD4<Float>(0.05, 0.9, 0.35, 1)
                    : SIMD4<Float>(1.0, 0.78, 0.18, 1)
                spheres.append(RenderSphereDescriptor(
                    position: SIMD3<Float>(point.x, point.y, point.z + 0.12),
                    radius: radius,
                    color: color
                ))
            }
        } else if robotSpawned, let target = locomotionController?.navigationTarget {
            spheres.append(RenderSphereDescriptor(
                position: SIMD3<Float>(target.x, target.y, target.z + 0.12),
                radius: 0.08,
                color: SIMD4<Float>(0.05, 0.9, 0.35, 1)
            ))
        }

        return MuJoCoRenderScene(
            environmentMesh: environmentMesh,
            navigationDebugMesh: nil,
            visualMeshes: visualMeshes,
            spheres: spheres,
            primitives: primitives,
            meshInstances: meshInstances,
            navigationMesh: navigationMesh,
            robotPose: robotSpawned ? locomotionController?.rootPose() : nil,
            worldTransform: worldTransform
        )
    }

    private func clearModel() {
        if let data {
            mj_deleteData(data)
        }
        if let model {
            mj_deleteModel(model)
        }
        model = nil
        data = nil
        locomotionController = nil
        terrainBodyID = nil
        robotGeomCollisionMasks = []
        collisionProbeSlots = []
        collisionProbeActive = []
        nextCollisionProbeSlot = 0
        renderTerrainPrimitives = false
        environmentMesh = nil
        visualMeshes = [:]
        visualMeshParts = []
        navigationMesh = nil
        worldTransform = matrix_identity_float4x4
        lastStepTime = nil
    }

    private var hasActiveCollisionProbes: Bool {
        collisionProbeActive.contains(true)
    }

    private func applyRobotCollisionEnabled(_ enabled: Bool, model: UnsafeMutablePointer<mjModel>) {
        for mask in robotGeomCollisionMasks {
            model.pointee.geom_contype[mask.geomID] = enabled ? mask.contype : 0
            model.pointee.geom_conaffinity[mask.geomID] = enabled ? mask.conaffinity : 0
        }
    }

    private func collisionProbeSlotIndex(forBodyID bodyID: Int) -> Int? {
        collisionProbeSlots.firstIndex { $0.bodyID == bodyID }
    }

    private func parkAllCollisionProbes(model: UnsafeMutablePointer<mjModel>, data: UnsafeMutablePointer<mjData>) {
        for slotIndex in collisionProbeSlots.indices {
            parkCollisionProbe(slotIndex: slotIndex, model: model, data: data)
        }
        collisionProbeActive = Array(repeating: false, count: collisionProbeSlots.count)
        nextCollisionProbeSlot = 0
    }

    private func parkCollisionProbe(
        slotIndex: Int,
        model: UnsafeMutablePointer<mjModel>,
        data: UnsafeMutablePointer<mjData>
    ) {
        guard collisionProbeSlots.indices.contains(slotIndex) else {
            return
        }

        let slot = collisionProbeSlots[slotIndex]
        data.pointee.qpos[slot.qposAddress + 0] = 0
        data.pointee.qpos[slot.qposAddress + 1] = 0
        data.pointee.qpos[slot.qposAddress + 2] = -20
        data.pointee.qpos[slot.qposAddress + 3] = 1
        data.pointee.qpos[slot.qposAddress + 4] = 0
        data.pointee.qpos[slot.qposAddress + 5] = 0
        data.pointee.qpos[slot.qposAddress + 6] = 0
        for offset in 0..<6 {
            data.pointee.qvel[slot.qvelAddress + offset] = 0
        }
        mj_forward(model, data)
    }

    private func deactivateFallenCollisionProbes(
        model: UnsafeMutablePointer<mjModel>,
        data: UnsafeMutablePointer<mjData>
    ) {
        for slotIndex in collisionProbeSlots.indices where collisionProbeActive[slotIndex] {
            let slot = collisionProbeSlots[slotIndex]
            let z = data.pointee.qpos[slot.qposAddress + 2]
            if z < -10 {
                collisionProbeActive[slotIndex] = false
                parkCollisionProbe(slotIndex: slotIndex, model: model, data: data)
            }
        }
    }

    private func visibleNavigationPath(from path: [SIMD3<Float>]) -> [SIMD3<Float>] {
        let maxVisiblePoints = 28
        guard path.count > maxVisiblePoints else {
            return path
        }
        return Array(path.prefix(maxVisiblePoints - 1)) + [path[path.count - 1]]
    }

    private func makeStateSnapshot() -> SimulationState? {
        guard let model, let data else {
            return nil
        }

        let qpos = (0..<Int(model.pointee.nq)).map { data.pointee.qpos[$0] }
        let qvel = (0..<Int(model.pointee.nv)).map { data.pointee.qvel[$0] }
        let ctrl = (0..<Int(model.pointee.nu)).map { data.pointee.ctrl[$0] }

        return SimulationState(
            qpos: qpos,
            qvel: qvel,
            ctrl: ctrl,
            time: data.pointee.time,
            rootStartPosition: rootStartPosition,
            robotSpawned: robotSpawned,
            collisionProbeActive: collisionProbeActive,
            nextCollisionProbeSlot: nextCollisionProbeSlot,
            controllerState: locomotionController?.snapshot()
        )
    }

    private static func restoreSimulationState(
        _ snapshot: SimulationState,
        model: UnsafeMutablePointer<mjModel>,
        data: UnsafeMutablePointer<mjData>
    ) -> Bool {
        guard Int(model.pointee.nq) == snapshot.qpos.count,
              Int(model.pointee.nv) == snapshot.qvel.count,
              Int(model.pointee.nu) == snapshot.ctrl.count else {
            return false
        }

        data.pointee.time = snapshot.time
        for i in 0..<snapshot.qpos.count {
            data.pointee.qpos[i] = snapshot.qpos[i]
        }
        for i in 0..<snapshot.qvel.count {
            data.pointee.qvel[i] = snapshot.qvel[i]
        }
        for i in 0..<snapshot.ctrl.count {
            data.pointee.ctrl[i] = snapshot.ctrl[i]
        }
        return true
    }

    private static func makeDebugScene(
        robot: LocomotionRobotKind,
        robotScale: Float,
        revision: Int,
        variationPhase: Double? = nil
    ) -> SceneDefinition {
        if let xml = loadBundledXML(named: robot.roughSceneResourceName) {
            let mazeXML = addDebugMazeWalls(to: xml)
            let sceneXML = variationPhase.map { makeDynamicDebugTerrainXML(mazeXML, phase: $0) } ?? mazeXML
            let scaledXML = scaleRobotXML(injectCollisionProbeBodies(into: sceneXML), scale: robotScale)
            return SceneDefinition(
                xml: scaledXML,
                environmentMesh: nil,
                navigationMesh: nil,
                rootStartPosition: scaledRootStartPosition(robot.rootStartPosition, scale: robotScale),
                rootHeightAboveTerrain: robot.rootStartPosition.z * Double(robotScale),
                robotInitiallySpawned: true,
                renderTerrainPrimitives: true,
                buildNavigationMeshFromModel: true
            )
        }

        print("Missing \(robot.displayName) rough XML resource; using fallback debug terrain")
        return makeFallbackDebugScene(robot: robot, robotScale: robotScale, revision: revision)
    }

    private static func makeFallbackDebugScene(
        robot: LocomotionRobotKind,
        robotScale: Float,
        revision: Int
    ) -> SceneDefinition {
        let terrain = makeDebugTerrainMesh(revision: revision)
        let nrow = 25
        let ncol = 25
        let elevations = terrain.heights.map { String(format: "%.5f", $0) }
        let elevationRows = stride(from: 0, to: elevations.count, by: ncol).map {
            elevations[$0..<min($0 + ncol, elevations.count)].joined(separator: " ")
        }.joined(separator: "\n                       ")

        let terrainAsset = """
            <hfield name="debug_terrain" nrow="\(nrow)" ncol="\(ncol)" size="1.8 1.8 0.34 0.05"
                    elevation="\(elevationRows)"/>
        """
        let terrainBody = """
            <body name="terrain">
              <geom name="terrain" type="hfield" hfield="debug_terrain" group="0" rgba="0.50 0.63 0.56 1" friction="1.0 0.04 0.001"/>
            </body>
        """
        let rootHeightOffset = robot.fallbackRootHeightOffset * robotScale
        let rootHeight = Double(terrain.heightAtOrigin + rootHeightOffset)
        let xml = scaleRobotXML(
            makeRobotXML(robot: robot, terrainAsset: terrainAsset, terrainBody: terrainBody),
            scale: robotScale
        )

        return SceneDefinition(
            xml: xml,
            environmentMesh: terrain.mesh,
            navigationMesh: nil,
            rootStartPosition: SIMD3<Double>(0, 0, rootHeight),
            rootHeightAboveTerrain: Double(rootHeightOffset),
            robotInitiallySpawned: true,
            renderTerrainPrimitives: false,
            buildNavigationMeshFromModel: false
        )
    }

    private func makeARScene(
        robot: LocomotionRobotKind,
        robotScale: Float,
        vertices: [SIMD3<Float>],
        indices: [UInt32],
        coveragePoints: [SIMD2<Float>],
        revision: Int
    ) -> SceneDefinition {
        return makeARScene(
            robot: robot,
            robotScale: robotScale,
            chunks: [
                EnvironmentMeshChunkDescriptor(
                    identifier: UUID(uuidString: "00000000-0000-0000-0000-000000000000") ?? UUID(),
                    vertices: vertices,
                    indices: indices,
                    floorFaceMask: nil,
                    revision: revision
                )
            ],
            coveragePoints: coveragePoints,
            revision: revision
        )
    }

    private func makeARScene(
        robot: LocomotionRobotKind,
        robotScale: Float,
        chunks: [EnvironmentMeshChunkDescriptor],
        coveragePoints: [SIMD2<Float>],
        revision: Int,
        progress: ((LoadPhase) -> Void)? = nil
    ) -> SceneDefinition {
        let includedChunks = Self.clippedARCollisionChunks(from: chunks)
        let collisionGeometry = makeARCollisionHeightFields(
            chunks: includedChunks,
            coveragePoints: coveragePoints,
            revision: revision,
            progress: { heightFieldProgress in
                progress?(.generatingHeightField(heightFieldProgress))
            }
        )

        let meshAssets: [String]
        let terrainGeoms: [String]
        let displayMesh: RenderMeshDescriptor
        let navigationMesh: NavigationMeshDescriptor?
        if let collisionGeometry {
            meshAssets = [collisionGeometry.assetXML]
            terrainGeoms = [collisionGeometry.geomXML]
            displayMesh = collisionGeometry.renderMesh
            navigationMesh = collisionGeometry.navigationMesh
        } else {
            var combinedVertices: [SIMD3<Float>] = []
            var combinedIndices: [UInt32] = []
            var combinedFloorFaceMask: [Bool]? = includedChunks.contains { $0.floorFaceMask != nil } ? [] : nil
            var rawMeshAssets: [String] = []
            var rawMeshGeoms: [String] = []
            var emittedMeshNames: [String] = []
            combinedVertices.reserveCapacity(includedChunks.reduce(0) { $0 + $1.vertices.count })
            combinedIndices.reserveCapacity(includedChunks.reduce(0) { $0 + $1.indices.count })

            for chunk in includedChunks {
                let chunkVertices = chunk.vertices
                let chunkIndices = chunk.indices
                let baseIndex = UInt32(combinedVertices.count)
                combinedVertices.append(contentsOf: chunkVertices)
                combinedIndices.append(contentsOf: chunkIndices.map { baseIndex + $0 })
                if combinedFloorFaceMask != nil {
                    let chunkFaceCount = chunkIndices.count / 3
                    for faceIndex in 0..<chunkFaceCount {
                        combinedFloorFaceMask?.append(chunk.floorFaceMask?[faceIndex] == true)
                    }
                }

                let meshName = "ar_chunk_" + chunk.identifier.uuidString.replacingOccurrences(of: "-", with: "_")
                let vertexAttribute = chunkVertices.map {
                    String(format: "%.5f %.5f %.5f", $0.x, $0.y, $0.z)
                }.joined(separator: "  ")
                let faceAttribute = stride(from: 0, to: chunkIndices.count, by: 3).map {
                    "\(chunkIndices[$0]) \(chunkIndices[$0 + 1]) \(chunkIndices[$0 + 2])"
                }.joined(separator: "  ")
                rawMeshAssets.append("""
                    <mesh name="\(meshName)" vertex="\(vertexAttribute)" face="\(faceAttribute)" maxhullvert="128"/>
                """)
                emittedMeshNames.append(meshName)
            }

            for meshName in emittedMeshNames {
                rawMeshGeoms.append("""
                      <geom name="terrain_\(meshName)" type="mesh" mesh="\(meshName)" group="0" contype="1" conaffinity="1" condim="6" rgba="0.08 0.75 0.95 0.24" friction="1.0 0.04 0.001"/>
                """)
            }

            let rawMesh = Self.makeMeshDescriptor(
                vertices: combinedVertices,
                indices: combinedIndices,
                color: SIMD4<Float>(0.08, 0.75, 0.95, 0.32),
                revision: revision
            )
            meshAssets = rawMeshAssets
            terrainGeoms = rawMeshGeoms
            displayMesh = rawMesh
            navigationMesh = Self.makeNavigationMeshDescriptor(from: Self.makeARFloorSurfaceRenderMesh(
                vertices: combinedVertices,
                indices: combinedIndices,
                floorFaceMask: combinedFloorFaceMask,
                revision: revision
            ) ?? rawMesh)
        }

        let terrainAsset = meshAssets.joined(separator: "\n")
        let terrainBody = """
            <body name="terrain">
        \(terrainGeoms.joined(separator: "\n"))
            </body>
        """
        let center = displayMesh.vertices.isEmpty
            ? SIMD3<Float>(0, 0, 0)
            : displayMesh.vertices.reduce(SIMD3<Float>(repeating: 0), +) / Float(displayMesh.vertices.count)
        let xml = Self.scaleRobotXML(
            Self.makeRobotXML(robot: robot, terrainAsset: terrainAsset, terrainBody: terrainBody),
            scale: robotScale
        )
        let rootHeightOffset = robot.arStartHeightAboveSurface * robotScale

        return SceneDefinition(
            xml: xml,
            environmentMesh: displayMesh.vertices.isEmpty ? nil : displayMesh,
            navigationMesh: navigationMesh,
            rootStartPosition: SIMD3<Double>(
                Double(center.x),
                Double(center.y),
                Double(center.z + rootHeightOffset)
            ),
            rootHeightAboveTerrain: Double(rootHeightOffset),
            robotInitiallySpawned: false,
            renderTerrainPrimitives: false,
            buildNavigationMeshFromModel: false
        )
    }

    private static func clippedARCollisionChunks(
        from chunks: [EnvironmentMeshChunkDescriptor]
    ) -> [EnvironmentMeshChunkDescriptor] {
        let sortedChunks = chunks.sorted { lhs, rhs in
            lhs.identifier.uuidString < rhs.identifier.uuidString
        }
        var result: [EnvironmentMeshChunkDescriptor] = []
        var vertexCount = 0
        var indexCount = 0

        for chunk in sortedChunks {
            let remainingVertices = maxARCollisionVertexCount - vertexCount
            let remainingIndices = maxARCollisionIndexCount - indexCount
            guard remainingVertices >= 3, remainingIndices >= 3 else {
                break
            }

            let chunkVertexCount = min(chunk.vertices.count, remainingVertices)
            let chunkVertices = Array(chunk.vertices.prefix(chunkVertexCount))
            let maxFaceCount = remainingIndices / 3
            var chunkIndices: [UInt32] = []
            var selectedFloorMask: [Bool]? = chunk.floorFaceMask == nil ? nil : []
            chunkIndices.reserveCapacity(maxFaceCount * 3)
            for faceIndex in 0..<(chunk.indices.count / 3) {
                guard chunkIndices.count / 3 < maxFaceCount else {
                    break
                }
                let faceStart = faceIndex * 3
                let a = chunk.indices[faceStart]
                let b = chunk.indices[faceStart + 1]
                let c = chunk.indices[faceStart + 2]
                guard a < UInt32(chunkVertexCount),
                      b < UInt32(chunkVertexCount),
                      c < UInt32(chunkVertexCount) else {
                    continue
                }
                chunkIndices.append(contentsOf: [a, b, c])
                if let floorFaceMask = chunk.floorFaceMask {
                    selectedFloorMask?.append(faceIndex < floorFaceMask.count && floorFaceMask[faceIndex])
                }
            }
            guard chunkVertices.count >= 3, chunkIndices.count >= 3 else {
                continue
            }

            result.append(EnvironmentMeshChunkDescriptor(
                identifier: chunk.identifier,
                vertices: chunkVertices,
                indices: chunkIndices,
                floorFaceMask: selectedFloorMask,
                revision: chunk.revision
            ))
            vertexCount += chunkVertices.count
            indexCount += chunkIndices.count
        }

        return result
    }

    private func cachedARCollisionHFieldCount(
        chunks: [EnvironmentMeshChunkDescriptor],
        coverageByChunk: [UUID: [SIMD2<Float>]]
    ) -> Int {
        chunks.reduce(0) { count, chunk in
            let chunkCoveragePoints = coverageByChunk[chunk.identifier] ?? []
            let coverageSignature = Self.coverageSignature(for: chunkCoveragePoints)
            guard let cached = arCollisionHFieldCache[chunk.identifier],
                  cached.revision == chunk.revision,
                  cached.vertexCount == chunk.vertices.count,
                  cached.indexCount == chunk.indices.count,
                  cached.coverageSignature == coverageSignature,
                  cached.detailEnabled == arCollisionHFieldDetailEnabled else {
                return count
            }
            return count + 1
        }
    }

    private func makeARCollisionHeightFields(
        chunks: [EnvironmentMeshChunkDescriptor],
        coveragePoints: [SIMD2<Float>],
        revision: Int,
        progress: ((HeightFieldProgress) -> Void)? = nil
    ) -> ARCollisionGeometry? {
        guard !chunks.isEmpty else {
            arCollisionHFieldCache.removeAll()
            progress?(HeightFieldProgress(processedCount: 0, totalCount: 0))
            return nil
        }

        let chunkIDs = Set(chunks.map(\.identifier))
        arCollisionHFieldCache = arCollisionHFieldCache.filter { chunkIDs.contains($0.key) }

        let coverageByChunk = Self.coveragePointsByChunk(chunks: chunks, coveragePoints: coveragePoints)
        var geometries: [ARCollisionGeometry] = []
        geometries.reserveCapacity(chunks.count)
        var processedCount = cachedARCollisionHFieldCount(chunks: chunks, coverageByChunk: coverageByChunk)
        progress?(HeightFieldProgress(processedCount: processedCount, totalCount: chunks.count))

        for chunk in chunks {
            let chunkCoveragePoints = coverageByChunk[chunk.identifier] ?? []
            let coverageSignature = Self.coverageSignature(for: chunkCoveragePoints)
            if let cached = arCollisionHFieldCache[chunk.identifier],
               cached.revision == chunk.revision,
               cached.vertexCount == chunk.vertices.count,
               cached.indexCount == chunk.indices.count,
               cached.coverageSignature == coverageSignature,
               cached.detailEnabled == arCollisionHFieldDetailEnabled {
                geometries.append(cached.geometry)
                continue
            }

            let name = Self.arCollisionHFieldName(for: chunk.identifier)
            guard let geometry = Self.makeARCollisionHeightField(
                name: name,
                vertices: chunk.vertices,
                indices: chunk.indices,
                floorFaceMask: chunk.floorFaceMask,
                coveragePoints: chunkCoveragePoints,
                detailEnabled: arCollisionHFieldDetailEnabled,
                revision: revision
            ) else {
                arCollisionHFieldCache.removeValue(forKey: chunk.identifier)
                processedCount += 1
                progress?(HeightFieldProgress(processedCount: processedCount, totalCount: chunks.count))
                continue
            }

            arCollisionHFieldCache[chunk.identifier] = CachedARCollisionHField(
                revision: chunk.revision,
                vertexCount: chunk.vertices.count,
                indexCount: chunk.indices.count,
                coverageSignature: coverageSignature,
                detailEnabled: arCollisionHFieldDetailEnabled,
                geometry: geometry
            )
            geometries.append(geometry)
            processedCount += 1
            progress?(HeightFieldProgress(processedCount: processedCount, totalCount: chunks.count))
        }

        return Self.combineARCollisionGeometries(geometries, revision: revision)
    }

    private static func coveragePointsByChunk(
        chunks: [EnvironmentMeshChunkDescriptor],
        coveragePoints: [SIMD2<Float>]
    ) -> [UUID: [SIMD2<Float>]] {
        guard !coveragePoints.isEmpty else {
            return [:]
        }

        let bounds = chunks.compactMap { chunk -> ARChunkBounds? in
            guard let first = chunk.vertices.first else {
                return nil
            }
            var minX = first.x
            var maxX = first.x
            var minY = first.y
            var maxY = first.y
            for vertex in chunk.vertices.dropFirst() {
                minX = min(minX, vertex.x)
                maxX = max(maxX, vertex.x)
                minY = min(minY, vertex.y)
                maxY = max(maxY, vertex.y)
            }
            return ARChunkBounds(
                identifier: chunk.identifier,
                minX: minX,
                maxX: maxX,
                minY: minY,
                maxY: maxY
            )
        }
        guard !bounds.isEmpty else {
            return [:]
        }

        var result: [UUID: [SIMD2<Float>]] = [:]
        for point in coveragePoints {
            guard let nearest = bounds.min(by: {
                $0.squaredDistance(to: point) < $1.squaredDistance(to: point)
            }) else {
                continue
            }
            result[nearest.identifier, default: []].append(point)
        }
        return result
    }

    private static func coverageSignature(for points: [SIMD2<Float>]) -> Int {
        var hasher = Hasher()
        for point in points {
            hasher.combine(Int((point.x * 4).rounded()))
            hasher.combine(Int((point.y * 4).rounded()))
        }
        return hasher.finalize()
    }

    nonisolated private static func arCollisionHFieldName(for identifier: UUID) -> String {
        "ar_hfield_" + identifier.uuidString.replacingOccurrences(of: "-", with: "_")
    }

    private static func combineARCollisionGeometries(
        _ geometries: [ARCollisionGeometry],
        revision: Int
    ) -> ARCollisionGeometry? {
        guard !geometries.isEmpty else {
            return nil
        }

        var vertices: [SIMD3<Float>] = []
        var indices: [UInt32] = []
        vertices.reserveCapacity(geometries.reduce(0) { $0 + $1.renderMesh.vertices.count })
        indices.reserveCapacity(geometries.reduce(0) { $0 + $1.renderMesh.indices.count })

        for geometry in geometries {
            guard vertices.count <= Int(UInt32.max) - geometry.renderMesh.vertices.count else {
                break
            }
            let base = UInt32(vertices.count)
            vertices.append(contentsOf: geometry.renderMesh.vertices)
            indices.append(contentsOf: geometry.renderMesh.indices.map { base + $0 })
        }

        let renderMesh = makeMeshDescriptor(
            vertices: vertices,
            indices: indices,
            color: SIMD4<Float>(1.0, 0.58, 0.05, 0.42),
            revision: revision
        )
        return ARCollisionGeometry(
            assetXML: geometries.map(\.assetXML).joined(separator: "\n"),
            geomXML: geometries.map(\.geomXML).joined(separator: "\n"),
            renderMesh: renderMesh,
            navigationMesh: makeNavigationMeshDescriptor(from: renderMesh)
        )
    }

    nonisolated private static func makeARCollisionHeightField(
        name: String,
        vertices: [SIMD3<Float>],
        indices: [UInt32],
        floorFaceMask: [Bool]?,
        coveragePoints: [SIMD2<Float>],
        detailEnabled: Bool,
        revision: Int
    ) -> ARCollisionGeometry? {
        let floorTriangles = makeARFloorSurfaceTriangles(
            vertices: vertices,
            indices: indices,
            floorFaceMask: floorFaceMask
        )
        guard floorTriangles.count >= 2 else {
            return nil
        }

        var horizontalSamples: [SIMD3<Float>] = []
        horizontalSamples.reserveCapacity(min(floorTriangles.count * 4, 12000))
        for triangle in floorTriangles {
            horizontalSamples.append(contentsOf: [triangle.a, triangle.b, triangle.c, triangle.center])
        }

        let sortedZ = horizontalSamples.map(\.z).sorted()
        let floorReference = sortedZ[min(sortedZ.count - 1, max(0, Int(Double(sortedZ.count) * 0.20)))]
        let samples = horizontalSamples.filter { $0.z <= floorReference + 0.75 }
        guard samples.count >= 8 else {
            return nil
        }

        let first = samples[0]
        var minX = first.x
        var maxX = first.x
        var minY = first.y
        var maxY = first.y
        for sample in samples.dropFirst() {
            minX = min(minX, sample.x)
            maxX = max(maxX, sample.x)
            minY = min(minY, sample.y)
            maxY = max(maxY, sample.y)
        }
        let coveragePadding: Float = 0.35
        for point in coveragePoints {
            minX = min(minX, point.x - coveragePadding)
            maxX = max(maxX, point.x + coveragePadding)
            minY = min(minY, point.y - coveragePadding)
            maxY = max(maxY, point.y + coveragePadding)
        }

        let width = maxX - minX
        let depth = maxY - minY
        guard width > 0.15, depth > 0.15 else {
            return nil
        }

        let margin: Float = 0.08
        let centerX = (minX + maxX) * 0.5
        let centerY = (minY + maxY) * 0.5
        let halfX = width * 0.5 + margin
        let halfY = depth * 0.5 + margin
        let cellSize: Float = detailEnabled ? 0.04 : 0.08
        let maxGridSize = detailEnabled ? 96 : 64
        let ncol = min(maxGridSize, max(12, Int(ceil((halfX * 2) / cellSize)) + 1))
        let nrow = min(maxGridSize, max(12, Int(ceil((halfY * 2) / cellSize)) + 1))
        var cellHeights = [Float?](repeating: nil, count: nrow * ncol)
        let gridX = { (col: Int) -> Float in
            centerX - halfX + (Float(col) / Float(max(ncol - 1, 1))) * halfX * 2
        }
        let gridY = { (row: Int) -> Float in
            centerY - halfY + (Float(row) / Float(max(nrow - 1, 1))) * halfY * 2
        }

        for triangle in floorTriangles {
            let minCol = clampedGridIndex(
                value: min(min(triangle.a.x, triangle.b.x), triangle.c.x),
                center: centerX,
                halfExtent: halfX,
                count: ncol
            )
            let maxCol = clampedGridIndex(
                value: max(max(triangle.a.x, triangle.b.x), triangle.c.x),
                center: centerX,
                halfExtent: halfX,
                count: ncol
            )
            let minRow = clampedGridIndex(
                value: min(min(triangle.a.y, triangle.b.y), triangle.c.y),
                center: centerY,
                halfExtent: halfY,
                count: nrow
            )
            let maxRow = clampedGridIndex(
                value: max(max(triangle.a.y, triangle.b.y), triangle.c.y),
                center: centerY,
                halfExtent: halfY,
                count: nrow
            )

            for row in minRow...maxRow {
                for col in minCol...maxCol {
                    guard let height = interpolatedHeight(
                        at: SIMD2<Float>(gridX(col), gridY(row)),
                        in: triangle
                    ) else {
                        continue
                    }
                    let cellIndex = row * ncol + col
                    if let current = cellHeights[cellIndex] {
                        cellHeights[cellIndex] = min(current, height)
                    } else {
                        cellHeights[cellIndex] = height
                    }
                }
            }
        }

        let fallbackSampleLimit = detailEnabled ? 6000 : 2500
        let sampleStride = max(1, samples.count / fallbackSampleLimit)
        for row in 0..<nrow {
            for col in 0..<ncol where cellHeights[row * ncol + col] == nil {
                let x = gridX(col)
                let y = gridY(row)
                var bestDistance = Float.greatestFiniteMagnitude
                var bestHeight = floorReference
                for sampleIndex in stride(from: 0, to: samples.count, by: sampleStride) {
                    let sample = samples[sampleIndex]
                    let offset = SIMD2<Float>(sample.x - x, sample.y - y)
                    let distance = simd_dot(offset, offset)
                    if distance < bestDistance {
                        bestDistance = distance
                        bestHeight = sample.z
                    }
                }
                cellHeights[row * ncol + col] = bestHeight
            }
        }

        let resolvedHeights = cellHeights.map { $0 ?? floorReference }
        guard let minZ = resolvedHeights.min(), let maxZ = resolvedHeights.max() else {
            return nil
        }
        let heightScale = max(maxZ - minZ, 0.03)
        var gridVertices: [SIMD3<Float>] = []
        var gridIndices: [UInt32] = []
        gridVertices.reserveCapacity(nrow * ncol)
        gridIndices.reserveCapacity((nrow - 1) * (ncol - 1) * 6)
        for row in 0..<nrow {
            let y = gridY(row)
            for col in 0..<ncol {
                let x = gridX(col)
                gridVertices.append(SIMD3<Float>(x, y, resolvedHeights[row * ncol + col]))
            }
        }
        for row in 0..<(nrow - 1) {
            for col in 0..<(ncol - 1) {
                let a = UInt32(row * ncol + col)
                let b = UInt32(row * ncol + col + 1)
                let c = UInt32((row + 1) * ncol + col)
                let d = UInt32((row + 1) * ncol + col + 1)
                gridIndices.append(contentsOf: [a, b, d, a, d, c])
            }
        }

        let renderMesh = makeMeshDescriptor(
            vertices: gridVertices,
            indices: gridIndices,
            color: SIMD4<Float>(1.0, 0.58, 0.05, 0.42),
            revision: revision
        )
        let elevationRows = (0..<nrow).map { row in
            (0..<ncol).map { col in
                let height = resolvedHeights[row * ncol + col]
                let normalized = min(max((height - minZ) / heightScale, 0), 1)
                return String(format: "%.5f", normalized)
            }.joined(separator: " ")
        }.joined(separator: "\n                       ")

        let assetXML = """
            <hfield name="\(name)" nrow="\(nrow)" ncol="\(ncol)" size="\(String(format: "%.5f", halfX)) \(String(format: "%.5f", halfY)) \(String(format: "%.5f", heightScale)) 0.04"
                    elevation="\(elevationRows)"/>
        """
        let geomXML = """
              <geom name="terrain_\(name)" type="hfield" hfield="\(name)" pos="\(String(format: "%.5f", centerX)) \(String(format: "%.5f", centerY)) \(String(format: "%.5f", minZ))" group="0" contype="1" conaffinity="1" condim="6" rgba="0.08 0.75 0.95 0.08" friction="1.1 0.05 0.001"/>
        """
        return ARCollisionGeometry(
            assetXML: assetXML,
            geomXML: geomXML,
            renderMesh: renderMesh,
            navigationMesh: makeNavigationMeshDescriptor(from: renderMesh)
        )
    }

    private static func makeARFloorSurfaceRenderMesh(
        vertices: [SIMD3<Float>],
        indices: [UInt32],
        floorFaceMask: [Bool]?,
        revision: Int
    ) -> RenderMeshDescriptor? {
        let floorTriangles = makeARFloorSurfaceTriangles(
            vertices: vertices,
            indices: indices,
            floorFaceMask: floorFaceMask
        )
        guard !floorTriangles.isEmpty else {
            return nil
        }

        let maxTriangleCount = 6000
        let selectedCount = min(maxTriangleCount, floorTriangles.count)
        var surfaceVertices: [SIMD3<Float>] = []
        var surfaceIndices: [UInt32] = []
        surfaceVertices.reserveCapacity(selectedCount * 3)
        surfaceIndices.reserveCapacity(selectedCount * 3)
        for sampleIndex in 0..<selectedCount {
            let triangle = floorTriangles[sampleIndex * floorTriangles.count / selectedCount]
            let base = UInt32(surfaceVertices.count)
            surfaceVertices.append(contentsOf: [triangle.a, triangle.b, triangle.c])
            surfaceIndices.append(contentsOf: [base, base + 1, base + 2])
        }

        return makeMeshDescriptor(
            vertices: surfaceVertices,
            indices: surfaceIndices,
            color: SIMD4<Float>(0.08, 0.75, 0.95, 0.32),
            revision: revision
        )
    }

    nonisolated private static func makeARFloorSurfaceTriangles(
        vertices: [SIMD3<Float>],
        indices: [UInt32],
        floorFaceMask: [Bool]?
    ) -> [ARSurfaceTriangle] {
        guard vertices.count >= 3, indices.count >= 3 else {
            return []
        }

        var classifiedFloorTriangles: [ARSurfaceTriangle] = []
        var horizontalTriangles: [ARSurfaceTriangle] = []
        let faceCount = indices.count / 3
        classifiedFloorTriangles.reserveCapacity(min(faceCount, 12000))
        horizontalTriangles.reserveCapacity(min(faceCount, 12000))
        for faceStart in stride(from: 0, to: indices.count - 2, by: 3) {
            let faceIndex = faceStart / 3
            let ia = Int(indices[faceStart])
            let ib = Int(indices[faceStart + 1])
            let ic = Int(indices[faceStart + 2])
            guard ia < vertices.count, ib < vertices.count, ic < vertices.count else {
                continue
            }

            let a = vertices[ia]
            var b = vertices[ib]
            var c = vertices[ic]
            var normal = simd_cross(b - a, c - a)
            let area = simd_length(normal)
            guard area > 0.00001 else {
                continue
            }
            normal /= area
            if normal.z < 0 {
                swap(&b, &c)
                normal = -normal
            }

            let triangle = ARSurfaceTriangle(a: a, b: b, c: c, normal: normal)
            if let floorFaceMask, faceIndex < floorFaceMask.count, floorFaceMask[faceIndex] {
                classifiedFloorTriangles.append(triangle)
            }
            if normal.z > 0.55 {
                horizontalTriangles.append(triangle)
            }
        }

        if classifiedFloorTriangles.count >= 2 {
            return classifiedFloorTriangles
        }

        guard horizontalTriangles.count >= 2 else {
            return horizontalTriangles
        }

        let zSamples = horizontalTriangles.flatMap { [$0.a.z, $0.b.z, $0.c.z, $0.center.z] }.sorted()
        let floorReference = zSamples[min(zSamples.count - 1, max(0, Int(Double(zSamples.count) * 0.20)))]
        let floorTriangles = horizontalTriangles.filter { triangle in
            let minTriangleZ = min(min(triangle.a.z, triangle.b.z), triangle.c.z)
            return minTriangleZ <= floorReference + 0.75
        }
        return floorTriangles.count >= 2 ? floorTriangles : horizontalTriangles
    }

    nonisolated private static func makeNavigationMeshDescriptor(from mesh: RenderMeshDescriptor) -> NavigationMeshDescriptor {
        NavigationMeshDescriptor(
            vertices: mesh.vertices,
            indices: mesh.indices,
            revision: mesh.revision
        )
    }

    nonisolated private static func interpolatedHeight(
        at point: SIMD2<Float>,
        in triangle: ARSurfaceTriangle
    ) -> Float? {
        let a = SIMD2<Float>(triangle.a.x, triangle.a.y)
        let b = SIMD2<Float>(triangle.b.x, triangle.b.y)
        let c = SIMD2<Float>(triangle.c.x, triangle.c.y)
        let v0 = b - a
        let v1 = c - a
        let v2 = point - a
        let determinant = v0.x * v1.y - v1.x * v0.y
        guard abs(determinant) > 0.000001 else {
            return nil
        }

        let u = (v2.x * v1.y - v1.x * v2.y) / determinant
        let v = (v0.x * v2.y - v2.x * v0.y) / determinant
        let epsilon: Float = 0.0001
        guard u >= -epsilon, v >= -epsilon, u + v <= 1 + epsilon else {
            return nil
        }

        return triangle.a.z
            + u * (triangle.b.z - triangle.a.z)
            + v * (triangle.c.z - triangle.a.z)
    }

    nonisolated private static func clampedGridIndex(value: Float, center: Float, halfExtent: Float, count: Int) -> Int {
        let normalized = (value - (center - halfExtent)) / max(halfExtent * 2, 0.0001)
        return min(max(Int((normalized * Float(count - 1)).rounded()), 0), count - 1)
    }

    private static func makeDebugTerrainMesh(revision: Int) -> TerrainMesh {
        let nrow = 25
        let ncol = 25
        let halfX: Float = 1.8
        let halfY: Float = 1.8
        let heightScale: Float = 0.34

        var heights: [Float] = []
        var vertices: [SIMD3<Float>] = []
        heights.reserveCapacity(nrow * ncol)
        vertices.reserveCapacity(nrow * ncol)

        for row in 0..<nrow {
            let v = Float(row) / Float(nrow - 1)
            let y = (v * 2 - 1) * halfY
            for col in 0..<ncol {
                let u = Float(col) / Float(ncol - 1)
                let x = (u * 2 - 1) * halfX
                let ridge = 0.48 + 0.22 * sinf(8.0 * x + 1.4 * cosf(3.0 * y))
                let bumps = 0.16 * cosf(7.0 * y) + 0.12 * sinf(5.0 * (x + y))
                let edgeFalloff = smoothstep(1.75, 0.7, max(abs(x), abs(y)))
                let normalizedHeight = min(max(0.18 + (ridge + bumps - 0.35) * edgeFalloff, 0.0), 1.0)
                heights.append(normalizedHeight)
                vertices.append(SIMD3<Float>(x, y, normalizedHeight * heightScale))
            }
        }

        var indices: [UInt32] = []
        indices.reserveCapacity((nrow - 1) * (ncol - 1) * 6)
        for row in 0..<(nrow - 1) {
            for col in 0..<(ncol - 1) {
                let a = UInt32(row * ncol + col)
                let b = UInt32(row * ncol + col + 1)
                let c = UInt32((row + 1) * ncol + col)
                let d = UInt32((row + 1) * ncol + col + 1)
                indices.append(contentsOf: [a, b, c, b, d, c])
            }
        }

        let mesh = makeMeshDescriptor(
            vertices: vertices,
            indices: indices,
            color: SIMD4<Float>(0.50, 0.63, 0.56, 1.0),
            revision: revision
        )
        let originIndex = (nrow / 2) * ncol + (ncol / 2)
        return TerrainMesh(mesh: mesh, heights: heights, heightAtOrigin: heights[originIndex] * heightScale)
    }

    nonisolated private static func makeMeshDescriptor(
        vertices: [SIMD3<Float>],
        indices: [UInt32],
        color: SIMD4<Float>,
        revision: Int
    ) -> RenderMeshDescriptor {
        var normals = Array(repeating: SIMD3<Float>(0, 0, 1), count: vertices.count)
        guard vertices.count >= 3, indices.count >= 3 else {
            return RenderMeshDescriptor(vertices: vertices, normals: normals, indices: indices, color: color, revision: revision)
        }

        normals = Array(repeating: SIMD3<Float>(repeating: 0), count: vertices.count)
        for faceStart in stride(from: 0, to: indices.count - 2, by: 3) {
            let ia = Int(indices[faceStart])
            let ib = Int(indices[faceStart + 1])
            let ic = Int(indices[faceStart + 2])
            guard ia < vertices.count, ib < vertices.count, ic < vertices.count else {
                continue
            }
            let normal = simd_cross(vertices[ib] - vertices[ia], vertices[ic] - vertices[ia])
            normals[ia] += normal
            normals[ib] += normal
            normals[ic] += normal
        }

        normals = normals.map { length_squared($0) > 0.000001 ? normalize($0) : SIMD3<Float>(0, 0, 1) }
        return RenderMeshDescriptor(vertices: vertices, normals: normals, indices: indices, color: color, revision: revision)
    }

    private static func loadRobotVisualAssets(
        robot: LocomotionRobotKind,
        model: UnsafeMutablePointer<mjModel>,
        revision: Int,
        scale: Float
    ) -> VisualAssetSet {
        guard let manifestURL = bundledResourceURL(
            path: "render_manifests/\(robot.renderManifestResourceName).json"
        ) else {
            return VisualAssetSet(meshes: [:], parts: [])
        }

        do {
            let manifest = try JSONDecoder().decode(
                RenderAssetManifest.self,
                from: Data(contentsOf: manifestURL)
            )
            var meshIDByPath: [String: Int] = [:]
            var meshes: [Int: RenderMeshDescriptor] = [:]
            var parts: [VisualMeshPart] = []

            for part in manifest.parts {
                guard let bodyID = objectID(model, type: mjOBJ_BODY, name: part.bodyName),
                      let meshPath = normalizedRenderAssetPath(part.url) else {
                    continue
                }

                let meshID: Int
                if let existingMeshID = meshIDByPath[meshPath] {
                    meshID = existingMeshID
                } else {
                    meshID = meshIDByPath.count
                    guard let meshURL = bundledResourceURL(path: meshPath),
                          let mesh = try? loadGLBRenderMesh(
                              at: meshURL,
                              revision: revision * 100_000 + meshID,
                              scale: scale
                          ) else {
                        continue
                    }
                    meshIDByPath[meshPath] = meshID
                    meshes[meshID] = mesh
                }

                parts.append(VisualMeshPart(
                    meshID: meshID,
                    bodyID: bodyID,
                    localTransform: matrix4x4_fromMuJoCoLocalTransform(
                        position: vector3(part.pos, default: SIMD3<Float>(repeating: 0)) * scale,
                        quat: part.quat
                    ),
                    color: vector4(part.rgba, default: SIMD4<Float>(0.2, 0.2, 0.2, 1))
                ))
            }

            return VisualAssetSet(meshes: meshes, parts: parts)
        } catch {
            print("Failed to load \(robot.displayName) render manifest: \(error)")
            return VisualAssetSet(meshes: [:], parts: [])
        }
    }

    private static func loadGLBRenderMesh(
        at url: URL,
        revision: Int,
        scale: Float
    ) throws -> RenderMeshDescriptor {
        let data = try Data(contentsOf: url)
        guard data.count >= 20,
              readUInt32(data, at: 0) == 0x4654_6C67,
              readUInt32(data, at: 4) == 2 else {
            throw VisualAssetError.invalidGLB
        }

        var offset = 12
        var jsonData: Data?
        var binaryData: Data?
        while offset + 8 <= data.count {
            let chunkLength = Int(readUInt32(data, at: offset))
            let chunkType = readUInt32(data, at: offset + 4)
            offset += 8
            guard offset + chunkLength <= data.count else {
                throw VisualAssetError.invalidGLB
            }
            let chunk = data.subdata(in: offset..<(offset + chunkLength))
            if chunkType == 0x4E4F_534A {
                jsonData = chunk
            } else if chunkType == 0x004E_4942 {
                binaryData = chunk
            }
            offset += chunkLength
        }

        guard let jsonData, let binaryData else {
            throw VisualAssetError.missingGLBChunk
        }

        let glb = try JSONDecoder().decode(GLBAsset.self, from: jsonData)
        guard let primitive = glb.meshes.first?.primitives.first,
              let positionAccessor = primitive.attributes["POSITION"],
              let indexAccessor = primitive.indices else {
            throw VisualAssetError.unsupportedGLB
        }

        let vertices = try readFloat3Accessor(
            asset: glb,
            binaryData: binaryData,
            accessorIndex: positionAccessor
        ).map { $0 * scale }
        let indices = try readIndexAccessor(
            asset: glb,
            binaryData: binaryData,
            accessorIndex: indexAccessor
        )
        let normals: [SIMD3<Float>]
        if let normalAccessor = primitive.attributes["NORMAL"],
           let parsedNormals = try? readFloat3Accessor(
               asset: glb,
               binaryData: binaryData,
               accessorIndex: normalAccessor
           ),
           parsedNormals.count == vertices.count {
            normals = parsedNormals
        } else {
            return makeMeshDescriptor(
                vertices: vertices,
                indices: indices,
                color: SIMD4<Float>(1, 1, 1, 1),
                revision: revision
            )
        }

        return RenderMeshDescriptor(
            vertices: vertices,
            normals: normals,
            indices: indices,
            color: SIMD4<Float>(1, 1, 1, 1),
            revision: revision
        )
    }

    private static func readFloat3Accessor(
        asset: GLBAsset,
        binaryData: Data,
        accessorIndex: Int
    ) throws -> [SIMD3<Float>] {
        guard accessorIndex >= 0, accessorIndex < asset.accessors.count else {
            throw VisualAssetError.unsupportedGLB
        }
        let accessor = asset.accessors[accessorIndex]
        guard accessor.componentType == 5126,
              componentCount(for: accessor.type) >= 3,
              let viewIndex = accessor.bufferView,
              viewIndex >= 0,
              viewIndex < asset.bufferViews.count else {
            throw VisualAssetError.unsupportedGLB
        }

        let view = asset.bufferViews[viewIndex]
        let start = (view.byteOffset ?? 0) + (accessor.byteOffset ?? 0)
        let stride = view.byteStride ?? componentCount(for: accessor.type) * MemoryLayout<Float>.stride
        var values: [SIMD3<Float>] = []
        values.reserveCapacity(accessor.count)
        for index in 0..<accessor.count {
            let base = start + index * stride
            guard base + 12 <= binaryData.count else {
                throw VisualAssetError.unsupportedGLB
            }
            values.append(SIMD3<Float>(
                readFloat32(binaryData, at: base),
                readFloat32(binaryData, at: base + 4),
                readFloat32(binaryData, at: base + 8)
            ))
        }
        return values
    }

    private static func readIndexAccessor(
        asset: GLBAsset,
        binaryData: Data,
        accessorIndex: Int
    ) throws -> [UInt32] {
        guard accessorIndex >= 0, accessorIndex < asset.accessors.count else {
            throw VisualAssetError.unsupportedGLB
        }
        let accessor = asset.accessors[accessorIndex]
        guard let viewIndex = accessor.bufferView,
              viewIndex >= 0,
              viewIndex < asset.bufferViews.count else {
            throw VisualAssetError.unsupportedGLB
        }

        let view = asset.bufferViews[viewIndex]
        let start = (view.byteOffset ?? 0) + (accessor.byteOffset ?? 0)
        let componentSize: Int
        switch accessor.componentType {
        case 5123:
            componentSize = MemoryLayout<UInt16>.stride
        case 5125:
            componentSize = MemoryLayout<UInt32>.stride
        default:
            throw VisualAssetError.unsupportedGLB
        }
        let stride = view.byteStride ?? componentSize

        var values: [UInt32] = []
        values.reserveCapacity(accessor.count)
        for index in 0..<accessor.count {
            let offset = start + index * stride
            guard offset + componentSize <= binaryData.count else {
                throw VisualAssetError.unsupportedGLB
            }
            if accessor.componentType == 5123 {
                values.append(UInt32(readUInt16(binaryData, at: offset)))
            } else {
                values.append(readUInt32(binaryData, at: offset))
            }
        }
        return values
    }

    private static func bundledResourceURL(path: String) -> URL? {
        guard let resourceURL = Bundle.main.resourceURL else {
            return nil
        }
        let cleanPath = path.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        let flatName = URL(fileURLWithPath: cleanPath).lastPathComponent
        let candidates = [
            resourceURL.appendingPathComponent("Resources", isDirectory: true).appendingPathComponent(cleanPath),
            resourceURL.appendingPathComponent(cleanPath),
            resourceURL.appendingPathComponent(flatName),
        ]
        return candidates.first { FileManager.default.fileExists(atPath: $0.path) }
    }

    private static func normalizedRenderAssetPath(_ url: String) -> String? {
        let path = url.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        guard !path.isEmpty else {
            return nil
        }
        return path
    }

    private static func componentCount(for accessorType: String) -> Int {
        switch accessorType {
        case "SCALAR":
            return 1
        case "VEC2":
            return 2
        case "VEC3":
            return 3
        case "VEC4":
            return 4
        default:
            return 0
        }
    }

    private static func matrix4x4_fromMuJoCoLocalTransform(
        position: SIMD3<Float>,
        quat values: [Float]
    ) -> simd_float4x4 {
        let quatValues = values.count == 4 ? values : [1, 0, 0, 0]
        let rotation = simd_float4x4(simd_quatf(
            ix: quatValues[1],
            iy: quatValues[2],
            iz: quatValues[3],
            r: quatValues[0]
        ))
        return matrix4x4_translation(position) * rotation
    }

    private static func vector3(_ values: [Float]?, default defaultValue: SIMD3<Float>) -> SIMD3<Float> {
        guard let values, values.count >= 3 else {
            return defaultValue
        }
        return SIMD3<Float>(values[0], values[1], values[2])
    }

    private static func vector4(_ values: [Float]?, default defaultValue: SIMD4<Float>) -> SIMD4<Float> {
        guard let values, values.count >= 4 else {
            return defaultValue
        }
        return SIMD4<Float>(values[0], values[1], values[2], values[3])
    }

    private static func readUInt16(_ data: Data, at offset: Int) -> UInt16 {
        var value: UInt16 = 0
        _ = withUnsafeMutableBytes(of: &value) { buffer in
            data.copyBytes(to: buffer, from: offset..<(offset + MemoryLayout<UInt16>.stride))
        }
        return UInt16(littleEndian: value)
    }

    private static func readUInt32(_ data: Data, at offset: Int) -> UInt32 {
        var value: UInt32 = 0
        _ = withUnsafeMutableBytes(of: &value) { buffer in
            data.copyBytes(to: buffer, from: offset..<(offset + MemoryLayout<UInt32>.stride))
        }
        return UInt32(littleEndian: value)
    }

    private static func readFloat32(_ data: Data, at offset: Int) -> Float {
        Float(bitPattern: readUInt32(data, at: offset))
    }

    private static func makeDynamicDebugTerrainXML(_ xml: String, phase: Double) -> String {
        xml.split(separator: "\n", omittingEmptySubsequences: false).map { rawLine in
            let line = String(rawLine)
            guard line.contains("name=\"terrain_"),
                  let terrainIndex = terrainIndex(in: line),
                  var position = vectorAttribute("pos", in: line),
                  position.count == 3 else {
                return line
            }

            let isMovableHField = line.contains("type=\"hfield\"")
            let isMovableBox: Bool
            let amplitude: Double
            if line.contains("type=\"box\""),
               let size = vectorAttribute("size", in: line),
               size.count == 3,
               size[0] <= 4.5,
               size[1] <= 4.5,
               size[2] <= 0.55 {
                isMovableBox = true
                amplitude = size[2] > 0.45 ? 0.028 : 0.045
            } else {
                isMovableBox = false
                amplitude = 0.035
            }

            guard isMovableBox || isMovableHField else {
                return line
            }

            let chunkID = debugTerrainChunkID(in: line, terrainIndex: terrainIndex, position: position)
            let chunkPhase = Double(chunkID % 4096) * 0.017
            let broadWave = amplitude * sin(phase + chunkPhase)
            let fineWave = amplitude * 0.18 * sin(phase * 0.73 + chunkPhase * 1.9)
            position[2] += broadWave + fineWave
            return replacingVectorAttribute("pos", in: line, values: position)
        }.joined(separator: "\n")
    }

    private static func addDebugMazeWalls(to xml: String) -> String {
        guard !xml.contains("maze_wall_0") else {
            return xml
        }

        let anchor = #"      <light pos="0 0 24" dir="0 0 -1" type="directional" />"#
        guard xml.contains(anchor) else {
            return xml
        }
        return xml.replacingOccurrences(of: anchor, with: debugMazeWallsXML() + "\n" + anchor)
    }

    private static func debugMazeWallsXML() -> String {
        """
              <geom name="maze_wall_0" size="0.12 2.0 1.05" pos="0.9 -0.6 -0.15" type="box" mass="0" friction="1.1 0.04 0.001" rgba="0.12 0.42 0.48 1" />
              <geom name="maze_wall_1" size="1.55 0.12 1.05" pos="-0.55 1.3 -0.15" type="box" mass="0" friction="1.1 0.04 0.001" rgba="0.12 0.42 0.48 1" />
              <geom name="maze_wall_2" size="0.12 1.15 1.05" pos="-1.95 2.35 -0.15" type="box" mass="0" friction="1.1 0.04 0.001" rgba="0.12 0.42 0.48 1" />
              <geom name="maze_wall_3" size="1.35 0.12 1.05" pos="0.65 -2.7 -0.15" type="box" mass="0" friction="1.1 0.04 0.001" rgba="0.12 0.42 0.48 1" />
        """
    }

    private static func terrainIndex(in line: String) -> Int? {
        guard let nameRange = line.range(of: #"name="terrain_"#) else {
            return nil
        }
        let indexStart = nameRange.upperBound
        guard let indexEnd = line[indexStart...].firstIndex(of: "\"") else {
            return nil
        }
        return Int(line[indexStart..<indexEnd])
    }

    private static func debugTerrainChunkID(in line: String, terrainIndex: Int, position: [Double]) -> Int {
        let tileX = Int(floor((position[0] + 20.0) / 8.0))
        let tileY = Int(floor((position[1] + 20.0) / 8.0))
        return stableChunkHash("\(tileX):\(tileY):\(debugTerrainChunkKey(in: line, terrainIndex: terrainIndex))")
    }

    private static func debugTerrainChunkKey(in line: String, terrainIndex: Int) -> String {
        if let valueRange = attributeValueRange("rgba", in: line) {
            return "rgba:" + line[valueRange]
        }
        if let valueRange = attributeValueRange("hfield", in: line) {
            return "hfield:" + line[valueRange]
        }
        return "terrain:\(terrainIndex)"
    }

    private static func stableChunkHash(_ string: String) -> Int {
        var hash: UInt32 = 2_166_136_261
        for byte in string.utf8 {
            hash ^= UInt32(byte)
            hash = hash &* 16_777_619
        }
        return Int(hash & 0x7fff_ffff)
    }

    private static func vectorAttribute(_ attribute: String, in line: String) -> [Double]? {
        guard let valueRange = attributeValueRange(attribute, in: line) else {
            return nil
        }
        return line[valueRange].split(separator: " ").compactMap { Double($0) }
    }

    private static func replacingVectorAttribute(_ attribute: String, in line: String, values: [Double]) -> String {
        guard let valueRange = attributeValueRange(attribute, in: line) else {
            return line
        }

        var updated = line
        let value = values.map { String(format: "%.6f", $0) }.joined(separator: " ")
        updated.replaceSubrange(valueRange, with: value)
        return updated
    }

    private static func makeNavigationMesh(
        model: UnsafeMutablePointer<mjModel>,
        data: UnsafeMutablePointer<mjData>,
        terrainBodyID: Int?,
        revision: Int
    ) -> NavigationMeshDescriptor? {
        guard let terrainBodyID else {
            return nil
        }

        var vertices: [SIMD3<Float>] = []
        var indices: [UInt32] = []
        vertices.reserveCapacity(16384)
        indices.reserveCapacity(49152)
        let obstacles = makeNavigationObstacles(model: model, data: data, terrainBodyID: terrainBodyID)

        for geomID in 0..<Int(model.pointee.ngeom) {
            guard Int(model.pointee.geom_bodyid[geomID]) == terrainBodyID else {
                continue
            }
            let name = geomName(model: model, geomID: geomID) ?? ""
            if name.hasPrefix("maze_wall_") {
                continue
            }

            switch Int(model.pointee.geom_type[geomID]) {
            case 1:
                appendNavigationHField(
                    geomID: geomID,
                    model: model,
                    data: data,
                    obstacles: obstacles,
                    vertices: &vertices,
                    indices: &indices
                )
            case 6:
                appendNavigationBoxTop(
                    geomID: geomID,
                    model: model,
                    data: data,
                    obstacles: obstacles,
                    vertices: &vertices,
                    indices: &indices
                )
            case 7:
                appendNavigationMesh(
                    geomID: geomID,
                    model: model,
                    data: data,
                    obstacles: obstacles,
                    vertices: &vertices,
                    indices: &indices
                )
            default:
                continue
            }
        }

        guard vertices.count >= 3, indices.count >= 3 else {
            return nil
        }
        return NavigationMeshDescriptor(vertices: vertices, indices: indices, revision: revision)
    }

    private static func appendNavigationMesh(
        geomID: Int,
        model: UnsafeMutablePointer<mjModel>,
        data: UnsafeMutablePointer<mjData>,
        obstacles: [NavigationObstacle],
        vertices: inout [SIMD3<Float>],
        indices: inout [UInt32]
    ) {
        let meshID = Int(model.pointee.geom_dataid[geomID])
        guard meshID >= 0 else {
            return
        }

        let vertexAddress = Int(model.pointee.mesh_vertadr[meshID])
        let vertexCount = Int(model.pointee.mesh_vertnum[meshID])
        let faceAddress = Int(model.pointee.mesh_faceadr[meshID])
        let faceCount = Int(model.pointee.mesh_facenum[meshID])
        guard vertexCount >= 3,
              faceCount >= 1,
              vertices.count <= Int(UInt32.max) - vertexCount else {
            return
        }

        let base = UInt32(vertices.count)
        for vertexIndex in 0..<vertexCount {
            let source = model.pointee.mesh_vert.advanced(by: 3 * (vertexAddress + vertexIndex))
            vertices.append(transformLocalPoint(
                geomID: geomID,
                data: data,
                local: SIMD3<Double>(Double(source[0]), Double(source[1]), Double(source[2]))
            ))
        }

        for faceIndex in 0..<faceCount {
            let face = model.pointee.mesh_face.advanced(by: 3 * (faceAddress + faceIndex))
            let a = UInt32(face[0])
            let b = UInt32(face[1])
            let c = UInt32(face[2])
            guard a < UInt32(vertexCount), b < UInt32(vertexCount), c < UInt32(vertexCount) else {
                continue
            }

            let va = vertices[Int(base + a)]
            let vb = vertices[Int(base + b)]
            let vc = vertices[Int(base + c)]
            let center = (va + vb + vc) / 3
            if obstacles.contains(where: { $0.contains(center) }) {
                continue
            }
            indices.append(contentsOf: [base + a, base + b, base + c])
        }
    }

    private static func appendNavigationBoxTop(
        geomID: Int,
        model: UnsafeMutablePointer<mjModel>,
        data: UnsafeMutablePointer<mjData>,
        obstacles: [NavigationObstacle],
        vertices: inout [SIMD3<Float>],
        indices: inout [UInt32]
    ) {
        guard vertices.count <= Int(UInt32.max) - 4 else {
            return
        }

        let size = SIMD3<Double>(
            model.pointee.geom_size[3 * geomID + 0],
            model.pointee.geom_size[3 * geomID + 1],
            model.pointee.geom_size[3 * geomID + 2]
        )
        guard size.x > 0, size.y > 0, size.z >= 0 else {
            return
        }

        if let bounds = navigationBoxTopBounds(geomID: geomID, data: data, size: size),
           obstacles.contains(where: { $0.intersects(bounds) }) {
            appendNavigationBoxTopGrid(
                geomID: geomID,
                data: data,
                size: size,
                obstacles: obstacles,
                vertices: &vertices,
                indices: &indices
            )
            return
        }

        let base = UInt32(vertices.count)
        let localCorners = [
            SIMD3<Double>(-size.x, -size.y, size.z),
            SIMD3<Double>( size.x, -size.y, size.z),
            SIMD3<Double>( size.x,  size.y, size.z),
            SIMD3<Double>(-size.x,  size.y, size.z),
        ]
        for local in localCorners {
            vertices.append(transformLocalPoint(geomID: geomID, data: data, local: local))
        }

        indices.append(contentsOf: [
            base, base + 1, base + 2,
            base, base + 2, base + 3,
        ])

        guard size.z >= 0.30, vertices.count <= Int(UInt32.max) - 8 else {
            return
        }

        let bottomBase = UInt32(vertices.count)
        let bottomCorners = [
            SIMD3<Double>(-size.x, -size.y, -size.z),
            SIMD3<Double>( size.x, -size.y, -size.z),
            SIMD3<Double>( size.x,  size.y, -size.z),
            SIMD3<Double>(-size.x,  size.y, -size.z),
        ]
        for local in bottomCorners {
            vertices.append(transformLocalPoint(geomID: geomID, data: data, local: local))
        }

        indices.append(contentsOf: [
            bottomBase + 0, bottomBase + 1, base + 1,
            bottomBase + 0, base + 1, base + 0,
            bottomBase + 1, bottomBase + 2, base + 2,
            bottomBase + 1, base + 2, base + 1,
            bottomBase + 2, bottomBase + 3, base + 3,
            bottomBase + 2, base + 3, base + 2,
            bottomBase + 3, bottomBase + 0, base + 0,
            bottomBase + 3, base + 0, base + 3,
        ])
    }

    private static func appendNavigationHField(
        geomID: Int,
        model: UnsafeMutablePointer<mjModel>,
        data: UnsafeMutablePointer<mjData>,
        obstacles: [NavigationObstacle],
        vertices: inout [SIMD3<Float>],
        indices: inout [UInt32]
    ) {
        let hfieldID = Int(model.pointee.geom_dataid[geomID])
        guard hfieldID >= 0 else {
            return
        }
        let nrow = Int(model.pointee.hfield_nrow[hfieldID])
        let ncol = Int(model.pointee.hfield_ncol[hfieldID])
        guard nrow >= 2, ncol >= 2, vertices.count <= Int(UInt32.max) - nrow * ncol else {
            return
        }

        let size = model.pointee.hfield_size.advanced(by: 4 * hfieldID)
        let halfX = Double(size[0])
        let halfY = Double(size[1])
        let heightScale = Double(size[2])
        let dataAddress = Int(model.pointee.hfield_adr[hfieldID])
        let base = UInt32(vertices.count)

        for row in 0..<nrow {
            let y = (Double(row) / Double(nrow - 1) * 2 - 1) * halfY
            for col in 0..<ncol {
                let x = (Double(col) / Double(ncol - 1) * 2 - 1) * halfX
                let elevation = Double(model.pointee.hfield_data[dataAddress + row * ncol + col]) * heightScale
                vertices.append(transformLocalPoint(
                    geomID: geomID,
                    data: data,
                    local: SIMD3<Double>(x, y, elevation)
                ))
            }
        }

        for row in 0..<(nrow - 1) {
            for col in 0..<(ncol - 1) {
                let a = base + UInt32(row * ncol + col)
                let b = base + UInt32(row * ncol + col + 1)
                let c = base + UInt32((row + 1) * ncol + col)
                let d = base + UInt32((row + 1) * ncol + col + 1)
                let center = (vertices[Int(a)] + vertices[Int(b)] + vertices[Int(c)] + vertices[Int(d)]) * 0.25
                if obstacles.contains(where: { $0.contains(center) }) {
                    continue
                }
                indices.append(contentsOf: [a, b, d, a, d, c])
            }
        }
    }

    private static func appendNavigationBoxTopGrid(
        geomID: Int,
        data: UnsafeMutablePointer<mjData>,
        size: SIMD3<Double>,
        obstacles: [NavigationObstacle],
        vertices: inout [SIMD3<Float>],
        indices: inout [UInt32]
    ) {
        let cellSize = 0.25
        let xSteps = max(1, Int(ceil(size.x * 2 / cellSize)))
        let ySteps = max(1, Int(ceil(size.y * 2 / cellSize)))
        guard xSteps * ySteps < 12000 else {
            return
        }

        for ix in 0..<xSteps {
            let x0 = -size.x + Double(ix) / Double(xSteps) * size.x * 2
            let x1 = -size.x + Double(ix + 1) / Double(xSteps) * size.x * 2
            for iy in 0..<ySteps {
                let y0 = -size.y + Double(iy) / Double(ySteps) * size.y * 2
                let y1 = -size.y + Double(iy + 1) / Double(ySteps) * size.y * 2
                let center = transformLocalPoint(
                    geomID: geomID,
                    data: data,
                    local: SIMD3<Double>((x0 + x1) * 0.5, (y0 + y1) * 0.5, size.z)
                )
                if obstacles.contains(where: { $0.contains(center) }) {
                    continue
                }
                guard vertices.count <= Int(UInt32.max) - 4 else {
                    return
                }

                let base = UInt32(vertices.count)
                vertices.append(transformLocalPoint(geomID: geomID, data: data, local: SIMD3<Double>(x0, y0, size.z)))
                vertices.append(transformLocalPoint(geomID: geomID, data: data, local: SIMD3<Double>(x1, y0, size.z)))
                vertices.append(transformLocalPoint(geomID: geomID, data: data, local: SIMD3<Double>(x1, y1, size.z)))
                vertices.append(transformLocalPoint(geomID: geomID, data: data, local: SIMD3<Double>(x0, y1, size.z)))
                indices.append(contentsOf: [
                    base, base + 1, base + 2,
                    base, base + 2, base + 3,
                ])
            }
        }
    }

    private static func makeNavigationObstacles(
        model: UnsafeMutablePointer<mjModel>,
        data: UnsafeMutablePointer<mjData>,
        terrainBodyID: Int
    ) -> [NavigationObstacle] {
        var obstacles: [NavigationObstacle] = []
        for geomID in 0..<Int(model.pointee.ngeom) {
            guard Int(model.pointee.geom_bodyid[geomID]) == terrainBodyID,
                  geomName(model: model, geomID: geomID)?.hasPrefix("maze_wall_") == true,
                  Int(model.pointee.geom_type[geomID]) == 6 else {
                continue
            }

            let size = SIMD3<Double>(
                model.pointee.geom_size[3 * geomID + 0],
                model.pointee.geom_size[3 * geomID + 1],
                model.pointee.geom_size[3 * geomID + 2]
            )
            guard let bounds = navigationBoxTopBounds(geomID: geomID, data: data, size: size) else {
                continue
            }
            obstacles.append(bounds.expanded(by: navigationObstacleMargin))
        }
        return obstacles
    }

    private static func navigationBoxTopBounds(
        geomID: Int,
        data: UnsafeMutablePointer<mjData>,
        size: SIMD3<Double>
    ) -> NavigationObstacle? {
        let corners = [
            SIMD3<Double>(-size.x, -size.y, size.z),
            SIMD3<Double>( size.x, -size.y, size.z),
            SIMD3<Double>( size.x,  size.y, size.z),
            SIMD3<Double>(-size.x,  size.y, size.z),
        ].map { transformLocalPoint(geomID: geomID, data: data, local: $0) }
        guard let first = corners.first else {
            return nil
        }
        var minX = first.x
        var maxX = first.x
        var minY = first.y
        var maxY = first.y
        for corner in corners.dropFirst() {
            minX = min(minX, corner.x)
            maxX = max(maxX, corner.x)
            minY = min(minY, corner.y)
            maxY = max(maxY, corner.y)
        }
        return NavigationObstacle(minX: minX, maxX: maxX, minY: minY, maxY: maxY)
    }

    private static func transformLocalPoint(
        geomID: Int,
        data: UnsafeMutablePointer<mjData>,
        local: SIMD3<Double>
    ) -> SIMD3<Float> {
        let position = data.pointee.geom_xpos.advanced(by: 3 * geomID)
        let xmat = data.pointee.geom_xmat.advanced(by: 9 * geomID)
        return SIMD3<Float>(
            Float(position[0] + xmat[0] * local.x + xmat[1] * local.y + xmat[2] * local.z),
            Float(position[1] + xmat[3] * local.x + xmat[4] * local.y + xmat[5] * local.z),
            Float(position[2] + xmat[6] * local.x + xmat[7] * local.y + xmat[8] * local.z)
        )
    }

    private static func attributeValueRange(_ attribute: String, in line: String) -> Range<String.Index>? {
        guard let attributeRange = line.range(of: attribute + "=\"") else {
            return nil
        }
        let valueStart = attributeRange.upperBound
        guard let valueEnd = line[valueStart...].firstIndex(of: "\"") else {
            return nil
        }
        return valueStart..<valueEnd
    }

    private static func scaleRobotXML(_ xml: String, scale: Float) -> String {
        guard abs(scale - 1.0) > 0.0001 else {
            return xml
        }

        let factor = Double(scale)
        var robotBodyDepth = 0
        var robotDefaultDepth = 0
        let lines = xml.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        let scaledLines = lines.map { originalLine -> String in
            let opensRobotBody = bodyName(in: originalLine)?.hasPrefix("robot/") ?? false
            let opensRobotDefault = defaultClass(in: originalLine)?.hasPrefix("robot/") ?? false
            let isRobotScoped = robotBodyDepth > 0
                || robotDefaultDepth > 0
                || opensRobotBody
                || opensRobotDefault
                || originalLine.contains("\"robot/")

            var line = originalLine
            if isRobotScoped {
                line = scaleNumericAttribute("pos", in: line, by: factor)
                line = scaleNumericAttribute("size", in: line, by: factor)
                line = scaleNumericAttribute("fromto", in: line, by: factor)
                line = scaleNumericAttribute("diaginertia", in: line, by: factor * factor)
                line = scaleNumericAttribute("armature", in: line, by: factor * factor)
                line = scaleNumericAttribute("gainprm", in: line, by: factor)
                line = scaleNumericAttribute("biasprm", in: line, by: factor)
                line = scaleNumericAttribute("forcerange", in: line, by: factor)
                line = scaleNumericAttribute("actuatorfrcrange", in: line, by: factor)
                line = scaleNumericAttribute("damping", in: line, by: factor)
                line = scaleNumericAttribute("frictionloss", in: line, by: factor)
            }
            if line.contains("<key"), line.contains("qpos=") {
                line = scaleRootQposAttribute(in: line, by: factor)
            }

            let opensBody = opensXMLTag("body", in: originalLine)
            if opensBody, robotBodyDepth > 0 || opensRobotBody {
                robotBodyDepth += 1
            }
            if closesXMLTag("body", in: originalLine), robotBodyDepth > 0 {
                robotBodyDepth -= 1
            }

            let opensDefault = opensXMLTag("default", in: originalLine)
            if opensDefault, robotDefaultDepth > 0 || opensRobotDefault {
                robotDefaultDepth += 1
            }
            if closesXMLTag("default", in: originalLine), robotDefaultDepth > 0 {
                robotDefaultDepth -= 1
            }

            return line
        }
        return scaledLines.joined(separator: "\n")
    }

    private static func bodyName(in line: String) -> String? {
        guard line.range(of: #"<body\b"#, options: .regularExpression) != nil,
              let range = attributeValueRange("name", in: line) else {
            return nil
        }
        return String(line[range])
    }

    private static func defaultClass(in line: String) -> String? {
        guard line.range(of: #"<default\b"#, options: .regularExpression) != nil,
              let range = attributeValueRange("class", in: line) else {
            return nil
        }
        return String(line[range])
    }

    private static func opensXMLTag(_ tag: String, in line: String) -> Bool {
        line.range(of: "<\(tag)\\b", options: .regularExpression) != nil
            && line.range(of: "</\(tag)\\b", options: .regularExpression) == nil
            && !line.contains("/>")
    }

    private static func closesXMLTag(_ tag: String, in line: String) -> Bool {
        line.range(of: "</\(tag)\\b", options: .regularExpression) != nil
    }

    private static func scaleNumericAttribute(_ attribute: String, in line: String, by factor: Double) -> String {
        guard let range = attributeValueRange(attribute, in: line) else {
            return line
        }
        let values = line[range].split(whereSeparator: \.isWhitespace).compactMap { Double($0) }
        guard !values.isEmpty else {
            return line
        }
        let scaledValue = values
            .map { formatXMLNumber($0 * factor) }
            .joined(separator: " ")
        var output = line
        output.replaceSubrange(range, with: scaledValue)
        return output
    }

    private static func scaleRootQposAttribute(in line: String, by factor: Double) -> String {
        guard let range = attributeValueRange("qpos", in: line) else {
            return line
        }
        var values = line[range].split(whereSeparator: \.isWhitespace).compactMap { Double($0) }
        guard values.count >= 3 else {
            return line
        }
        values[0] *= factor
        values[1] *= factor
        values[2] *= factor
        let scaledValue = values.map(formatXMLNumber).joined(separator: " ")
        var output = line
        output.replaceSubrange(range, with: scaledValue)
        return output
    }

    private static func scaledRootStartPosition(_ position: SIMD3<Double>, scale: Float) -> SIMD3<Double> {
        SIMD3<Double>(position.x, position.y, position.z * Double(scale))
    }

    private static func formatXMLNumber(_ value: Double) -> String {
        if abs(value) < 1.0e-12 {
            return "0"
        }
        return String(format: "%.8g", value)
    }

    private static func rootPositionAboveTerrain(
        x: Double,
        y: Double,
        heightAboveTerrain: Double,
        model: UnsafeMutablePointer<mjModel>,
        data: UnsafeMutablePointer<mjData>,
        robot: LocomotionRobotKind
    ) -> SIMD3<Double>? {
        let spec = robot.policySpec
        let bodyExclude = objectID(model, type: mjOBJ_BODY, name: spec.terrainScanExcludeBodyName) ?? -1
        let rayStartZ = max(25.0, heightAboveTerrain + 10.0)
        var pnt = [mjtNum](arrayLiteral: x, y, rayStartZ)
        var vec = [mjtNum](arrayLiteral: 0, 0, -1)
        var geomGroup = [mjtByte](repeating: 0, count: 6)
        geomGroup[0] = 1
        var geomID = [Int32](repeating: -1, count: 1)
        var normal = [mjtNum](repeating: 0, count: 3)
        let distance = mj_ray(
            model,
            data,
            &pnt,
            &vec,
            &geomGroup,
            1,
            Int32(bodyExclude),
            &geomID,
            &normal
        )
        guard distance >= 0, distance <= 100 else {
            return nil
        }
        let terrainZ = rayStartZ - distance
        return SIMD3<Double>(x, y, terrainZ + heightAboveTerrain)
    }

    private static func makeRobotXML(robot: LocomotionRobotKind, terrainAsset: String, terrainBody: String) -> String {
        guard let template = loadBundledXML(named: robot.flatSceneResourceName) else {
            print("Missing \(robot.displayName) XML resource")
            return """
            <mujoco model="missing_robot">
              <worldbody>
                <body name="terrain">
                  <geom name="terrain" type="plane" size="0 0 0.01"/>
                </body>
              </worldbody>
            </mujoco>
            """
        }

        let terrainAnchor = """
            <body name="terrain">
              <geom name="terrain" size="0 0 0.01" type="plane" material="groundplane" />
            </body>
        """
        let templateWithAsset: String
        if terrainAsset.isEmpty {
            templateWithAsset = template
        } else {
            templateWithAsset = template.replacingOccurrences(of: "  </asset>", with: terrainAsset + "\n  </asset>")
        }
        let templateWithTerrain = templateWithAsset.replacingOccurrences(of: terrainAnchor, with: terrainBody)
        return injectCollisionProbeBodies(into: templateWithTerrain)
    }

    private static func injectCollisionProbeBodies(into xml: String) -> String {
        let xmlWithBodies = xml.replacingOccurrences(
            of: "  </worldbody>",
            with: Self.makeCollisionProbeBodiesXML() + "\n  </worldbody>"
        )
        return appendCollisionProbeKeyframeQpos(to: xmlWithBodies)
    }

    private static func makeCollisionProbeBodiesXML() -> String {
        (0..<collisionProbeSlotCount).map { index in
            """
                <body name="collision_probe_\(index)" pos="0 0 -20">
                  <joint name="collision_probe_\(index)_free" type="free" />
                  <geom name="collision_probe_\(index)_geom" type="sphere" size="\(String(format: "%.3f", collisionProbeRadius))" mass="0.08" group="4" condim="6" friction="0.85 0.08 0.002" rgba="1.0 0.28 0.08 1" />
                </body>
            """
        }.joined(separator: "\n")
    }

    private static func appendCollisionProbeKeyframeQpos(to xml: String) -> String {
        let parkedQpos = Array(repeating: "0 0 -20 1 0 0 0", count: collisionProbeSlotCount).joined(separator: " ")
        return xml.replacingOccurrences(
            of: #"<key\b([^>]*)\bqpos="([^"]*)""#,
            with: #"<key$1qpos="$2 \#(parkedQpos)""#,
            options: .regularExpression
        )
    }

    private static func makeCollisionProbeSlots(model: UnsafeMutablePointer<mjModel>) -> [CollisionProbeSlot] {
        (0..<collisionProbeSlotCount).compactMap { index in
            guard let bodyID = objectID(model, type: mjOBJ_BODY, name: "collision_probe_\(index)"),
                  let jointID = objectID(model, type: mjOBJ_JOINT, name: "collision_probe_\(index)_free") else {
                return nil
            }

            return CollisionProbeSlot(
                bodyID: bodyID,
                qposAddress: Int(model.pointee.jnt_qposadr[jointID]),
                qvelAddress: Int(model.pointee.jnt_dofadr[jointID])
            )
        }
    }

    private static func makeRobotGeomCollisionMasks(model: UnsafeMutablePointer<mjModel>) -> [GeomCollisionMask] {
        var masks: [GeomCollisionMask] = []
        for geomID in 0..<Int(model.pointee.ngeom) {
            guard geomName(model: model, geomID: geomID)?.hasPrefix("robot/") == true else {
                continue
            }
            let contype = model.pointee.geom_contype[geomID]
            let conaffinity = model.pointee.geom_conaffinity[geomID]
            guard contype != 0 || conaffinity != 0 else {
                continue
            }
            masks.append(GeomCollisionMask(geomID: geomID, contype: contype, conaffinity: conaffinity))
        }
        return masks
    }

    private static func loadBundledXML(named name: String) -> String? {
        guard let url = Bundle.main.url(forResource: name, withExtension: "xml", subdirectory: "Resources")
                ?? Bundle.main.url(forResource: name, withExtension: "xml") else {
            return nil
        }
        return try? String(contentsOf: url, encoding: .utf8)
    }

    private static func geomName(model: UnsafeMutablePointer<mjModel>, geomID: Int) -> String? {
        guard let name = mj_id2name(model, Int32(mjOBJ_GEOM.rawValue), Int32(geomID)) else {
            return nil
        }
        return String(cString: name)
    }

    private static func objectID(_ model: UnsafeMutablePointer<mjModel>, type: mjtObj, name: String) -> Int? {
        let id = name.withCString { mj_name2id(model, Int32(type.rawValue), $0) }
        return id >= 0 ? Int(id) : nil
    }
}

nonisolated private struct SceneDefinition {
    let xml: String
    let environmentMesh: RenderMeshDescriptor?
    let navigationMesh: NavigationMeshDescriptor?
    let rootStartPosition: SIMD3<Double>
    let rootHeightAboveTerrain: Double?
    let robotInitiallySpawned: Bool
    let renderTerrainPrimitives: Bool
    let buildNavigationMeshFromModel: Bool
}

nonisolated struct ARCollisionGeometry: Sendable {
    let assetXML: String
    let geomXML: String
    let renderMesh: RenderMeshDescriptor
    let navigationMesh: NavigationMeshDescriptor
}

nonisolated private struct CachedARCollisionHField {
    let revision: Int
    let vertexCount: Int
    let indexCount: Int
    let coverageSignature: Int
    let detailEnabled: Bool
    let geometry: ARCollisionGeometry
}

nonisolated private struct ARChunkBounds {
    let identifier: UUID
    let minX: Float
    let maxX: Float
    let minY: Float
    let maxY: Float

    func squaredDistance(to point: SIMD2<Float>) -> Float {
        let dx = point.x < minX ? minX - point.x : max(point.x - maxX, 0)
        let dy = point.y < minY ? minY - point.y : max(point.y - maxY, 0)
        return dx * dx + dy * dy
    }
}

nonisolated private struct ARSurfaceTriangle {
    let a: SIMD3<Float>
    let b: SIMD3<Float>
    let c: SIMD3<Float>
    let normal: SIMD3<Float>

    nonisolated var center: SIMD3<Float> {
        (a + b + c) / 3
    }
}

nonisolated private struct VisualAssetSet {
    let meshes: [Int: RenderMeshDescriptor]
    let parts: [VisualMeshPart]
}

nonisolated private struct VisualMeshPart {
    let meshID: Int
    let bodyID: Int
    let localTransform: simd_float4x4
    let color: SIMD4<Float>
}

nonisolated private struct GeomCollisionMask {
    let geomID: Int
    let contype: Int32
    let conaffinity: Int32
}

nonisolated private struct CollisionProbeSlot {
    let bodyID: Int
    let qposAddress: Int
    let qvelAddress: Int
}

nonisolated private struct RenderAssetManifest: Decodable {
    let parts: [RenderAssetManifestPart]
}

nonisolated private struct RenderAssetManifestPart: Decodable {
    let bodyName: String
    let url: String
    let pos: [Float]?
    let quat: [Float]
    let rgba: [Float]?
}

nonisolated private struct GLBAsset: Decodable {
    let accessors: [GLBAccessor]
    let bufferViews: [GLBBufferView]
    let meshes: [GLBMesh]
}

nonisolated private struct GLBAccessor: Decodable {
    let bufferView: Int?
    let byteOffset: Int?
    let componentType: Int
    let count: Int
    let type: String
}

nonisolated private struct GLBBufferView: Decodable {
    let byteOffset: Int?
    let byteStride: Int?
}

nonisolated private struct GLBMesh: Decodable {
    let primitives: [GLBPrimitive]
}

nonisolated private struct GLBPrimitive: Decodable {
    let attributes: [String: Int]
    let indices: Int?
}

nonisolated private enum VisualAssetError: Error {
    case invalidGLB
    case missingGLBChunk
    case unsupportedGLB
}

nonisolated private struct SimulationState {
    let qpos: [Double]
    let qvel: [Double]
    let ctrl: [Double]
    let time: Double
    let rootStartPosition: SIMD3<Double>
    let robotSpawned: Bool
    let collisionProbeActive: [Bool]
    let nextCollisionProbeSlot: Int
    let controllerState: Go1PolicyController.State?
}

nonisolated private struct TerrainMesh {
    let mesh: RenderMeshDescriptor
    let heights: [Float]
    let heightAtOrigin: Float
}

nonisolated private struct NavigationObstacle {
    let minX: Float
    let maxX: Float
    let minY: Float
    let maxY: Float

    func expanded(by margin: Float) -> NavigationObstacle {
        NavigationObstacle(
            minX: minX - margin,
            maxX: maxX + margin,
            minY: minY - margin,
            maxY: maxY + margin
        )
    }

    func contains(_ point: SIMD3<Float>) -> Bool {
        point.x >= minX && point.x <= maxX && point.y >= minY && point.y <= maxY
    }

    func intersects(_ other: NavigationObstacle) -> Bool {
        minX <= other.maxX && maxX >= other.minX && minY <= other.maxY && maxY >= other.minY
    }
}

nonisolated private func hsvToRGB(hue: Double, saturation: Double, value: Double) -> SIMD3<Double> {
    let i = floor(hue * 6)
    let f = hue * 6 - i
    let p = value * (1 - saturation)
    let q = value * (1 - f * saturation)
    let t = value * (1 - (1 - f) * saturation)

    switch Int(i) % 6 {
    case 0: return SIMD3(value, t, p)
    case 1: return SIMD3(q, value, p)
    case 2: return SIMD3(p, value, t)
    case 3: return SIMD3(p, q, value)
    case 4: return SIMD3(t, p, value)
    default: return SIMD3(value, p, q)
    }
}

nonisolated private func smoothstep(_ edge0: Float, _ edge1: Float, _ x: Float) -> Float {
    let t = min(max((x - edge0) / (edge1 - edge0), 0), 1)
    return t * t * (3 - 2 * t)
}

nonisolated private func matrix4x4_fromMuJoCoPose(position: SIMD3<Float>, xmat: UnsafePointer<mjtNum>) -> matrix_float4x4 {
    matrix_float4x4(columns: (
        SIMD4<Float>(Float(xmat[0]), Float(xmat[3]), Float(xmat[6]), 0),
        SIMD4<Float>(Float(xmat[1]), Float(xmat[4]), Float(xmat[7]), 0),
        SIMD4<Float>(Float(xmat[2]), Float(xmat[5]), Float(xmat[8]), 0),
        SIMD4<Float>(position.x, position.y, position.z, 1)
    ))
}
