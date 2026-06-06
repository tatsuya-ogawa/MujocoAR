//
//  RecastNavigator.swift
//  MujocoAR
//
//  Created by Tatsuya Ogawa on 2026/05/29.
//

import Foundation
import RecastNavigationKit
import simd

@MainActor
final class RecastNavigator {
    private var navMesh: RecastNavMesh?
    private var debugMesh: RenderMeshDescriptor?
    private var sourceRevision: Int?

    func clear() {
        navMesh = nil
        debugMesh = nil
        sourceRevision = nil
    }

    func sceneWithDebugMesh(_ scene: MuJoCoRenderScene) -> MuJoCoRenderScene {
        MuJoCoRenderScene(
            environmentMesh: scene.environmentMesh,
            navigationDebugMesh: makeDebugMesh(in: scene),
            visualMeshes: scene.visualMeshes,
            spheres: scene.spheres,
            primitives: scene.primitives,
            meshInstances: scene.meshInstances,
            navigationMesh: scene.navigationMesh,
            robotPose: scene.robotPose,
            worldTransform: scene.worldTransform
        )
    }

    func path(from start: SIMD3<Float>, to target: SIMD3<Float>, in scene: MuJoCoRenderScene) -> [SIMD3<Float>]? {
        guard rebuildIfNeeded(from: scene.navigationMesh), let navMesh else {
            return nil
        }

        let startOnMesh = navMesh.findNearestPoint(recastPoint(fromMuJoCo: start))
        let endOnMesh = navMesh.findNearestPoint(recastPoint(fromMuJoCo: target))
        guard let path = navMesh.findPathResult(from: startOnMesh, to: endOnMesh) else {
            return nil
        }

        var points: [SIMD3<Float>] = []
        points.reserveCapacity(Int(path.pointCount) + 2)
        for point in path.points() {
            let mujocoPoint = muJoCoPoint(fromRecast: point)
            if let last = points.last, simd_distance(SIMD2(last.x, last.y), SIMD2(mujocoPoint.x, mujocoPoint.y)) < 0.08 {
                continue
            }
            points.append(mujocoPoint)
        }

        let projectedStart = muJoCoPoint(fromRecast: startOnMesh)
        let projectedEnd = muJoCoPoint(fromRecast: endOnMesh)
        if points.first.map({ simd_distance(SIMD2($0.x, $0.y), SIMD2(projectedStart.x, projectedStart.y)) > 0.08 }) ?? true {
            points.insert(projectedStart, at: 0)
        } else {
            points[0] = projectedStart
        }
        if points.last.map({ simd_distance(SIMD2($0.x, $0.y), SIMD2(projectedEnd.x, projectedEnd.y)) > 0.08 }) ?? true {
            points.append(projectedEnd)
        } else {
            points[points.count - 1] = projectedEnd
        }

        points = samplePath(points, spacing: 0.35, using: navMesh)

        while let first = points.first,
              simd_distance(SIMD2(first.x, first.y), SIMD2(start.x, start.y)) < 0.25 {
            points.removeFirst()
        }
        if points.isEmpty {
            points.append(projectedEnd)
        }
        return points
    }

    @discardableResult
    private func rebuildIfNeeded(from mesh: NavigationMeshDescriptor?) -> Bool {
        guard let mesh, mesh.vertices.count >= 3, mesh.indices.count >= 3 else {
            clear()
            return false
        }
        if sourceRevision == mesh.revision, navMesh != nil {
            return true
        }

        var vertices: [Float] = []
        vertices.reserveCapacity(mesh.vertices.count * 3)
        for vertex in mesh.vertices {
            let recast = recastPoint(fromMuJoCo: vertex)
            vertices.append(contentsOf: [recast.x, recast.y, recast.z])
        }

        var indices: [Int32] = []
        indices.reserveCapacity(mesh.indices.count)
        for faceStart in stride(from: 0, to: mesh.indices.count - 2, by: 3) {
            let a = mesh.indices[faceStart]
            let b = mesh.indices[faceStart + 1]
            let c = mesh.indices[faceStart + 2]
            guard a <= UInt32(Int32.max), b <= UInt32(Int32.max), c <= UInt32(Int32.max) else {
                continue
            }
            indices.append(contentsOf: [Int32(a), Int32(c), Int32(b)])
        }

        let agentRadius: Float = 0.28
        let config = RCNavMeshConfig.defaultConfig(withAgentHeight: 0.45, radius: agentRadius, climb: 0.25)
        config.cellSize = 0.10
        config.cellHeight = 0.05
        config.walkableSlopeAngle = 50
        config.walkableHeight = Int32(ceilf(0.45 / config.cellHeight))
        config.walkableClimb = Int32(floorf(0.25 / config.cellHeight))
        config.walkableRadius = Int32(ceilf(agentRadius / config.cellSize))
        config.minRegionArea = 0
        config.mergeRegionArea = 0

        var buildError: NSError?
        navMesh = vertices.withUnsafeBufferPointer { vertexBuffer in
            indices.withUnsafeBufferPointer { indexBuffer in
                guard let vertexBase = vertexBuffer.baseAddress,
                      let indexBase = indexBuffer.baseAddress else {
                    return nil
                }
                return RCNavMeshBuilder.buildNavMesh(
                    withVertices: vertexBase,
                    vertexCount: Int32(vertices.count / 3),
                    indices: indexBase,
                    indexCount: Int32(indices.count),
                    config: config,
                    error: &buildError
                )
            }
        }
        if let buildError {
            print("Recast navmesh build failed: \(buildError)")
        }

        sourceRevision = navMesh == nil ? nil : mesh.revision
        debugMesh = nil
        return navMesh != nil
    }

    private func makeDebugMesh(in scene: MuJoCoRenderScene) -> RenderMeshDescriptor? {
        guard rebuildIfNeeded(from: scene.navigationMesh), let navMesh else {
            return nil
        }
        if let debugMesh, debugMesh.revision == scene.navigationMesh?.revision {
            return debugMesh
        }

        var vertexCount: Int32 = 0
        let data = navMesh.navMeshTriangleVertices(withVertexCount: &vertexCount)
        guard vertexCount >= 3, data.count >= Int(vertexCount) * 3 * MemoryLayout<Float>.stride else {
            print("Recast navmesh debug mesh is empty")
            return nil
        }

        let vertices = data.withUnsafeBytes { rawBuffer -> [SIMD3<Float>] in
            guard let base = rawBuffer.bindMemory(to: Float.self).baseAddress else {
                return []
            }
            return (0..<Int(vertexCount)).map { index in
                let recast = SIMD3<Float>(
                    base[index * 3 + 0],
                    base[index * 3 + 1],
                    base[index * 3 + 2]
                )
                return muJoCoPoint(fromRecast: recast) + SIMD3<Float>(0, 0, 0.035)
            }
        }
        guard vertices.count >= 3 else {
            return nil
        }

        let indices = (0..<vertices.count).map { UInt32($0) }
        let mesh = RenderMeshDescriptor(
            vertices: vertices,
            normals: makeTriangleNormals(vertices: vertices),
            indices: indices,
            color: SIMD4<Float>(0.05, 0.95, 0.58, 0.18),
            revision: scene.navigationMesh?.revision ?? -100
        )
        debugMesh = mesh
        return mesh
    }

    private func recastPoint(fromMuJoCo point: SIMD3<Float>) -> SIMD3<Float> {
        SIMD3<Float>(point.x, point.z, point.y)
    }

    private func muJoCoPoint(fromRecast point: SIMD3<Float>) -> SIMD3<Float> {
        SIMD3<Float>(point.x, point.z, point.y)
    }

    private func samplePath(
        _ corners: [SIMD3<Float>],
        spacing: Float,
        using navMesh: RecastNavMesh
    ) -> [SIMD3<Float>] {
        guard corners.count >= 2 else {
            return corners
        }

        var sampled = [corners[0]]
        for index in 0..<(corners.count - 1) {
            let start = corners[index]
            let end = corners[index + 1]
            let distance = simd_distance(SIMD2(start.x, start.y), SIMD2(end.x, end.y))
            let stepCount = max(1, Int(ceil(distance / max(spacing, 0.05))))
            for step in 1...stepCount {
                let t = Float(step) / Float(stepCount)
                let point = start + (end - start) * t
                let projected = muJoCoPoint(fromRecast: navMesh.findNearestPoint(recastPoint(fromMuJoCo: point)))
                if let last = sampled.last,
                   simd_distance(SIMD2(last.x, last.y), SIMD2(projected.x, projected.y)) < 0.08 {
                    continue
                }
                sampled.append(projected)
            }
        }
        return sampled
    }

    private func makeTriangleNormals(vertices: [SIMD3<Float>]) -> [SIMD3<Float>] {
        var normals = Array(repeating: SIMD3<Float>(0, 0, 1), count: vertices.count)
        for faceStart in stride(from: 0, to: vertices.count - 2, by: 3) {
            let a = vertices[faceStart]
            let b = vertices[faceStart + 1]
            let c = vertices[faceStart + 2]
            let normal = simd_cross(b - a, c - a)
            let normalized = length_squared(normal) > 0.000001
                ? normalize(normal)
                : SIMD3<Float>(0, 0, 1)
            normals[faceStart] = normalized
            normals[faceStart + 1] = normalized
            normals[faceStart + 2] = normalized
        }
        return normals
    }
}
