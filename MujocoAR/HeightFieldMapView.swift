//
//  HeightFieldMapView.swift
//  MujocoAR
//
//  Top-down 2D map view that visualises the current MuJoCo environment mesh
//  (typically derived from the AR heightfield) and lets the user tap to pick a
//  far-away navigation target without having to physically point the device at
//  the location.
//

import UIKit
import simd

@MainActor
final class HeightFieldMapView: UIView {
    /// Called when the user taps on the map. Provides a world-space (MuJoCo)
    /// point with `z` estimated from the nearest mesh vertex.
    var onTap: ((SIMD3<Float>) -> Void)?

    /// Latest navigation target marker to render on the map (optional).
    var navigationTarget: SIMD3<Float>? {
        didSet { setNeedsDisplay() }
    }

    /// Latest navigation path waypoints to render on the map (optional).
    var navigationPath: [SIMD3<Float>]? {
        didSet { setNeedsDisplay() }
    }

    /// Whether to overlay the navigation debug mesh (semi-transparent blue).
    var showsNavigationOverlay: Bool = false {
        didSet { setNeedsDisplay() }
    }

    var mapBackgroundColor: UIColor = UIColor(white: 0.05, alpha: 1.0) {
        didSet {
            backgroundColor = mapBackgroundColor
            isOpaque = mapBackgroundColor.cgColor.alpha >= 1.0
            setNeedsDisplay()
        }
    }

    private var scene: MuJoCoRenderScene = .empty
    private var cachedTerrainImage: UIImage?
    private var cachedTerrainKey: TerrainCacheKey?
    private var sceneBounds: SceneBounds?

    private struct SceneBounds {
        let minXY: SIMD2<Float>
        let maxXY: SIMD2<Float>
        let minZ: Float
        let maxZ: Float
    }

    private struct TerrainCacheKey: Equatable {
        let environmentRevision: Int
        let navigationRevision: Int
        let showsNavigation: Bool
        let width: CGFloat
        let height: CGFloat
        let scale: CGFloat
    }

    override init(frame: CGRect) {
        super.init(frame: frame)
        backgroundColor = mapBackgroundColor
        isOpaque = mapBackgroundColor.cgColor.alpha >= 1.0
        contentMode = .redraw
        let tap = UITapGestureRecognizer(target: self, action: #selector(handleTap(_:)))
        addGestureRecognizer(tap)
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    override var bounds: CGRect {
        didSet {
            if bounds.size != oldValue.size {
                cachedTerrainImage = nil
                cachedTerrainKey = nil
                setNeedsDisplay()
            }
        }
    }

    func updateScene(_ scene: MuJoCoRenderScene) {
        let oldEnvRev = self.scene.environmentMesh?.revision ?? -1
        let oldNavRev = self.scene.navigationDebugMesh?.revision ?? -1
        self.scene = scene
        let newEnvRev = scene.environmentMesh?.revision ?? -1
        let newNavRev = scene.navigationDebugMesh?.revision ?? -1
        if oldEnvRev != newEnvRev || oldNavRev != newNavRev {
            cachedTerrainImage = nil
            cachedTerrainKey = nil
        }
        setNeedsDisplay()
    }

    @objc private func handleTap(_ gesture: UITapGestureRecognizer) {
        guard let bounds = sceneBounds else { return }
        let location = gesture.location(in: self)
        let drawRect = mapRect(for: bounds)
        guard drawRect.width > 0, drawRect.height > 0 else { return }

        let nx = Float((location.x - drawRect.minX) / drawRect.width)
        let ny = Float((location.y - drawRect.minY) / drawRect.height)
        guard nx >= 0, nx <= 1, ny >= 0, ny <= 1 else { return }

        let worldX = bounds.minXY.x + nx * (bounds.maxXY.x - bounds.minXY.x)
        // Y is flipped: top of view = max world Y.
        let worldY = bounds.maxXY.y - ny * (bounds.maxXY.y - bounds.minXY.y)
        let xy = SIMD2<Float>(worldX, worldY)
        let z = sampleHeight(at: xy) ?? scene.robotPose?.position.z ?? 0
        onTap?(SIMD3<Float>(xy.x, xy.y, z))
    }

    // MARK: - Drawing

    override func draw(_ rect: CGRect) {
        guard let ctx = UIGraphicsGetCurrentContext() else { return }
        ctx.setFillColor(mapBackgroundColor.cgColor)
        ctx.fill(self.bounds)

        guard let mesh = scene.environmentMesh,
              mesh.vertices.count >= 3,
              mesh.indices.count >= 3 else {
            sceneBounds = nil
            drawCenteredMessage("No terrain mesh yet.\nScan the environment in AR mode.", in: ctx)
            return
        }

        let bounds = computeBounds(from: mesh)
        sceneBounds = bounds

        let scale = window?.screen.scale ?? UIScreen.main.scale
        let cacheKey = TerrainCacheKey(
            environmentRevision: mesh.revision,
            navigationRevision: scene.navigationDebugMesh?.revision ?? -1,
            showsNavigation: showsNavigationOverlay,
            width: self.bounds.width,
            height: self.bounds.height,
            scale: scale
        )

        if cachedTerrainImage == nil || cachedTerrainKey != cacheKey {
            cachedTerrainImage = renderTerrainImage(bounds: bounds, mesh: mesh, scale: scale)
            cachedTerrainKey = cacheKey
        }

        if let image = cachedTerrainImage {
            image.draw(in: self.bounds)
        }

        drawNavigationPath(ctx: ctx, sceneBounds: bounds)
        drawNavigationTarget(ctx: ctx, sceneBounds: bounds)
        drawRobot(ctx: ctx, sceneBounds: bounds)
        drawScaleBar(ctx: ctx, sceneBounds: bounds)
    }

    private func drawCenteredMessage(_ message: String, in ctx: CGContext) {
        let para = NSMutableParagraphStyle()
        para.alignment = .center
        let text = NSAttributedString(
            string: message,
            attributes: [
                .foregroundColor: UIColor.white.withAlphaComponent(0.8),
                .font: UIFont.systemFont(ofSize: 15, weight: .medium),
                .paragraphStyle: para,
            ]
        )
        let size = text.size()
        let origin = CGPoint(x: self.bounds.midX - size.width / 2,
                             y: self.bounds.midY - size.height / 2)
        text.draw(at: origin)
    }

    private func renderTerrainImage(
        bounds: SceneBounds,
        mesh: RenderMeshDescriptor,
        scale: CGFloat
    ) -> UIImage? {
        let format = UIGraphicsImageRendererFormat()
        format.scale = scale
        format.opaque = false
        let renderer = UIGraphicsImageRenderer(size: self.bounds.size, format: format)
        return renderer.image { rendererCtx in
            let ctx = rendererCtx.cgContext
            let zRange = max(bounds.maxZ - bounds.minZ, 0.0001)
            let vertices = mesh.vertices
            let indices = mesh.indices
            var i = 0
            while i + 2 < indices.count {
                let i0 = Int(indices[i])
                let i1 = Int(indices[i + 1])
                let i2 = Int(indices[i + 2])
                i += 3
                guard i0 < vertices.count, i1 < vertices.count, i2 < vertices.count else { continue }
                let v0 = vertices[i0]
                let v1 = vertices[i1]
                let v2 = vertices[i2]
                let zAvg = (v0.z + v1.z + v2.z) / 3
                let t = (zAvg - bounds.minZ) / zRange
                ctx.setFillColor(heightColor(t).cgColor)
                let p0 = project(SIMD2(v0.x, v0.y), sceneBounds: bounds)
                let p1 = project(SIMD2(v1.x, v1.y), sceneBounds: bounds)
                let p2 = project(SIMD2(v2.x, v2.y), sceneBounds: bounds)
                ctx.beginPath()
                ctx.move(to: p0)
                ctx.addLine(to: p1)
                ctx.addLine(to: p2)
                ctx.closePath()
                ctx.fillPath()
            }

            if showsNavigationOverlay,
               let nav = scene.navigationDebugMesh,
               nav.vertices.count >= 3,
               nav.indices.count >= 3 {
                ctx.setFillColor(UIColor.systemBlue.withAlphaComponent(0.30).cgColor)
                ctx.setStrokeColor(UIColor.systemBlue.withAlphaComponent(0.7).cgColor)
                ctx.setLineWidth(0.5)
                var k = 0
                while k + 2 < nav.indices.count {
                    let i0 = Int(nav.indices[k])
                    let i1 = Int(nav.indices[k + 1])
                    let i2 = Int(nav.indices[k + 2])
                    k += 3
                    guard i0 < nav.vertices.count, i1 < nav.vertices.count, i2 < nav.vertices.count else { continue }
                    let v0 = nav.vertices[i0]
                    let v1 = nav.vertices[i1]
                    let v2 = nav.vertices[i2]
                    let p0 = project(SIMD2(v0.x, v0.y), sceneBounds: bounds)
                    let p1 = project(SIMD2(v1.x, v1.y), sceneBounds: bounds)
                    let p2 = project(SIMD2(v2.x, v2.y), sceneBounds: bounds)
                    ctx.beginPath()
                    ctx.move(to: p0)
                    ctx.addLine(to: p1)
                    ctx.addLine(to: p2)
                    ctx.closePath()
                    ctx.drawPath(using: .fillStroke)
                }
            }
        }
    }

    private func drawNavigationPath(ctx: CGContext, sceneBounds: SceneBounds) {
        guard let path = navigationPath, path.count >= 2 else { return }
        ctx.saveGState()
        ctx.setStrokeColor(UIColor.systemGreen.cgColor)
        ctx.setLineWidth(3)
        ctx.setLineJoin(.round)
        ctx.setLineCap(.round)
        ctx.beginPath()
        let p0 = project(SIMD2(path[0].x, path[0].y), sceneBounds: sceneBounds)
        ctx.move(to: p0)
        for i in 1..<path.count {
            let p = project(SIMD2(path[i].x, path[i].y), sceneBounds: sceneBounds)
            ctx.addLine(to: p)
        }
        ctx.strokePath()

        ctx.setFillColor(UIColor.systemGreen.cgColor)
        for waypoint in path {
            let p = project(SIMD2(waypoint.x, waypoint.y), sceneBounds: sceneBounds)
            ctx.fillEllipse(in: CGRect(x: p.x - 2.5, y: p.y - 2.5, width: 5, height: 5))
        }
        ctx.restoreGState()
    }

    private func drawNavigationTarget(ctx: CGContext, sceneBounds: SceneBounds) {
        guard let target = navigationTarget else { return }
        let p = project(SIMD2(target.x, target.y), sceneBounds: sceneBounds)
        ctx.saveGState()
        ctx.setStrokeColor(UIColor.systemRed.cgColor)
        ctx.setFillColor(UIColor.systemRed.withAlphaComponent(0.35).cgColor)
        ctx.setLineWidth(2)
        let r: CGFloat = 12
        ctx.addEllipse(in: CGRect(x: p.x - r, y: p.y - r, width: 2 * r, height: 2 * r))
        ctx.drawPath(using: .fillStroke)

        ctx.setStrokeColor(UIColor.systemRed.cgColor)
        ctx.setLineWidth(2)
        ctx.move(to: CGPoint(x: p.x - r - 4, y: p.y))
        ctx.addLine(to: CGPoint(x: p.x + r + 4, y: p.y))
        ctx.move(to: CGPoint(x: p.x, y: p.y - r - 4))
        ctx.addLine(to: CGPoint(x: p.x, y: p.y + r + 4))
        ctx.strokePath()
        ctx.restoreGState()
    }

    private func drawRobot(ctx: CGContext, sceneBounds: SceneBounds) {
        guard let pose = scene.robotPose else { return }
        let p = project(SIMD2(pose.position.x, pose.position.y), sceneBounds: sceneBounds)
        let r: CGFloat = 9
        ctx.saveGState()
        ctx.setFillColor(UIColor.systemYellow.cgColor)
        ctx.setStrokeColor(UIColor.black.cgColor)
        ctx.setLineWidth(1.5)
        ctx.addEllipse(in: CGRect(x: p.x - r, y: p.y - r, width: 2 * r, height: 2 * r))
        ctx.drawPath(using: .fillStroke)

        // Heading arrow. In MuJoCo (Z-up), yaw rotates around Z; world heading
        // direction projected to ground plane is (cos(yaw), sin(yaw)).
        let yaw = pose.yaw
        let dirX = cos(yaw)
        let dirY = sin(yaw)
        let arrowMeters: Float = 0.5
        let tipWorld = SIMD2<Float>(pose.position.x + dirX * arrowMeters,
                                    pose.position.y + dirY * arrowMeters)
        let tip = project(tipWorld, sceneBounds: sceneBounds)
        ctx.setStrokeColor(UIColor.systemYellow.cgColor)
        ctx.setLineWidth(3)
        ctx.move(to: p)
        ctx.addLine(to: tip)
        ctx.strokePath()
        ctx.restoreGState()
    }

    private func drawScaleBar(ctx: CGContext, sceneBounds: SceneBounds) {
        let drawRect = mapRect(for: sceneBounds)
        let widthMeters = sceneBounds.maxXY.x - sceneBounds.minXY.x
        guard widthMeters > 0, drawRect.width > 0 else { return }
        let pxPerMeter = drawRect.width / CGFloat(widthMeters)

        // Pick a "nice" length close to ~15% of the draw width.
        let targetMeters = Float(drawRect.width * 0.15 / pxPerMeter)
        let niceMeters = niceScaleLength(targetMeters)
        let lengthPx = CGFloat(niceMeters) * pxPerMeter

        let y = drawRect.maxY - 16
        let x0 = drawRect.minX + 12

        ctx.saveGState()
        ctx.setStrokeColor(UIColor.white.cgColor)
        ctx.setLineWidth(2)
        ctx.move(to: CGPoint(x: x0, y: y))
        ctx.addLine(to: CGPoint(x: x0 + lengthPx, y: y))
        ctx.move(to: CGPoint(x: x0, y: y - 4))
        ctx.addLine(to: CGPoint(x: x0, y: y + 4))
        ctx.move(to: CGPoint(x: x0 + lengthPx, y: y - 4))
        ctx.addLine(to: CGPoint(x: x0 + lengthPx, y: y + 4))
        ctx.strokePath()

        let label = String(format: niceMeters >= 1 ? "%.0f m" : "%.1f m", niceMeters) as NSString
        label.draw(at: CGPoint(x: x0, y: y - 18), withAttributes: [
            .foregroundColor: UIColor.white,
            .font: UIFont.systemFont(ofSize: 11, weight: .semibold),
        ])
        ctx.restoreGState()
    }

    private func niceScaleLength(_ value: Float) -> Float {
        guard value > 0 else { return 1 }
        let exponent = floor(log10(value))
        let base = pow(10, exponent)
        let normalised = value / base
        let snap: Float
        if normalised < 1.5 { snap = 1 }
        else if normalised < 3.5 { snap = 2 }
        else if normalised < 7.5 { snap = 5 }
        else { snap = 10 }
        return snap * base
    }

    // MARK: - Helpers

    private func computeBounds(from mesh: RenderMeshDescriptor) -> SceneBounds {
        var minXY = SIMD2<Float>(.greatestFiniteMagnitude, .greatestFiniteMagnitude)
        var maxXY = SIMD2<Float>(-.greatestFiniteMagnitude, -.greatestFiniteMagnitude)
        var minZ: Float = .greatestFiniteMagnitude
        var maxZ: Float = -.greatestFiniteMagnitude
        for v in mesh.vertices {
            if v.x < minXY.x { minXY.x = v.x }
            if v.y < minXY.y { minXY.y = v.y }
            if v.x > maxXY.x { maxXY.x = v.x }
            if v.y > maxXY.y { maxXY.y = v.y }
            if v.z < minZ { minZ = v.z }
            if v.z > maxZ { maxZ = v.z }
        }
        // Include robot position so it never falls off the map.
        if let pose = scene.robotPose {
            minXY.x = min(minXY.x, pose.position.x)
            minXY.y = min(minXY.y, pose.position.y)
            maxXY.x = max(maxXY.x, pose.position.x)
            maxXY.y = max(maxXY.y, pose.position.y)
        }
        let pad: Float = 0.5
        minXY -= SIMD2<Float>(pad, pad)
        maxXY += SIMD2<Float>(pad, pad)
        return SceneBounds(minXY: minXY, maxXY: maxXY, minZ: minZ, maxZ: maxZ)
    }

    private func mapRect(for bounds: SceneBounds) -> CGRect {
        let inset: CGFloat = 16
        let avail = self.bounds.insetBy(dx: inset, dy: inset)
        let w = CGFloat(bounds.maxXY.x - bounds.minXY.x)
        let h = CGFloat(bounds.maxXY.y - bounds.minXY.y)
        guard w > 0, h > 0, avail.width > 0, avail.height > 0 else { return avail }
        let scale = min(avail.width / w, avail.height / h)
        let outW = w * scale
        let outH = h * scale
        return CGRect(x: avail.midX - outW / 2,
                      y: avail.midY - outH / 2,
                      width: outW, height: outH)
    }

    private func project(_ p: SIMD2<Float>, sceneBounds: SceneBounds) -> CGPoint {
        let drawRect = mapRect(for: sceneBounds)
        let rangeX = max(sceneBounds.maxXY.x - sceneBounds.minXY.x, 0.0001)
        let rangeY = max(sceneBounds.maxXY.y - sceneBounds.minXY.y, 0.0001)
        let nx = (p.x - sceneBounds.minXY.x) / rangeX
        let ny = 1 - (p.y - sceneBounds.minXY.y) / rangeY
        return CGPoint(x: drawRect.minX + CGFloat(nx) * drawRect.width,
                       y: drawRect.minY + CGFloat(ny) * drawRect.height)
    }

    private func sampleHeight(at xy: SIMD2<Float>) -> Float? {
        guard let mesh = scene.environmentMesh, !mesh.vertices.isEmpty else { return nil }
        // Nearest-vertex sampling is sufficient for tap targets (the navigator
        // re-projects the point onto the nav mesh anyway).
        var bestSqDist: Float = .greatestFiniteMagnitude
        var bestZ: Float = 0
        for v in mesh.vertices {
            let dx = v.x - xy.x
            let dy = v.y - xy.y
            let sq = dx * dx + dy * dy
            if sq < bestSqDist {
                bestSqDist = sq
                bestZ = v.z
            }
        }
        return bestZ
    }

    private func heightColor(_ t: Float) -> UIColor {
        let tc = max(0, min(1, t))
        // Blue (low) -> Cyan -> Green -> Yellow -> Red (high).
        let r: Float
        let g: Float
        let b: Float
        if tc < 0.25 {
            let k = tc / 0.25
            r = 0; g = k; b = 1
        } else if tc < 0.5 {
            let k = (tc - 0.25) / 0.25
            r = 0; g = 1; b = 1 - k
        } else if tc < 0.75 {
            let k = (tc - 0.5) / 0.25
            r = k; g = 1; b = 0
        } else {
            let k = (tc - 0.75) / 0.25
            r = 1; g = 1 - k; b = 0
        }
        return UIColor(red: CGFloat(r), green: CGFloat(g), blue: CGFloat(b), alpha: 1.0)
    }
}
