//
//  Go1PolicyController.swift
//  MujocoAR
//
//  Created by Tatsuya Ogawa on 2026/05/29.
//

import CoreML
import Foundation
import MuJoCo
import simd

nonisolated private struct JointAddress: Sendable {
    let qpos: Int
    let qvel: Int
}

nonisolated struct LocomotionCommandLimits: Sendable {
    let forwardMin: Float
    let forwardMax: Float
    let lateralMin: Float
    let lateralMax: Float
    let yawRateMin: Float
    let yawRateMax: Float
}

nonisolated struct LocomotionPolicySpec: Sendable {
    let logName: String
    let modelName: String
    let rootJointName: String
    let terrainScanBodyName: String
    let terrainScanExcludeBodyName: String
    let linearVelocitySensorName: String
    let angularVelocitySensorName: String
    let jointNames: [String]
    let actuatorNames: [String]
    let defaultJointPos: [Double]
    let actionScale: [Double]
    let commandDefault: SIMD3<Float>
    let commandLimits: LocomotionCommandLimits
    let inputSize: Int
    let outputSize: Int
    let heightScanSizeX: Double
    let heightScanSizeY: Double
    let heightScanResolution: Double
    let heightScanMaxDistance: Double
    let navigationForwardWhileTurning: Float
    let usesHolonomicNavigation: Bool
}

nonisolated enum LocomotionRobotKind: Int, CaseIterable, Sendable {
    case go1 = 0
    case g1 = 1

    var displayName: String {
        switch self {
        case .go1:
            return "Go1"
        case .g1:
            return "G1"
        }
    }

    var flatSceneResourceName: String {
        switch self {
        case .go1:
            return "go1_flat_scene"
        case .g1:
            return "g1_flat_scene"
        }
    }

    var roughSceneResourceName: String {
        switch self {
        case .go1:
            return "go1_rough_scene"
        case .g1:
            return "g1_rough_scene"
        }
    }

    var renderManifestResourceName: String {
        switch self {
        case .go1:
            return "go1_render_manifest"
        case .g1:
            return "g1_render_manifest"
        }
    }

    var rootStartPosition: SIMD3<Double> {
        switch self {
        case .go1:
            return SIMD3<Double>(0, 0, 0.278)
        case .g1:
            return SIMD3<Double>(0, 0, 0.76)
        }
    }

    var spawnHeightAboveSurface: Float {
        switch self {
        case .go1:
            return 0.30
        case .g1:
            return 0.76
        }
    }

    var arStartHeightAboveSurface: Float {
        switch self {
        case .go1:
            return 0.55
        case .g1:
            return 0.90
        }
    }

    var fallbackRootHeightOffset: Float {
        switch self {
        case .go1:
            return 0.38
        case .g1:
            return 0.76
        }
    }

    var policySpec: LocomotionPolicySpec {
        switch self {
        case .go1:
            return LocomotionPolicySpec(
                logName: "Go1",
                modelName: "go1_velocity_rough",
                rootJointName: "robot/floating_base_joint",
                terrainScanBodyName: "robot/trunk",
                terrainScanExcludeBodyName: "robot/trunk",
                linearVelocitySensorName: "robot/imu_lin_vel",
                angularVelocitySensorName: "robot/imu_ang_vel",
                jointNames: [
                    "robot/FR_hip_joint",
                    "robot/FR_thigh_joint",
                    "robot/FR_calf_joint",
                    "robot/FL_hip_joint",
                    "robot/FL_thigh_joint",
                    "robot/FL_calf_joint",
                    "robot/RR_hip_joint",
                    "robot/RR_thigh_joint",
                    "robot/RR_calf_joint",
                    "robot/RL_hip_joint",
                    "robot/RL_thigh_joint",
                    "robot/RL_calf_joint",
                ],
                actuatorNames: [
                    "robot/FR_hip_joint",
                    "robot/FR_thigh_joint",
                    "robot/FL_hip_joint",
                    "robot/FL_thigh_joint",
                    "robot/RR_hip_joint",
                    "robot/RR_thigh_joint",
                    "robot/RL_hip_joint",
                    "robot/RL_thigh_joint",
                    "robot/FR_calf_joint",
                    "robot/FL_calf_joint",
                    "robot/RR_calf_joint",
                    "robot/RL_calf_joint",
                ],
                defaultJointPos: [
                    0.1, 0.9, -1.8,
                    -0.1, 0.9, -1.8,
                    0.1, 0.9, -1.8,
                    -0.1, 0.9, -1.8,
                ],
                actionScale: [
                    0.37275403895515624,
                    0.37275403895515624,
                    0.2485019978022777,
                    0.37275403895515624,
                    0.37275403895515624,
                    0.2485019978022777,
                    0.37275403895515624,
                    0.37275403895515624,
                    0.2485019978022777,
                    0.37275403895515624,
                    0.37275403895515624,
                    0.2485019978022777,
                ],
                commandDefault: SIMD3<Float>(0.8, 0, 0),
                commandLimits: LocomotionCommandLimits(
                    forwardMin: -0.35,
                    forwardMax: 1.0,
                    lateralMin: -0.45,
                    lateralMax: 0.45,
                    yawRateMin: -1.6,
                    yawRateMax: 1.6
                ),
                inputSize: 48 + 187,
                outputSize: 12,
                heightScanSizeX: 1.6,
                heightScanSizeY: 1.0,
                heightScanResolution: 0.1,
                heightScanMaxDistance: 5.0,
                navigationForwardWhileTurning: 0.0,
                usesHolonomicNavigation: false
            )
        case .g1:
            return LocomotionPolicySpec(
                logName: "G1",
                modelName: "g1_velocity_rough",
                rootJointName: "robot/floating_base_joint",
                terrainScanBodyName: "robot/pelvis",
                terrainScanExcludeBodyName: "robot/pelvis",
                linearVelocitySensorName: "robot/imu_lin_vel",
                angularVelocitySensorName: "robot/imu_ang_vel",
                jointNames: [
                    "robot/left_hip_pitch_joint",
                    "robot/left_hip_roll_joint",
                    "robot/left_hip_yaw_joint",
                    "robot/left_knee_joint",
                    "robot/left_ankle_pitch_joint",
                    "robot/left_ankle_roll_joint",
                    "robot/right_hip_pitch_joint",
                    "robot/right_hip_roll_joint",
                    "robot/right_hip_yaw_joint",
                    "robot/right_knee_joint",
                    "robot/right_ankle_pitch_joint",
                    "robot/right_ankle_roll_joint",
                    "robot/waist_yaw_joint",
                    "robot/waist_roll_joint",
                    "robot/waist_pitch_joint",
                    "robot/left_shoulder_pitch_joint",
                    "robot/left_shoulder_roll_joint",
                    "robot/left_shoulder_yaw_joint",
                    "robot/left_elbow_joint",
                    "robot/left_wrist_roll_joint",
                    "robot/left_wrist_pitch_joint",
                    "robot/left_wrist_yaw_joint",
                    "robot/right_shoulder_pitch_joint",
                    "robot/right_shoulder_roll_joint",
                    "robot/right_shoulder_yaw_joint",
                    "robot/right_elbow_joint",
                    "robot/right_wrist_roll_joint",
                    "robot/right_wrist_pitch_joint",
                    "robot/right_wrist_yaw_joint",
                ],
                actuatorNames: [
                    "robot/left_hip_pitch_joint",
                    "robot/left_hip_roll_joint",
                    "robot/left_hip_yaw_joint",
                    "robot/left_knee_joint",
                    "robot/left_ankle_pitch_joint",
                    "robot/left_ankle_roll_joint",
                    "robot/right_hip_pitch_joint",
                    "robot/right_hip_roll_joint",
                    "robot/right_hip_yaw_joint",
                    "robot/right_knee_joint",
                    "robot/right_ankle_pitch_joint",
                    "robot/right_ankle_roll_joint",
                    "robot/waist_yaw_joint",
                    "robot/waist_roll_joint",
                    "robot/waist_pitch_joint",
                    "robot/left_shoulder_pitch_joint",
                    "robot/left_shoulder_roll_joint",
                    "robot/left_shoulder_yaw_joint",
                    "robot/left_elbow_joint",
                    "robot/left_wrist_roll_joint",
                    "robot/left_wrist_pitch_joint",
                    "robot/left_wrist_yaw_joint",
                    "robot/right_shoulder_pitch_joint",
                    "robot/right_shoulder_roll_joint",
                    "robot/right_shoulder_yaw_joint",
                    "robot/right_elbow_joint",
                    "robot/right_wrist_roll_joint",
                    "robot/right_wrist_pitch_joint",
                    "robot/right_wrist_yaw_joint",
                ],
                defaultJointPos: [
                    -0.31200000643730164, 0, 0, 0.6690000295639038, -0.3630000054836273, 0,
                    -0.31200000643730164, 0, 0, 0.6690000295639038, -0.3630000054836273, 0,
                    0, 0, 0,
                    0.20000000298023224, 0.20000000298023224, 0, 0.6000000238418579, 0, 0, 0,
                    0.20000000298023224, -0.20000000298023224, 0, 0.6000000238418579, 0, 0, 0,
                ],
                actionScale: [
                    0.5475464463233948, 0.3506614565849304, 0.5475464463233948, 0.3506614565849304,
                    0.4385773241519928, 0.4385773241519928, 0.5475464463233948, 0.3506614565849304,
                    0.5475464463233948, 0.3506614565849304, 0.4385773241519928, 0.4385773241519928,
                    0.5475464463233948, 0.4385773241519928, 0.4385773241519928, 0.4385773241519928,
                    0.4385773241519928, 0.4385773241519928, 0.4385773241519928, 0.4385773241519928,
                    0.07450087368488312, 0.07450087368488312, 0.4385773241519928, 0.4385773241519928,
                    0.4385773241519928, 0.4385773241519928, 0.4385773241519928, 0.07450087368488312,
                    0.07450087368488312,
                ],
                commandDefault: SIMD3<Float>(0.8, 0, 0),
                commandLimits: LocomotionCommandLimits(
                    forwardMin: -1.5,
                    forwardMax: 2.0,
                    lateralMin: -1.0,
                    lateralMax: 1.0,
                    yawRateMin: -0.7,
                    yawRateMax: 0.7
                ),
                inputSize: 99 + 187,
                outputSize: 29,
                heightScanSizeX: 1.6,
                heightScanSizeY: 1.0,
                heightScanResolution: 0.1,
                heightScanMaxDistance: 5.0,
                navigationForwardWhileTurning: 0.22,
                usesHolonomicNavigation: true
            )
        }
    }
}

nonisolated final class Go1PolicyController {
    struct State: Sendable {
        let rootStartPosition: SIMD3<Double>
        let lastAction: [Float]
        let execAction: [Float]
        let targetJointPos: [Double]
        let controlAccumulator: Double
        let hasReportedPolicyFailure: Bool
        let velocityCommand: VelocityCommand
        let navigationPath: [SIMD3<Float>]
    }

    struct VelocityCommand: Sendable {
        let forward: Float
        let lateral: Float
        let yawRate: Float
    }

    private let model: UnsafeMutablePointer<mjModel>
    private let data: UnsafeMutablePointer<mjData>
    private let jointAddresses: [JointAddress]
    private let actuatorCtrlIDs: [Int]
    private let actuatorToJointIndex: [Int]
    private let rootQposAddress: Int
    private let rootQvelAddress: Int
    private let linVelSensorAddress: Int
    private let angVelSensorAddress: Int
    private let heightScanBodyID: Int
    private let heightScanExcludeBodyID: Int
    private let heightScanOffsets: [(x: Double, y: Double)]
    private let policy: CoreMLPolicy?
    private let spec: LocomotionPolicySpec

    private var rootStartPosition = SIMD3<Double>(0, 0, 0.55)
    private var observation: [Float] = []
    private var lastAction: [Float] = []
    private var execAction: [Float] = []
    private var targetJointPos: [Double] = []
    private var controlAccumulator: Double = 0
    private var hasReportedPolicyFailure = false
    private var velocityCommand = VelocityCommand(forward: 0, lateral: 0, yawRate: 0)
    private(set) var navigationPath: [SIMD3<Float>] = []
    var navigationTarget: SIMD3<Float>? { navigationPath.first }

    private static let controlDt = 0.02
    private static let decimation = 4
    private static let navigationWaypointRadius: Float = 0.34
    private static let navigationLookaheadDistance: Float = 0.48
    private let robotScale: Double
    private var robotScaleFloat: Float { Float(robotScale) }

    init?(
        model: UnsafeMutablePointer<mjModel>,
        data: UnsafeMutablePointer<mjData>,
        robot: LocomotionRobotKind,
        robotScale: Float = 1.0
    ) {
        let spec = robot.policySpec
        self.spec = spec
        self.model = model
        self.data = data
        self.robotScale = Double(max(robotScale, 0.01))
        observation = [Float](repeating: 0, count: spec.inputSize)
        lastAction = [Float](repeating: 0, count: spec.outputSize)
        execAction = [Float](repeating: 0, count: spec.outputSize)
        targetJointPos = spec.defaultJointPos
        velocityCommand = VelocityCommand(
            forward: spec.commandDefault.x,
            lateral: spec.commandDefault.y,
            yawRate: spec.commandDefault.z
        )

        guard let rootJointID = Self.objectID(model, type: mjOBJ_JOINT, name: spec.rootJointName),
              let linSensorID = Self.objectID(model, type: mjOBJ_SENSOR, name: spec.linearVelocitySensorName),
              let angSensorID = Self.objectID(model, type: mjOBJ_SENSOR, name: spec.angularVelocitySensorName),
              let scanBodyID = Self.objectID(model, type: mjOBJ_BODY, name: spec.terrainScanBodyName),
              let excludeBodyID = Self.objectID(model, type: mjOBJ_BODY, name: spec.terrainScanExcludeBodyName) else {
            print("\(spec.logName) policy setup failed: required MuJoCo names are missing")
            return nil
        }

        let jointAddresses = spec.jointNames.compactMap { name -> JointAddress? in
            guard let jointID = Self.objectID(model, type: mjOBJ_JOINT, name: name) else {
                print("\(spec.logName) policy setup failed: joint not found \(name)")
                return nil
            }
            return JointAddress(
                qpos: Int(model.pointee.jnt_qposadr[jointID]),
                qvel: Int(model.pointee.jnt_dofadr[jointID])
            )
        }
        guard jointAddresses.count == spec.jointNames.count else {
            return nil
        }

        let actuatorCtrlIDs = spec.actuatorNames.compactMap { name -> Int? in
            guard let actuatorID = Self.objectID(model, type: mjOBJ_ACTUATOR, name: name) else {
                print("\(spec.logName) policy setup failed: actuator not found \(name)")
                return nil
            }
            return actuatorID
        }
        guard actuatorCtrlIDs.count == spec.actuatorNames.count else {
            return nil
        }

        let actuatorToJointIndex = spec.actuatorNames.compactMap { spec.jointNames.firstIndex(of: $0) }
        guard actuatorToJointIndex.count == spec.actuatorNames.count else {
            print("\(spec.logName) policy setup failed: actuator/joint order mapping is invalid")
            return nil
        }

        self.jointAddresses = jointAddresses
        self.actuatorCtrlIDs = actuatorCtrlIDs
        self.actuatorToJointIndex = actuatorToJointIndex
        self.rootQposAddress = Int(model.pointee.jnt_qposadr[rootJointID])
        self.rootQvelAddress = Int(model.pointee.jnt_dofadr[rootJointID])
        self.linVelSensorAddress = Int(model.pointee.sensor_adr[linSensorID])
        self.angVelSensorAddress = Int(model.pointee.sensor_adr[angSensorID])
        self.heightScanBodyID = scanBodyID
        self.heightScanExcludeBodyID = excludeBodyID
        self.heightScanOffsets = Self.makeHeightScanOffsets(
            sizeX: spec.heightScanSizeX * self.robotScale,
            sizeY: spec.heightScanSizeY * self.robotScale,
            resolution: spec.heightScanResolution * self.robotScale
        )
        self.policy = CoreMLPolicy.loadOptional(
            modelName: spec.modelName,
            outputSize: spec.outputSize,
            logName: spec.logName
        )
    }

    func reset(rootPosition: SIMD3<Double>) {
        rootStartPosition = rootPosition
        controlAccumulator = 0
        lastAction = [Float](repeating: 0, count: spec.outputSize)
        execAction = [Float](repeating: 0, count: spec.outputSize)
        targetJointPos = spec.defaultJointPos

        data.pointee.qpos[rootQposAddress + 0] = rootPosition.x
        data.pointee.qpos[rootQposAddress + 1] = rootPosition.y
        data.pointee.qpos[rootQposAddress + 2] = rootPosition.z
        data.pointee.qpos[rootQposAddress + 3] = 1
        data.pointee.qpos[rootQposAddress + 4] = 0
        data.pointee.qpos[rootQposAddress + 5] = 0
        data.pointee.qpos[rootQposAddress + 6] = 0

        for i in 0..<6 {
            data.pointee.qvel[rootQvelAddress + i] = 0
        }

        writeDefaultJointPositions()
        writeJointTargets()
        mj_forward(model, data)
    }

    func snapshot() -> State {
        State(
            rootStartPosition: rootStartPosition,
            lastAction: lastAction,
            execAction: execAction,
            targetJointPos: targetJointPos,
            controlAccumulator: controlAccumulator,
            hasReportedPolicyFailure: hasReportedPolicyFailure,
            velocityCommand: velocityCommand,
            navigationPath: navigationPath
        )
    }

    func restore(_ state: State) {
        guard state.lastAction.count == spec.outputSize,
              state.execAction.count == spec.outputSize,
              state.targetJointPos.count == spec.outputSize else {
            return
        }
        rootStartPosition = state.rootStartPosition
        lastAction = state.lastAction
        execAction = state.execAction
        targetJointPos = state.targetJointPos
        controlAccumulator = state.controlAccumulator
        hasReportedPolicyFailure = state.hasReportedPolicyFailure
        velocityCommand = state.velocityCommand
        navigationPath = state.navigationPath
        writeJointTargets()
    }

    func setVelocityCommand(forward: Float, lateral: Float, yawRate: Float) {
        velocityCommand = clampedCommand(forward: forward, lateral: lateral, yawRate: yawRate)
        navigationPath = []
    }

    func setNavigationTarget(_ target: SIMD3<Float>?) {
        navigationPath = target.map { [$0] } ?? []
        if target != nil {
            velocityCommand = VelocityCommand(forward: 0, lateral: 0, yawRate: 0)
        }
    }

    func setNavigationPath(_ path: [SIMD3<Float>]) {
        navigationPath = path
        if !path.isEmpty {
            velocityCommand = VelocityCommand(forward: 0, lateral: 0, yawRate: 0)
        }
    }

    func rootPose() -> RobotPose {
        let position = SIMD3<Float>(
            Float(data.pointee.qpos[rootQposAddress + 0]),
            Float(data.pointee.qpos[rootQposAddress + 1]),
            Float(data.pointee.qpos[rootQposAddress + 2])
        )
        let quat = SIMD4<Double>(
            data.pointee.qpos[rootQposAddress + 3],
            data.pointee.qpos[rootQposAddress + 4],
            data.pointee.qpos[rootQposAddress + 5],
            data.pointee.qpos[rootQposAddress + 6]
        )
        return RobotPose(position: position, yaw: Float(yaw(from: quat)))
    }

    func step(elapsed: Double) {
        controlAccumulator += elapsed
        var controlSteps = 0
        while controlAccumulator >= Self.controlDt && controlSteps < 4 {
            inferAndUpdateTargets()
            for _ in 0..<Self.decimation {
                writeJointTargets()
                mj_step(model, data)
            }
            controlAccumulator -= Self.controlDt
            controlSteps += 1
        }

        if controlSteps == 4 {
            controlAccumulator = 0
        }

        if data.pointee.qpos[rootQposAddress + 2] < rootStartPosition.z - Double(robotScaleFloat) {
            mj_resetDataKeyframe(model, data, 0)
            reset(rootPosition: rootStartPosition)
        }
    }

    private func inferAndUpdateTargets() {
        guard let policy else {
            targetJointPos = spec.defaultJointPos
            return
        }

        do {
            let action = try policy.predict(observation: buildObservation())
            for i in 0..<spec.outputSize {
                let clipped = max(-100.0, min(100.0, Double(action[i])))
                execAction[i] = Float(clipped)
                lastAction[i] = Float(clipped)
                targetJointPos[i] = spec.defaultJointPos[i] + clipped * spec.actionScale[i]
            }
        } catch {
            if !hasReportedPolicyFailure {
                print("\(spec.logName) Core ML inference failed: \(error)")
                hasReportedPolicyFailure = true
            }
            targetJointPos = spec.defaultJointPos
        }
    }

    private func buildObservation() -> [Float] {
        var offset = 0

        for i in 0..<3 {
            observation[offset] = Float(data.pointee.sensordata[linVelSensorAddress + i]) / robotScaleFloat
            offset += 1
        }
        for i in 0..<3 {
            observation[offset] = Float(data.pointee.sensordata[angVelSensorAddress + i])
            offset += 1
        }

        let quat = SIMD4<Double>(
            data.pointee.qpos[rootQposAddress + 3],
            data.pointee.qpos[rootQposAddress + 4],
            data.pointee.qpos[rootQposAddress + 5],
            data.pointee.qpos[rootQposAddress + 6]
        )
        let projectedGravity = quatRotateInverse(quat, SIMD3<Double>(0, 0, -1))
        for i in 0..<3 {
            observation[offset] = Float(projectedGravity[i])
            offset += 1
        }

        for i in 0..<jointAddresses.count {
            observation[offset] = Float(data.pointee.qpos[jointAddresses[i].qpos] - spec.defaultJointPos[i])
            offset += 1
        }
        for address in jointAddresses {
            observation[offset] = Float(data.pointee.qvel[address.qvel])
            offset += 1
        }
        for action in lastAction {
            observation[offset] = action
            offset += 1
        }

        let command = resolvedVelocityCommand()
        observation[offset + 0] = command.forward
        observation[offset + 1] = command.lateral
        observation[offset + 2] = command.yawRate
        offset += 3

        offset = writeHeightScan(startOffset: offset)
        return observation
    }

    private func resolvedVelocityCommand() -> VelocityCommand {
        pruneReachedNavigationWaypoints()

        guard !navigationPath.isEmpty else {
            return velocityCommand
        }

        let pose = rootPose()
        let navigationTarget = navigationLookaheadTarget(from: pose.position)
        let delta = SIMD2<Float>(navigationTarget.x - pose.position.x, navigationTarget.y - pose.position.y)
        let distance = simd_length(delta)
        let scaledDistance = distance / robotScaleFloat
        guard scaledDistance > 0.08 else {
            return VelocityCommand(forward: 0, lateral: 0, yawRate: 0)
        }

        let targetYaw = atan2f(delta.y, delta.x)
        let yawError = wrappedAngle(targetYaw - pose.yaw)
        if spec.usesHolonomicNavigation {
            return holonomicNavigationCommand(
                delta: delta,
                distance: distance,
                scaledDistance: scaledDistance,
                yawError: yawError,
                pose: pose
            )
        }

        let absYawError = abs(yawError)
        let baseForward = min(max(scaledDistance * 0.72, 0.35), 0.88)
        let forward = absYawError > 1.25
            ? min(baseForward, spec.navigationForwardWhileTurning)
            : baseForward * max(0.35, cosf(yawError))
        return clampedCommand(forward: forward, lateral: 0, yawRate: yawError * 1.8)
    }

    private func holonomicNavigationCommand(
        delta: SIMD2<Float>,
        distance: Float,
        scaledDistance: Float,
        yawError: Float,
        pose: RobotPose
    ) -> VelocityCommand {
        let directionWorld = delta / distance
        let speed = min(max(scaledDistance * 0.75, 0.22), 0.90)
        let velocityWorld = directionWorld * speed
        let cosYaw = cosf(pose.yaw)
        let sinYaw = sinf(pose.yaw)
        let forward = cosYaw * velocityWorld.x + sinYaw * velocityWorld.y
        let lateral = -sinYaw * velocityWorld.x + cosYaw * velocityWorld.y
        let yawDeadband: Float = 0.18
        let yawRate = abs(yawError) < yawDeadband ? 0 : yawError * 0.35
        return clampedCommand(forward: forward, lateral: lateral, yawRate: yawRate)
    }

    private func clampedCommand(forward: Float, lateral: Float, yawRate: Float) -> VelocityCommand {
        let limits = spec.commandLimits
        return VelocityCommand(
            forward: min(max(forward, limits.forwardMin), limits.forwardMax),
            lateral: min(max(lateral, limits.lateralMin), limits.lateralMax),
            yawRate: min(max(yawRate, limits.yawRateMin), limits.yawRateMax)
        )
    }

    private func pruneReachedNavigationWaypoints() {
        let pose = rootPose()
        let robotPosition = SIMD2<Float>(pose.position.x, pose.position.y)
        let waypointRadius = Self.navigationWaypointRadius * robotScaleFloat
        while let first = navigationPath.first {
            let firstDistance = simd_distance(robotPosition, SIMD2<Float>(first.x, first.y))
            if firstDistance < waypointRadius {
                navigationPath.removeFirst()
                continue
            }
            if navigationPath.count > 1 {
                let second = navigationPath[1]
                let segment = SIMD2<Float>(second.x - first.x, second.y - first.y)
                let segmentLengthSquared = simd_dot(segment, segment)
                if segmentLengthSquared > 0.0001 {
                    let progress = simd_dot(robotPosition - SIMD2<Float>(first.x, first.y), segment) / segmentLengthSquared
                    let clampedProgress = min(max(progress, 0), 1)
                    let closest = SIMD2<Float>(first.x, first.y) + segment * clampedProgress
                    let lateralDistance = simd_distance(robotPosition, closest)
                    if progress > 0.25 && lateralDistance < waypointRadius {
                        navigationPath.removeFirst()
                        continue
                    }
                }
                let secondDistance = simd_distance(robotPosition, SIMD2<Float>(second.x, second.y))
                if firstDistance < waypointRadius * 1.25 &&
                    secondDistance + 0.18 * robotScaleFloat < firstDistance {
                    navigationPath.removeFirst()
                    continue
                }
            }
            break
        }
    }

    private func navigationLookaheadTarget(from position: SIMD3<Float>) -> SIMD3<Float> {
        let robotPosition = SIMD2<Float>(position.x, position.y)
        var target = navigationPath[0]
        let lookaheadDistance = Self.navigationLookaheadDistance * robotScaleFloat
        for point in navigationPath {
            target = point
            if simd_distance(robotPosition, SIMD2<Float>(point.x, point.y)) >= lookaheadDistance {
                break
            }
        }
        return target
    }

    private func writeHeightScan(startOffset: Int) -> Int {
        let px = data.pointee.xpos[heightScanBodyID * 3 + 0]
        let py = data.pointee.xpos[heightScanBodyID * 3 + 1]
        let pz = data.pointee.xpos[heightScanBodyID * 3 + 2]
        let xmat = data.pointee.xmat.advanced(by: heightScanBodyID * 9)

        var cx = xmat[0]
        var cy = xmat[3]
        var cn = hypot(cx, cy)
        if cn < 0.1 {
            var yx = xmat[1]
            var yy = xmat[4]
            let yn = max(hypot(yx, yy), 1.0e-6)
            yx /= yn
            yy /= yn
            cx = yy
            cy = -yx
            cn = 1
        } else {
            cx /= cn
            cy /= cn
        }

        let cosYaw = cx
        let sinYaw = cy
        var pnt = [mjtNum](repeating: 0, count: 3)
        var vec = [mjtNum](arrayLiteral: 0, 0, -1)
        var geomGroup = [mjtByte](repeating: 0, count: 6)
        geomGroup[0] = 1
        var geomID = [Int32](repeating: -1, count: 1)
        var normal = [mjtNum](repeating: 0, count: 3)

        var offset = startOffset
        for ray in heightScanOffsets {
            pnt[0] = px + cosYaw * ray.x - sinYaw * ray.y
            pnt[1] = py + sinYaw * ray.x + cosYaw * ray.y
            pnt[2] = pz

            let distance = mj_ray(
                model,
                data,
                &pnt,
                &vec,
                &geomGroup,
                1,
                Int32(heightScanExcludeBodyID),
                &geomID,
                &normal
            )
            let maxScanDistance = spec.heightScanMaxDistance * robotScale
            let height = distance < 0 || distance > maxScanDistance
                ? maxScanDistance
                : distance
            observation[offset] = Float(height / maxScanDistance)
            offset += 1
        }
        return offset
    }

    private func writeDefaultJointPositions() {
        for i in 0..<jointAddresses.count {
            data.pointee.qpos[jointAddresses[i].qpos] = spec.defaultJointPos[i]
            data.pointee.qvel[jointAddresses[i].qvel] = 0
        }
    }

    private func writeJointTargets() {
        for i in 0..<actuatorCtrlIDs.count {
            let jointIndex = actuatorToJointIndex[i]
            data.pointee.ctrl[actuatorCtrlIDs[i]] = targetJointPos[jointIndex]
        }
    }

    private static func makeHeightScanOffsets(sizeX: Double, sizeY: Double, resolution: Double) -> [(x: Double, y: Double)] {
        let xs = strideAxis(size: sizeX, resolution: resolution)
        let ys = strideAxis(size: sizeY, resolution: resolution)
        var offsets: [(x: Double, y: Double)] = []
        offsets.reserveCapacity(xs.count * ys.count)
        for y in ys {
            for x in xs {
                offsets.append((x, y))
            }
        }
        return offsets
    }

    private static func strideAxis(size: Double, resolution: Double) -> [Double] {
        let halfSize = size / 2
        let stop = halfSize + resolution * 0.5
        let count = max(0, Int(floor((stop - -halfSize) / resolution)) + 1)
        return (0..<count).map { -halfSize + Double($0) * resolution }
    }

    private static func objectID(_ model: UnsafeMutablePointer<mjModel>, type: mjtObj, name: String) -> Int? {
        let id = name.withCString { mj_name2id(model, Int32(type.rawValue), $0) }
        return id >= 0 ? Int(id) : nil
    }
}

nonisolated private final class CoreMLPolicy {
    private let model: MLModel
    private let inputName: String
    private let outputName: String
    private let outputSize: Int

    private init(model: MLModel, outputSize: Int) {
        self.model = model
        self.inputName = model.modelDescription.inputDescriptionsByName.keys.first ?? "obs"
        self.outputName = model.modelDescription.outputDescriptionsByName.keys.first ?? "actions"
        self.outputSize = outputSize
    }

    static func loadOptional(modelName: String, outputSize: Int, logName: String) -> CoreMLPolicy? {
        do {
            let url: URL
            if let compiledURL = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc") {
                url = compiledURL
            } else if let packageURL = Bundle.main.url(forResource: modelName, withExtension: "mlpackage") {
                url = try MLModel.compileModel(at: packageURL)
            } else if let modelURL = Bundle.main.url(forResource: modelName, withExtension: "mlmodel") {
                url = try MLModel.compileModel(at: modelURL)
            } else {
                print("\(logName) Core ML model not found in bundle: \(modelName).mlmodelc")
                return nil
            }
            let config = MLModelConfiguration()
            config.computeUnits = .all
            return try CoreMLPolicy(model: MLModel(contentsOf: url, configuration: config), outputSize: outputSize)
        } catch {
            print("\(logName) Core ML model load failed: \(error)")
            return nil
        }
    }

    func predict(observation: [Float]) throws -> [Float] {
        let input = try MLMultiArray(
            shape: [1, NSNumber(value: observation.count)],
            dataType: .float32
        )
        let pointer = input.dataPointer.bindMemory(to: Float32.self, capacity: observation.count)
        for i in 0..<observation.count {
            pointer[i] = observation[i]
        }

        let features = try MLDictionaryFeatureProvider(dictionary: [
            inputName: MLFeatureValue(multiArray: input),
        ])
        let prediction = try model.prediction(from: features)
        guard let outputArray = prediction.featureValue(for: outputName)?.multiArrayValue else {
            throw NSError(
                domain: "MujocoAR.CoreMLPolicy",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Core ML output not found: \(outputName)"]
            )
        }

        var output = [Float](repeating: 0, count: outputSize)
        for i in 0..<min(outputSize, outputArray.count) {
            output[i] = outputArray[i].floatValue
        }
        return output
    }
}

nonisolated private func quatRotateInverse(_ quat: SIMD4<Double>, _ vector: SIMD3<Double>) -> SIMD3<Double> {
    let w = quat.x
    let q = SIMD3<Double>(quat.y, quat.z, quat.w)
    let conjugate = SIMD4<Double>(w, -q.x, -q.y, -q.z)
    return quatRotate(conjugate, vector)
}

nonisolated private func quatRotate(_ quat: SIMD4<Double>, _ vector: SIMD3<Double>) -> SIMD3<Double> {
    let w = quat.x
    let q = SIMD3<Double>(quat.y, quat.z, quat.w)
    return vector + 2.0 * cross(q, cross(q, vector) + w * vector)
}

nonisolated private func yaw(from quat: SIMD4<Double>) -> Double {
    let w = quat.x
    let x = quat.y
    let y = quat.z
    let z = quat.w
    return atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
}

nonisolated private func wrappedAngle(_ angle: Float) -> Float {
    atan2f(sinf(angle), cosf(angle))
}
