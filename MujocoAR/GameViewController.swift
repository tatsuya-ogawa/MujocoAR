//
//  GameViewController.swift
//  MujocoAR
//
//  Created by Tatsuya Ogawa on 2026/05/29.
//

import ARKit
import MetalKit
import simd
import UIKit

@MainActor
final class GameViewController: UIViewController, UIGestureRecognizerDelegate {
    private enum SceneMode: Int {
        case debug = 0
        case ar = 1
    }

    private enum DebugCameraMode: Int {
        case orbit = 0
        case tps = 1
    }

    private enum TapMode: Int {
        case navigate = 0
        case spawn = 1
    }

    private enum RobotDriveMode: Int {
        case navigation = 0
        case direct = 1
    }

    private var arSession: ARSession?
    private var renderer: Renderer!
    private var mtkView: MTKView!
    private var simulation: MuJoCoSimulation!
    private var latestSimulationScene = MuJoCoRenderScene.empty
    private let navigator = RecastNavigator()
    private var displayLink: CADisplayLink?
    private var sceneMode: SceneMode = .debug
    private var debugCameraMode: DebugCameraMode = .orbit
    private var tapMode: TapMode = .navigate
    private var robotDriveMode: RobotDriveMode = .navigation
    private var robotKind: LocomotionRobotKind = .go1
    private var robotScale: Float = 1.0
    private var navigationOverlayEnabled = false
    private var mapOverlayEnabled = false
    private var debugSceneUpdatesEnabled = false
    private var cameraBackgroundEnabled = true
    private var arMeshWireframeEnabled = true
    private var heightFieldEnabled = true
    private var arCollisionUpdatesEnabled = true
    private var arHFieldDetailEnabled = false
    private var modeControl: UISegmentedControl!
    private var cameraModeControl: UISegmentedControl!
    private var robotControl: UISegmentedControl!
    private var robotScaleStepper: UIStepper!
    private var robotScaleLabel: UILabel!
    private var tapModeControl: UISegmentedControl!
    private var robotDriveModeControl: UISegmentedControl!
    private var panelContainerView: UIView!
    private var panelToggleButton: UIButton!
    private var panelStackView: UIStackView!
    private var panelCollapsed = false
    // Section containers
    private var robotSectionView: UIStackView!
    private var displaySectionView: UIStackView!
    private var processingSectionView: UIStackView!
    // Row references
    private var debugSceneUpdateRow: UIStackView!
    private var cameraBackgroundRow: UIStackView!
    private var meshUpdateRow: UIStackView!
    private var collisionUpdateRow: UIStackView!
    private var hFieldDetailRow: UIStackView!
    private var meshUpdateLabel: UILabel!
    private var meshUpdateIndicator: UIActivityIndicatorView!
    private var navigationOverlaySwitch: UISwitch!
    private var mapOverlaySwitch: UISwitch!
    private var arMeshWireframeSwitch: UISwitch!
    private var heightFieldSwitch: UISwitch!
    private var debugSceneUpdateSwitch: UISwitch!
    private var cameraBackgroundSwitch: UISwitch!
    private var hFieldDetailSwitch: UISwitch!
    private var mapOverlayRow: UIStackView!
    private var collisionUpdateButton: UIButton!
    private var robotResetButton: UIButton!
    private var joystickControlView: UIView!
    private var joystickThumbView: UIView!
    private var heightFieldMapView: HeightFieldMapView!
    private var lastNavigationTarget: SIMD3<Float>?
    private var lastNavigationPath: [SIMD3<Float>]?
    private var arMeshChunks: [UUID: EnvironmentMeshChunkDescriptor] = [:]
    private var latestARMeshAnchors: [UUID: ARMeshAnchor] = [:]
    private var arMeshChunkRevision = 0
    private var arMeshProcessingCount = 0
    private var arMeshRebuildPending = false
    private var arHFieldIsGenerating = false
    private var arMuJoCoIsApplying = false
    private var arHFieldProcessedCount = 0
    private var arHFieldTotalCount = 0
    private var arCollisionUpdateGeneration = 0
    private var pendingARMeshRebuildWorkItem: DispatchWorkItem?
    private var pendingARMeshRebuildRequiresForce = false
    private var arHFieldBuildInProgress = false
    private var arHFieldBuildGeneration = 0
    private var simulationFrameInFlight = false
    private var simulationSceneGeneration = 0
    private var arMeshChunkLastScheduledTimes: [UUID: CFTimeInterval] = [:]
    private var lastARCollisionCoveragePoint: SIMD2<Float>?
    private var lastARMeshSceneBuildTime: CFTimeInterval = 0
    private var lastDebugTerrainBuildTime: CFTimeInterval = 0
    private var debugTerrainPhase: Double = 0
    private let arMeshProcessingQueue = DispatchQueue(
        label: "MujocoAR.ARMeshProcessing",
        qos: .utility,
        attributes: .concurrent
    )
    private let arHFieldBuildQueue = DispatchQueue(
        label: "MujocoAR.ARHeightFieldBuild",
        qos: .userInitiated
    )
    private let arMeshChunkUpdateInterval: CFTimeInterval = 1.0
    private let arMeshSceneRebuildInterval: CFTimeInterval = 2.5
    private let arMeshSceneRebuildDelay: TimeInterval = 1.0
    private let arCollisionCoverageRebuildDistance: Float = 0.45
    private let debugTerrainRebuildInterval: CFTimeInterval = 0.75

    override func viewDidLoad() {
        super.viewDidLoad()

        guard let mtkView = view as? MTKView else {
            print("GameViewController view is not an MTKView")
            return
        }
        self.mtkView = mtkView

        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Metal is not supported")
            return
        }

#if targetEnvironment(simulator)
        print("ARKit + Metal 4 rendering requires a physical device")
        return
#else
        if !device.supportsFamily(.metal4) {
            print("Metal 4 is not supported")
            return
        }
        if !device.supportsShaderBarycentricCoordinates {
            print("Metal shader barycentric coordinates are not supported")
            return
        }

        mtkView.device = device
        mtkView.backgroundColor = .black
        mtkView.contentScaleFactor = traitCollection.displayScale
        mtkView.preferredFramesPerSecond = 60
        mtkView.enableSetNeedsDisplay = true
        mtkView.isPaused = true

        guard let renderer = Renderer(metalKitView: mtkView) else {
            print("Renderer cannot be initialized")
            return
        }
        self.renderer = renderer
        mtkView.delegate = renderer
        simulation = MuJoCoSimulation()
        renderer.setCameraBackgroundEnabled(cameraBackgroundEnabled)
        renderer.updateMuJoCoScene(renderSceneForDisplay())

        if ARWorldTrackingConfiguration.isSupported {
            sceneMode = .ar
            let arSession = ARSession()
            arSession.delegate = self
            arSession.delegateQueue = .main
            self.arSession = arSession
        }

        setupModeControl()
        setupCameraModeControl()
        setupDebugOptionsControl()
        setupDebugCameraGestures()
#endif
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        startDisplayLink()
        applySceneMode()
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        Task { [simulation] in
            await simulation?.setRobotVelocityCommand(forward: 0, lateral: 0, yawRate: 0)
        }
        resetJoystickControl()
        arSession?.pause()
        displayLink?.invalidate()
        displayLink = nil
    }

    private func setupModeControl() {
        modeControl = UISegmentedControl(items: ["Debug", "AR"])
        modeControl.selectedSegmentIndex = sceneMode.rawValue
        modeControl.translatesAutoresizingMaskIntoConstraints = false
        modeControl.backgroundColor = UIColor.black.withAlphaComponent(0.35)
        modeControl.selectedSegmentTintColor = .systemCyan
        modeControl.addTarget(self, action: #selector(sceneModeChanged), for: .valueChanged)
        view.addSubview(modeControl)

        NSLayoutConstraint.activate([
            modeControl.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            modeControl.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 12),
            modeControl.widthAnchor.constraint(equalToConstant: 180),
        ])

        setupHeightFieldMapView()
    }

    private func setupHeightFieldMapView() {
        let mapView = HeightFieldMapView(frame: .zero)
        mapView.translatesAutoresizingMaskIntoConstraints = false
        mapView.isHidden = true
        mapView.mapBackgroundColor = UIColor(white: 0.04, alpha: 0.72)
        mapView.layer.cornerRadius = 8
        mapView.layer.borderColor = UIColor.white.withAlphaComponent(0.24).cgColor
        mapView.layer.borderWidth = 1
        mapView.clipsToBounds = true
        mapView.onTap = { [weak self] point in
            self?.handleMapTap(at: point)
        }
        view.insertSubview(mapView, belowSubview: modeControl)
        let proportionalWidth = mapView.widthAnchor.constraint(equalTo: view.widthAnchor, multiplier: 0.42)
        proportionalWidth.priority = .defaultHigh
        NSLayoutConstraint.activate([
            mapView.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -16),
            mapView.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -16),
            proportionalWidth,
            mapView.widthAnchor.constraint(lessThanOrEqualToConstant: 240),
            mapView.widthAnchor.constraint(greaterThanOrEqualToConstant: 150),
            mapView.heightAnchor.constraint(equalTo: mapView.widthAnchor),
        ])
        heightFieldMapView = mapView
    }

    private func setupCameraModeControl() {
        cameraModeControl = UISegmentedControl(items: ["Orbit", "TPS"])
        cameraModeControl.selectedSegmentIndex = debugCameraMode.rawValue
        cameraModeControl.translatesAutoresizingMaskIntoConstraints = false
        cameraModeControl.backgroundColor = UIColor.black.withAlphaComponent(0.35)
        cameraModeControl.selectedSegmentTintColor = .systemGreen
        cameraModeControl.addTarget(self, action: #selector(debugCameraModeChanged), for: .valueChanged)
        view.addSubview(cameraModeControl)

        NSLayoutConstraint.activate([
            cameraModeControl.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            cameraModeControl.topAnchor.constraint(equalTo: modeControl.bottomAnchor, constant: 8),
            cameraModeControl.widthAnchor.constraint(equalToConstant: 220),
        ])
    }

    private func setupDebugOptionsControl() {
        // --- Robot Section ---
        robotControl = UISegmentedControl(items: LocomotionRobotKind.allCases.map(\.displayName))
        robotControl.selectedSegmentIndex = robotKind.rawValue
        robotControl.selectedSegmentTintColor = .systemCyan
        robotControl.addTarget(self, action: #selector(robotKindChanged), for: .valueChanged)

        robotScaleStepper = UIStepper()
        robotScaleStepper.minimumValue = 0.8
        robotScaleStepper.maximumValue = 2.0
        robotScaleStepper.stepValue = 0.1
        robotScaleStepper.value = Double(robotScale)
        robotScaleStepper.addTarget(self, action: #selector(robotScaleChanged), for: .valueChanged)

        robotScaleLabel = UILabel()
        robotScaleLabel.text = Self.robotScaleText(robotScale)
        robotScaleLabel.textColor = .white
        robotScaleLabel.font = .monospacedDigitSystemFont(ofSize: 13, weight: .semibold)
        robotScaleLabel.textAlignment = .right
        robotScaleLabel.widthAnchor.constraint(equalToConstant: 42).isActive = true

        tapModeControl = UISegmentedControl(items: ["Navigate", "Spawn"])
        tapModeControl.selectedSegmentIndex = tapMode.rawValue
        tapModeControl.selectedSegmentTintColor = .systemOrange
        tapModeControl.addTarget(self, action: #selector(tapModeChanged), for: .valueChanged)

        robotDriveModeControl = UISegmentedControl(items: ["Nav", "Direct"])
        robotDriveModeControl.selectedSegmentIndex = robotDriveMode.rawValue
        robotDriveModeControl.selectedSegmentTintColor = .systemGreen
        robotDriveModeControl.addTarget(self, action: #selector(robotDriveModeChanged), for: .valueChanged)

        robotResetButton = UIButton(type: .system)
        robotResetButton.translatesAutoresizingMaskIntoConstraints = false
        robotResetButton.titleLabel?.font = .systemFont(ofSize: 13, weight: .semibold)
        robotResetButton.layer.cornerRadius = 6
        robotResetButton.addTarget(self, action: #selector(robotResetButtonTapped), for: .touchUpInside)
        robotResetButton.widthAnchor.constraint(equalToConstant: 96).isActive = true

        let robotRow = makeSegmentedRow(title: "Robot", control: robotControl)
        let scaleRow = makeStepperRow(title: "Scale", valueLabel: robotScaleLabel, stepper: robotScaleStepper)
        let tapRow = makeSegmentedRow(title: "Tap", control: tapModeControl)
        let driveRow = makeSegmentedRow(title: "Control", control: robotDriveModeControl)
        let resetRow = makeButtonRow(title: "Reset", button: robotResetButton)

        robotSectionView = makeCollapsibleSection(title: "Robot", rows: [robotRow, scaleRow, tapRow, driveRow, resetRow])

        // --- Display Layers Section ---
        arMeshWireframeSwitch = UISwitch()
        arMeshWireframeSwitch.isOn = arMeshWireframeEnabled
        arMeshWireframeSwitch.addTarget(self, action: #selector(arMeshWireframeChanged), for: .valueChanged)

        heightFieldSwitch = UISwitch()
        heightFieldSwitch.isOn = heightFieldEnabled
        heightFieldSwitch.addTarget(self, action: #selector(heightFieldChanged), for: .valueChanged)

        navigationOverlaySwitch = UISwitch()
        navigationOverlaySwitch.isOn = navigationOverlayEnabled
        navigationOverlaySwitch.addTarget(self, action: #selector(navigationOverlayChanged), for: .valueChanged)

        mapOverlaySwitch = UISwitch()
        mapOverlaySwitch.isOn = mapOverlayEnabled
        mapOverlaySwitch.addTarget(self, action: #selector(mapOverlayChanged), for: .valueChanged)

        cameraBackgroundSwitch = UISwitch()
        cameraBackgroundSwitch.isOn = cameraBackgroundEnabled
        cameraBackgroundSwitch.addTarget(self, action: #selector(cameraBackgroundChanged), for: .valueChanged)

        let arMeshRow = makeSwitchRow(title: "AR Mesh", control: arMeshWireframeSwitch)
        let heightFieldRow = makeSwitchRow(title: "HeightField", control: heightFieldSwitch)
        let navRow = makeSwitchRow(title: "NavMesh", control: navigationOverlaySwitch)
        let mapRow = makeSwitchRow(title: "Map", control: mapOverlaySwitch)
        let cameraRow = makeSwitchRow(title: "Camera BG", control: cameraBackgroundSwitch)
        mapOverlayRow = mapRow
        cameraBackgroundRow = cameraRow

        displaySectionView = makeCollapsibleSection(title: "Display", rows: [arMeshRow, heightFieldRow, navRow, mapRow, cameraRow])

        // --- Processing Section ---
        debugSceneUpdateSwitch = UISwitch()
        debugSceneUpdateSwitch.isOn = debugSceneUpdatesEnabled
        debugSceneUpdateSwitch.addTarget(self, action: #selector(debugSceneUpdateChanged), for: .valueChanged)

        hFieldDetailSwitch = UISwitch()
        hFieldDetailSwitch.isOn = arHFieldDetailEnabled
        hFieldDetailSwitch.addTarget(self, action: #selector(hFieldDetailChanged), for: .valueChanged)

        collisionUpdateButton = UIButton(type: .system)
        collisionUpdateButton.translatesAutoresizingMaskIntoConstraints = false
        collisionUpdateButton.titleLabel?.font = .systemFont(ofSize: 13, weight: .semibold)
        collisionUpdateButton.layer.cornerRadius = 6
        collisionUpdateButton.addTarget(self, action: #selector(collisionUpdateButtonTapped), for: .touchUpInside)
        collisionUpdateButton.widthAnchor.constraint(equalToConstant: 96).isActive = true

        meshUpdateIndicator = UIActivityIndicatorView(style: .medium)
        meshUpdateIndicator.color = .white
        meshUpdateIndicator.hidesWhenStopped = true

        meshUpdateLabel = UILabel()
        meshUpdateLabel.text = "Scanning"
        meshUpdateLabel.textColor = .white
        meshUpdateLabel.font = .monospacedDigitSystemFont(ofSize: 13, weight: .semibold)
        meshUpdateLabel.textAlignment = .right

        let updateRow = makeSwitchRow(title: "Scene Update", control: debugSceneUpdateSwitch)
        let hFieldDetailRow = makeSwitchRow(title: "HField Detail", control: hFieldDetailSwitch)
        let collisionRow = makeButtonRow(title: "Collision", button: collisionUpdateButton)
        let meshRow = makeStatusRow(title: "Mesh", valueLabel: meshUpdateLabel, indicator: meshUpdateIndicator)
        debugSceneUpdateRow = updateRow
        self.hFieldDetailRow = hFieldDetailRow
        collisionUpdateRow = collisionRow
        meshUpdateRow = meshRow
        updateCollisionUpdateButton()

        processingSectionView = makeCollapsibleSection(title: "Processing", rows: [updateRow, hFieldDetailRow, collisionRow, meshRow])

        // --- Panel Layout ---
        panelStackView = UIStackView(arrangedSubviews: [robotSectionView, displaySectionView, processingSectionView])
        panelStackView.axis = .vertical
        panelStackView.spacing = 4
        panelStackView.translatesAutoresizingMaskIntoConstraints = false

        panelToggleButton = UIButton(type: .system)
        panelToggleButton.translatesAutoresizingMaskIntoConstraints = false
        panelToggleButton.setImage(UIImage(systemName: "chevron.up"), for: .normal)
        panelToggleButton.tintColor = .white
        panelToggleButton.backgroundColor = UIColor.black.withAlphaComponent(0.50)
        panelToggleButton.layer.cornerRadius = 14
        panelToggleButton.addTarget(self, action: #selector(panelToggleTapped), for: .touchUpInside)

        panelContainerView = UIView()
        panelContainerView.translatesAutoresizingMaskIntoConstraints = false
        panelContainerView.backgroundColor = .clear
        panelContainerView.addSubview(panelToggleButton)
        panelContainerView.addSubview(panelStackView)
        view.addSubview(panelContainerView)

        NSLayoutConstraint.activate([
            panelContainerView.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            panelContainerView.topAnchor.constraint(equalTo: cameraModeControl.bottomAnchor, constant: 8),
            panelContainerView.widthAnchor.constraint(equalToConstant: 220),
            panelToggleButton.topAnchor.constraint(equalTo: panelContainerView.topAnchor),
            panelToggleButton.trailingAnchor.constraint(equalTo: panelContainerView.trailingAnchor),
            panelToggleButton.widthAnchor.constraint(equalToConstant: 28),
            panelToggleButton.heightAnchor.constraint(equalToConstant: 28),
            panelStackView.topAnchor.constraint(equalTo: panelToggleButton.bottomAnchor, constant: 4),
            panelStackView.leadingAnchor.constraint(equalTo: panelContainerView.leadingAnchor),
            panelStackView.trailingAnchor.constraint(equalTo: panelContainerView.trailingAnchor),
            panelStackView.bottomAnchor.constraint(equalTo: panelContainerView.bottomAnchor),
        ])
        setupJoystickControl()
    }

    private func makeCollapsibleSection(title: String, rows: [UIStackView]) -> UIStackView {
        let headerButton = UIButton(type: .system)
        headerButton.translatesAutoresizingMaskIntoConstraints = false
        headerButton.contentHorizontalAlignment = .leading
        var config = UIButton.Configuration.plain()
        config.title = title
        config.image = UIImage(systemName: "chevron.down")
        config.imagePlacement = .trailing
        config.imagePadding = 4
        config.baseForegroundColor = .white.withAlphaComponent(0.85)
        config.contentInsets = NSDirectionalEdgeInsets(top: 4, leading: 4, bottom: 4, trailing: 4)
        headerButton.configuration = config
        headerButton.titleLabel?.font = .systemFont(ofSize: 12, weight: .bold)
        headerButton.addTarget(self, action: #selector(sectionHeaderTapped(_:)), for: .touchUpInside)

        let contentStack = UIStackView(arrangedSubviews: rows)
        contentStack.axis = .vertical
        contentStack.spacing = 6

        let section = UIStackView(arrangedSubviews: [headerButton, contentStack])
        section.axis = .vertical
        section.spacing = 2
        section.backgroundColor = UIColor.black.withAlphaComponent(0.35)
        section.layer.cornerRadius = 8
        section.isLayoutMarginsRelativeArrangement = true
        section.directionalLayoutMargins = NSDirectionalEdgeInsets(top: 4, leading: 10, bottom: 8, trailing: 10)
        return section
    }

    @objc private func sectionHeaderTapped(_ sender: UIButton) {
        guard let section = sender.superview as? UIStackView,
              section.arrangedSubviews.count > 1,
              let contentStack = section.arrangedSubviews.last else {
            return
        }
        let willCollapse = !contentStack.isHidden
        UIView.animate(withDuration: 0.22, delay: 0, options: .curveEaseInOut) {
            contentStack.isHidden = willCollapse
            contentStack.alpha = willCollapse ? 0 : 1
        }
        var config = sender.configuration ?? UIButton.Configuration.plain()
        config.image = UIImage(systemName: willCollapse ? "chevron.right" : "chevron.down")
        sender.configuration = config
    }

    @objc private func panelToggleTapped() {
        panelCollapsed.toggle()
        UIView.animate(withDuration: 0.22, delay: 0, options: .curveEaseInOut) { [self] in
            panelStackView.isHidden = panelCollapsed
            panelStackView.alpha = panelCollapsed ? 0 : 1
        }
        let imageName = panelCollapsed ? "chevron.down" : "chevron.up"
        panelToggleButton.setImage(UIImage(systemName: imageName), for: .normal)
    }

    private func setupJoystickControl() {
        joystickControlView = UIView()
        joystickControlView.translatesAutoresizingMaskIntoConstraints = false
        joystickControlView.backgroundColor = UIColor.black.withAlphaComponent(0.28)
        joystickControlView.layer.cornerRadius = 66
        joystickControlView.layer.borderColor = UIColor.white.withAlphaComponent(0.30).cgColor
        joystickControlView.layer.borderWidth = 1
        joystickControlView.isHidden = true

        joystickThumbView = UIView()
        joystickThumbView.translatesAutoresizingMaskIntoConstraints = false
        joystickThumbView.backgroundColor = UIColor.systemGreen.withAlphaComponent(0.78)
        joystickThumbView.layer.cornerRadius = 26
        joystickThumbView.layer.borderColor = UIColor.white.withAlphaComponent(0.45).cgColor
        joystickThumbView.layer.borderWidth = 1

        joystickControlView.addSubview(joystickThumbView)
        view.addSubview(joystickControlView)

        let pan = UIPanGestureRecognizer(target: self, action: #selector(joystickPanChanged(_:)))
        joystickControlView.addGestureRecognizer(pan)

        NSLayoutConstraint.activate([
            joystickControlView.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 24),
            joystickControlView.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -24),
            joystickControlView.widthAnchor.constraint(equalToConstant: 132),
            joystickControlView.heightAnchor.constraint(equalTo: joystickControlView.widthAnchor),
            joystickThumbView.centerXAnchor.constraint(equalTo: joystickControlView.centerXAnchor),
            joystickThumbView.centerYAnchor.constraint(equalTo: joystickControlView.centerYAnchor),
            joystickThumbView.widthAnchor.constraint(equalToConstant: 52),
            joystickThumbView.heightAnchor.constraint(equalTo: joystickThumbView.widthAnchor),
        ])
    }

    private func makeSegmentedRow(title: String, control: UISegmentedControl) -> UIStackView {
        let label = UILabel()
        label.text = title
        label.textColor = .white
        label.font = .systemFont(ofSize: 13, weight: .semibold)

        control.translatesAutoresizingMaskIntoConstraints = false
        control.backgroundColor = UIColor.black.withAlphaComponent(0.20)
        control.widthAnchor.constraint(equalToConstant: 128).isActive = true

        let row = UIStackView(arrangedSubviews: [label, control])
        row.axis = .horizontal
        row.alignment = .center
        row.distribution = .equalSpacing
        return row
    }

    private func makeStepperRow(title: String, valueLabel: UILabel, stepper: UIStepper) -> UIStackView {
        let label = UILabel()
        label.text = title
        label.textColor = .white
        label.font = .systemFont(ofSize: 13, weight: .semibold)

        let controls = UIStackView(arrangedSubviews: [valueLabel, stepper])
        controls.axis = .horizontal
        controls.spacing = 6
        controls.alignment = .center

        let row = UIStackView(arrangedSubviews: [label, controls])
        row.axis = .horizontal
        row.alignment = .center
        row.distribution = .equalSpacing
        return row
    }

    private func makeSwitchRow(title: String, control: UISwitch) -> UIStackView {
        let label = UILabel()
        label.text = title
        label.textColor = .white
        label.font = .systemFont(ofSize: 13, weight: .semibold)

        let row = UIStackView(arrangedSubviews: [label, control])
        row.axis = .horizontal
        row.alignment = .center
        row.distribution = .equalSpacing
        return row
    }

    private func makeButtonRow(title: String, button: UIButton) -> UIStackView {
        let label = UILabel()
        label.text = title
        label.textColor = .white
        label.font = .systemFont(ofSize: 13, weight: .semibold)

        let row = UIStackView(arrangedSubviews: [label, button])
        row.axis = .horizontal
        row.alignment = .center
        row.distribution = .equalSpacing
        return row
    }

    private func makeStatusRow(
        title: String,
        valueLabel: UILabel,
        indicator: UIActivityIndicatorView
    ) -> UIStackView {
        let label = UILabel()
        label.text = title
        label.textColor = .white
        label.font = .systemFont(ofSize: 13, weight: .semibold)

        let controls = UIStackView(arrangedSubviews: [indicator, valueLabel])
        controls.axis = .horizontal
        controls.spacing = 6
        controls.alignment = .center

        let row = UIStackView(arrangedSubviews: [label, controls])
        row.axis = .horizontal
        row.alignment = .center
        row.distribution = .equalSpacing
        return row
    }

    private func setupDebugCameraGestures() {
        let orbitPan = UIPanGestureRecognizer(target: self, action: #selector(debugOrbitPanChanged(_:)))
        orbitPan.maximumNumberOfTouches = 1
        orbitPan.delegate = self
        mtkView.addGestureRecognizer(orbitPan)

        let targetPan = UIPanGestureRecognizer(target: self, action: #selector(debugTargetPanChanged(_:)))
        targetPan.minimumNumberOfTouches = 2
        targetPan.maximumNumberOfTouches = 2
        targetPan.delegate = self
        mtkView.addGestureRecognizer(targetPan)

        let pinch = UIPinchGestureRecognizer(target: self, action: #selector(debugPinchChanged(_:)))
        pinch.delegate = self
        mtkView.addGestureRecognizer(pinch)

        let resetTap = UITapGestureRecognizer(target: self, action: #selector(debugCameraResetTapped(_:)))
        resetTap.numberOfTapsRequired = 2
        resetTap.delegate = self
        mtkView.addGestureRecognizer(resetTap)

        let targetTap = UITapGestureRecognizer(target: self, action: #selector(debugTargetTapped(_:)))
        targetTap.numberOfTapsRequired = 1
        targetTap.delegate = self
        targetTap.require(toFail: resetTap)
        mtkView.addGestureRecognizer(targetTap)
    }

    private func startDisplayLink() {
        guard displayLink == nil else {
            return
        }
        let displayLink = CADisplayLink(target: self, selector: #selector(gameLoopTick(_:)))
        displayLink.preferredFrameRateRange = CAFrameRateRange(minimum: 30, maximum: 60, preferred: 60)
        displayLink.add(to: .main, forMode: .common)
        self.displayLink = displayLink
    }

    @objc private func gameLoopTick(_ displayLink: CADisplayLink) {
        guard !simulationFrameInFlight else {
            return
        }

        let timestamp = displayLink.timestamp
        let debugTerrainPhase = consumeDebugTerrainPhaseUpdateIfNeeded(at: timestamp)
        let generation = simulationSceneGeneration
        simulationFrameInFlight = true
        Task { [weak self] in
            guard let self else {
                return
            }
            if let debugTerrainPhase {
                await simulation.load(environment: .debugDynamicTerrain(phase: debugTerrainPhase), preservingState: true)
            }
            let scene = await simulation.stepAndRenderScene(at: timestamp)
            guard generation == simulationSceneGeneration else {
                simulationFrameInFlight = false
                return
            }
            updateRenderedSimulationScene(scene)
            simulationFrameInFlight = false

            if sceneMode == .debug {
                mtkView.draw()
            }
        }
    }

    @objc private func sceneModeChanged() {
        guard let mode = SceneMode(rawValue: modeControl.selectedSegmentIndex) else {
            return
        }
        sceneMode = mode
        applySceneMode()
    }

    @objc private func debugCameraModeChanged() {
        guard let mode = DebugCameraMode(rawValue: cameraModeControl.selectedSegmentIndex) else {
            return
        }
        debugCameraMode = mode
        applyDebugCameraMode()
        drawDebugFrame()
    }

    @objc private func tapModeChanged() {
        guard let mode = TapMode(rawValue: tapModeControl.selectedSegmentIndex) else {
            return
        }
        setTapMode(mode)
    }

    @objc private func robotDriveModeChanged() {
        guard let mode = RobotDriveMode(rawValue: robotDriveModeControl.selectedSegmentIndex) else {
            return
        }
        setRobotDriveMode(mode)
        drawFrame()
    }

    @objc private func robotScaleChanged() {
        let newScale = Float(robotScaleStepper.value)
        guard abs(newScale - robotScale) > 0.001 else {
            return
        }

        robotScale = newScale
        robotScaleLabel.text = Self.robotScaleText(robotScale)
        navigator.clear()
        debugTerrainPhase = 0
        lastDebugTerrainBuildTime = 0
        simulationSceneGeneration += 1
        let generation = simulationSceneGeneration

        switch sceneMode {
        case .debug:
            Task { [weak self] in
                guard let self else {
                    return
                }
                await simulation.setRobotScale(robotScale)
                await simulation.load(environment: .debugDynamicTerrain(phase: debugTerrainPhase))
                await simulation.setRobotVelocityCommand(forward: 0, lateral: 0, yawRate: 0)
                let scene = await simulation.renderScene()
                guard generation == simulationSceneGeneration else {
                    return
                }
                applyDebugCameraMode()
                updateRenderedSimulationScene(scene)
                drawDebugFrame()
                updateRobotDriveUI()
            }
        case .ar:
            Task { [weak self] in
                guard let self else {
                    return
                }
                await simulation.setRobotScale(robotScale)
                await simulation.clearRobotNavigationTarget()
                await simulation.setRobotVelocityCommand(forward: 0, lateral: 0, yawRate: 0)
                if !rebuildARMuJoCoSceneIfNeeded(force: true, preservingState: false) {
                    await simulation.clear()
                    let scene = await simulation.renderScene()
                    guard generation == simulationSceneGeneration else {
                        return
                    }
                    updateRenderedSimulationScene(scene)
                    drawFrame()
                }
                updateRobotDriveUI()
            }
        }
    }

    @objc private func robotKindChanged() {
        guard let kind = LocomotionRobotKind(rawValue: robotControl.selectedSegmentIndex),
              kind != robotKind else {
            return
        }

        robotKind = kind
        navigator.clear()
        debugTerrainPhase = 0
        lastDebugTerrainBuildTime = 0
        simulationSceneGeneration += 1
        let generation = simulationSceneGeneration

        switch sceneMode {
        case .debug:
            Task { [weak self] in
                guard let self else {
                    return
                }
                await simulation.setRobotKind(kind)
                await simulation.load(environment: .debugDynamicTerrain(phase: debugTerrainPhase))
                await simulation.setRobotVelocityCommand(forward: 0, lateral: 0, yawRate: 0)
                let scene = await simulation.renderScene()
                guard generation == simulationSceneGeneration else {
                    return
                }
                applyDebugCameraMode()
                updateRenderedSimulationScene(scene)
                drawDebugFrame()
                updateRobotDriveUI()
            }
        case .ar:
            Task { [weak self] in
                guard let self else {
                    return
                }
                await simulation.setRobotKind(kind)
                await simulation.clearRobotNavigationTarget()
                await simulation.setRobotVelocityCommand(forward: 0, lateral: 0, yawRate: 0)
                if !rebuildARMuJoCoSceneIfNeeded(force: true, preservingState: false) {
                    await simulation.clear()
                    let scene = await simulation.renderScene()
                    guard generation == simulationSceneGeneration else {
                        return
                    }
                    updateRenderedSimulationScene(scene)
                    drawFrame()
                }
                updateRobotDriveUI()
            }
        }
    }

    @objc private func navigationOverlayChanged() {
        navigationOverlayEnabled = navigationOverlaySwitch.isOn
        updateMapOverlayVisibility()
        drawFrame()
    }

    @objc private func mapOverlayChanged() {
        mapOverlayEnabled = mapOverlaySwitch.isOn
        updateMapOverlayVisibility()
    }

    @objc private func arMeshWireframeChanged() {
        arMeshWireframeEnabled = arMeshWireframeSwitch.isOn
        renderer.setARMeshWireframeEnabled(arMeshWireframeEnabled)
        drawFrame()
    }

    @objc private func heightFieldChanged() {
        heightFieldEnabled = heightFieldSwitch.isOn
        renderer.setHeightFieldEnabled(heightFieldEnabled)
        drawFrame()
    }

    @objc private func debugSceneUpdateChanged() {
        debugSceneUpdatesEnabled = debugSceneUpdateSwitch.isOn
        lastDebugTerrainBuildTime = 0
        drawDebugFrame()
    }

    @objc private func cameraBackgroundChanged() {
        cameraBackgroundEnabled = cameraBackgroundSwitch.isOn
        renderer.setCameraBackgroundEnabled(cameraBackgroundEnabled)
        drawFrame()
    }

    @objc private func hFieldDetailChanged() {
        arHFieldDetailEnabled = hFieldDetailSwitch.isOn
        arHFieldBuildGeneration += 1
        arHFieldBuildInProgress = false
        Task { [weak self] in
            guard let self else {
                return
            }
            await simulation.setARCollisionHFieldDetailEnabled(arHFieldDetailEnabled)
            guard isARSimulationActive else {
                return
            }
            scheduleARMuJoCoSceneRebuild(force: true)
        }
    }

    @objc private func collisionUpdateButtonTapped() {
        arCollisionUpdatesEnabled.toggle()
        if arCollisionUpdatesEnabled {
            resumeARCollisionUpdates()
        } else {
            pauseARCollisionUpdates()
        }
        updateMeshStatusUI()
    }

    @objc private func robotResetButtonTapped() {
        resetJoystickControl()
        Task { [weak self] in
            guard let self else {
                return
            }
            await simulation.clearRobotNavigationTarget()
            await simulation.setRobotVelocityCommand(forward: 0, lateral: 0, yawRate: 0)
            await simulation.reset()
            let scene = await simulation.renderScene()
            updateRenderedSimulationScene(scene)
            updateRobotDriveUI()
            drawFrame()
        }
    }

    @objc private func joystickPanChanged(_ gesture: UIPanGestureRecognizer) {
        guard robotDriveMode == .direct, currentRobotIsSpawned() else {
            resetJoystickControl()
            Task { [simulation] in
                await simulation?.setRobotVelocityCommand(forward: 0, lateral: 0, yawRate: 0)
            }
            return
        }

        switch gesture.state {
        case .ended, .cancelled, .failed:
            resetJoystickControl()
            Task { [simulation] in
                await simulation?.setRobotVelocityCommand(forward: 0, lateral: 0, yawRate: 0)
            }
        default:
            updateDirectRobotVelocity(from: joystickVector(for: gesture))
        }
        drawFrame()
    }

    @available(*, deprecated, message: "Temporary collision probe shooter for AR collision debugging; no UI is wired.")
    private func shootCollisionProbeForDebug() {
        let scene = renderSceneForDisplay()
        guard let ray = collisionProbeLaunchRay(in: scene) else {
            return
        }

        Task { [weak self] in
            guard let self else {
                return
            }
            await simulation.launchCollisionProbe(from: ray.origin, direction: ray.direction)
            let scene = await simulation.renderScene()
            updateRenderedSimulationScene(scene)
            drawFrame()
        }
    }

    @objc private func debugOrbitPanChanged(_ gesture: UIPanGestureRecognizer) {
        guard sceneMode == .debug else {
            return
        }
        if debugCameraMode == .tps {
            guard robotDriveMode == .direct else {
                return
            }
            updateTPSVelocityCommand(from: gesture)
            return
        }

        let delta = gesture.translation(in: mtkView)
        renderer.orbitDebugCamera(delta: delta)
        gesture.setTranslation(.zero, in: mtkView)
        drawDebugFrame()
    }

    @objc private func debugTargetPanChanged(_ gesture: UIPanGestureRecognizer) {
        guard sceneMode == .debug, debugCameraMode == .orbit else {
            return
        }
        let delta = gesture.translation(in: mtkView)
        renderer.panDebugCamera(delta: delta)
        gesture.setTranslation(.zero, in: mtkView)
        drawDebugFrame()
    }

    @objc private func debugPinchChanged(_ gesture: UIPinchGestureRecognizer) {
        guard sceneMode == .debug else {
            return
        }
        renderer.zoomDebugCamera(scale: gesture.scale)
        gesture.scale = 1
        drawDebugFrame()
    }

    @objc private func debugCameraResetTapped(_ gesture: UITapGestureRecognizer) {
        guard sceneMode == .debug, gesture.state == .ended else {
            return
        }
        renderer.resetDebugCamera()
        drawDebugFrame()
    }

    @objc private func debugTargetTapped(_ gesture: UITapGestureRecognizer) {
        guard gesture.state == .ended else {
            return
        }

        switch sceneMode {
        case .debug:
            handleDebugTap(at: gesture.location(in: mtkView))
        case .ar:
            handleARTap(at: gesture.location(in: mtkView))
        }
    }

    private func handleDebugTap(at location: CGPoint) {
        let scene = renderSceneForDisplay()
        guard let point = renderer.debugGroundPoint(for: location, in: mtkView, scene: scene) else {
            return
        }
        applyTapAction(at: point, in: scene)
        renderer.updateMuJoCoScene(renderSceneForDisplay())
        mtkView.draw()
    }

    private func handleARTap(at location: CGPoint) {
#if !targetEnvironment(simulator)
        guard let frame = arSession?.currentFrame else {
            return
        }
        let scene = renderSceneForDisplay()
        guard let point = arScenePoint(for: location, frame: frame, scene: scene) else {
            return
        }
        applyTapAction(at: point, in: scene)
        renderer.updateMuJoCoScene(renderSceneForDisplay())
        mtkView.draw()
#endif
    }

    private func handleMapTap(at point: SIMD3<Float>) {
        let scene = renderSceneForDisplay()
        applyTapAction(at: point, in: scene)
        renderer.updateMuJoCoScene(renderSceneForDisplay())
        heightFieldMapView?.updateScene(renderSceneForDisplay())
        mtkView.draw()
    }

    private func applyTapAction(at point: SIMD3<Float>, in scene: MuJoCoRenderScene) {
        switch tapMode {
        case .navigate:
            guard robotDriveMode == .navigation else {
                return
            }
            if let robotPose = scene.robotPose,
               let path = navigator.path(from: robotPose.position, to: point, in: scene) {
                lastNavigationPath = path
                lastNavigationTarget = path.last ?? point
                Task { [simulation] in
                    await simulation?.setRobotNavigationPath(path)
                }
            } else {
                lastNavigationPath = nil
                lastNavigationTarget = point
                Task { [simulation] in
                    await simulation?.setRobotNavigationTarget(point)
                }
            }
            heightFieldMapView?.navigationTarget = lastNavigationTarget
            heightFieldMapView?.navigationPath = lastNavigationPath
        case .spawn:
            setTapMode(.navigate)
            lastNavigationTarget = nil
            lastNavigationPath = nil
            heightFieldMapView?.navigationTarget = nil
            heightFieldMapView?.navigationPath = nil
            Task { [weak self] in
                guard let self else {
                    return
                }
                await simulation.spawnRobot(at: point)
                await simulation.setRobotVelocityCommand(forward: 0, lateral: 0, yawRate: 0)
                let scene = await simulation.renderScene()
                updateRenderedSimulationScene(scene)
                updateRobotDriveUI()
            }
        }
    }

    private func drawDebugFrame() {
        guard sceneMode == .debug else {
            return
        }
        drawFrame()
    }

    private func drawFrame() {
        renderer.updateMuJoCoScene(renderSceneForDisplay())
        updateMapOverlayVisibility()
        mtkView.draw()
    }

    private func renderSceneForDisplay() -> MuJoCoRenderScene {
        let scene = latestSimulationScene
        guard navigationOverlayEnabled else {
            return scene
        }
        return navigator.sceneWithDebugMesh(scene)
    }

    private func updateRenderedSimulationScene(_ scene: MuJoCoRenderScene) {
        latestSimulationScene = scene
        renderer.updateMuJoCoScene(renderSceneForDisplay())
        updateMapOverlayVisibility()
    }

    private func updateMapOverlayVisibility() {
        guard let mapView = heightFieldMapView else {
            return
        }
        let shouldShow = sceneMode == .ar && mapOverlayEnabled
        mapView.isHidden = !shouldShow
        guard shouldShow else {
            return
        }
        mapView.showsNavigationOverlay = navigationOverlayEnabled
        mapView.navigationTarget = lastNavigationTarget
        mapView.navigationPath = lastNavigationPath
        mapView.updateScene(renderSceneForDisplay())
    }

    private func refreshRenderedSimulationScene(draw: Bool = false) {
        Task { [weak self] in
            guard let self else {
                return
            }
            let scene = await simulation.renderScene()
            updateRenderedSimulationScene(scene)
            if draw {
                mtkView.draw()
            }
        }
    }

    @available(*, deprecated, message: "Temporary collision probe shooter support; no UI is wired.")
    private func collisionProbeLaunchRay(in scene: MuJoCoRenderScene) -> (origin: SIMD3<Float>, direction: SIMD3<Float>)? {
        switch sceneMode {
        case .debug:
            return renderer.debugLaunchRay(scene: scene)
        case .ar:
#if !targetEnvironment(simulator)
            guard let frame = arSession?.currentFrame else {
                return nil
            }
            let cameraTransform = frame.camera.transform
            let originAR = SIMD3<Float>(
                cameraTransform.columns.3.x,
                cameraTransform.columns.3.y,
                cameraTransform.columns.3.z
            )
            let forwardAR = -SIMD3<Float>(
                cameraTransform.columns.2.x,
                cameraTransform.columns.2.y,
                cameraTransform.columns.2.z
            )
            let inverseWorldTransform = simd_inverse(scene.worldTransform)
            let origin4 = inverseWorldTransform * SIMD4<Float>(originAR.x, originAR.y, originAR.z, 1)
            let direction4 = inverseWorldTransform * SIMD4<Float>(forwardAR.x, forwardAR.y, forwardAR.z, 0)
            let direction = simd_normalize(SIMD3<Float>(direction4.x, direction4.y, direction4.z))
            return (SIMD3<Float>(origin4.x, origin4.y, origin4.z), direction)
#else
            return nil
#endif
        }
    }

    private func arScenePoint(for point: CGPoint, frame: ARFrame, scene: MuJoCoRenderScene) -> SIMD3<Float>? {
        guard let mesh = scene.environmentMesh else {
            return nil
        }

        let orientation = currentInterfaceOrientation()
        let viewMatrix = frame.camera.viewMatrix(for: orientation)
        let projectionMatrix = frame.camera.projectionMatrix(
            for: orientation,
            viewportSize: mtkView.drawableSize,
            zNear: 0.001,
            zFar: 1000
        )

        let drawableWidth = max(Float(mtkView.drawableSize.width), 1)
        let drawableHeight = max(Float(mtkView.drawableSize.height), 1)
        let scale = Float(mtkView.contentScaleFactor)
        let x = (Float(point.x) * scale / drawableWidth) * 2 - 1
        let y = 1 - (Float(point.y) * scale / drawableHeight) * 2
        let inverseViewProjection = simd_inverse(projectionMatrix * viewMatrix)
        let near4 = inverseViewProjection * SIMD4<Float>(x, y, 0, 1)
        let far4 = inverseViewProjection * SIMD4<Float>(x, y, 1, 1)
        guard abs(near4.w) > 0.000001, abs(far4.w) > 0.000001 else {
            return nil
        }

        let rayOriginAR = SIMD3<Float>(near4.x, near4.y, near4.z) / near4.w
        let rayFarAR = SIMD3<Float>(far4.x, far4.y, far4.z) / far4.w
        let rayDirectionAR = simd_normalize(rayFarAR - rayOriginAR)

        let inverseWorldTransform = simd_inverse(scene.worldTransform)
        let origin4 = inverseWorldTransform * SIMD4<Float>(rayOriginAR.x, rayOriginAR.y, rayOriginAR.z, 1)
        let direction4 = inverseWorldTransform * SIMD4<Float>(rayDirectionAR.x, rayDirectionAR.y, rayDirectionAR.z, 0)
        let rayOrigin = SIMD3<Float>(origin4.x, origin4.y, origin4.z)
        let rayDirection = simd_normalize(SIMD3<Float>(direction4.x, direction4.y, direction4.z))
        if let meshPoint = raycastMesh(mesh, origin: rayOrigin, direction: rayDirection) {
            return meshPoint
        }

        let query = ARRaycastQuery(
            origin: rayOriginAR,
            direction: rayDirectionAR,
            allowing: .estimatedPlane,
            alignment: .any
        )
        guard let result = arSession?.raycast(query).first else {
            return nil
        }
        let translation = result.worldTransform.columns.3
        return matrix4x4_muJoCoPosition(fromARPosition: SIMD3<Float>(translation.x, translation.y, translation.z))
    }

    private func raycastMesh(_ mesh: RenderMeshDescriptor, origin: SIMD3<Float>, direction: SIMD3<Float>) -> SIMD3<Float>? {
        var bestDistance = Float.greatestFiniteMagnitude
        var bestPoint: SIMD3<Float>?

        for faceStart in stride(from: 0, to: mesh.indices.count - 2, by: 3) {
            let ia = Int(mesh.indices[faceStart])
            let ib = Int(mesh.indices[faceStart + 1])
            let ic = Int(mesh.indices[faceStart + 2])
            guard ia < mesh.vertices.count, ib < mesh.vertices.count, ic < mesh.vertices.count else {
                continue
            }
            guard let distance = rayTriangleDistance(
                origin: origin,
                direction: direction,
                a: mesh.vertices[ia],
                b: mesh.vertices[ib],
                c: mesh.vertices[ic]
            ), distance < bestDistance else {
                continue
            }
            bestDistance = distance
            bestPoint = origin + direction * distance
        }

        return bestPoint
    }

    private func rayTriangleDistance(
        origin: SIMD3<Float>,
        direction: SIMD3<Float>,
        a: SIMD3<Float>,
        b: SIMD3<Float>,
        c: SIMD3<Float>
    ) -> Float? {
        let epsilon: Float = 0.000001
        let edge1 = b - a
        let edge2 = c - a
        let h = simd_cross(direction, edge2)
        let determinant = simd_dot(edge1, h)
        guard abs(determinant) > epsilon else {
            return nil
        }
        let inverseDeterminant = 1 / determinant
        let s = origin - a
        let u = inverseDeterminant * simd_dot(s, h)
        guard u >= 0, u <= 1 else {
            return nil
        }
        let q = simd_cross(s, edge1)
        let v = inverseDeterminant * simd_dot(direction, q)
        guard v >= 0, u + v <= 1 else {
            return nil
        }
        let t = inverseDeterminant * simd_dot(edge2, q)
        return t > epsilon ? t : nil
    }

    private func currentInterfaceOrientation() -> UIInterfaceOrientation {
        if let orientation = view.window?.windowScene?.effectiveGeometry.interfaceOrientation, orientation != .unknown {
            return orientation
        }
        return .portrait
    }

    private func currentRobotIsSpawned() -> Bool {
        latestSimulationScene.robotPose != nil
    }

    private func setRobotDriveMode(_ mode: RobotDriveMode) {
        robotDriveMode = mode
        robotDriveModeControl?.selectedSegmentIndex = mode.rawValue
        Task { [simulation] in
            await simulation?.clearRobotNavigationTarget()
            await simulation?.setRobotVelocityCommand(forward: 0, lateral: 0, yawRate: 0)
        }

        if mode == .direct, sceneMode == .debug {
            debugCameraMode = .tps
            cameraModeControl?.selectedSegmentIndex = debugCameraMode.rawValue
            applyDebugCameraMode()
        }

        resetJoystickControl()
        updateRobotDriveUI()
    }

    private func updateRobotDriveUI() {
        guard robotDriveModeControl != nil else {
            return
        }

        let hasRobot = currentRobotIsSpawned()
        robotDriveModeControl.setEnabled(hasRobot, forSegmentAt: RobotDriveMode.direct.rawValue)
        updateRobotResetButton(isEnabled: hasRobot)
        if !hasRobot, robotDriveMode == .direct {
            robotDriveMode = .navigation
            robotDriveModeControl.selectedSegmentIndex = robotDriveMode.rawValue
            Task { [simulation] in
                await simulation?.setRobotVelocityCommand(forward: 0, lateral: 0, yawRate: 0)
            }
            resetJoystickControl()
        }
        joystickControlView?.isHidden = !(hasRobot && robotDriveMode == .direct)
    }

    private func updateRobotResetButton(isEnabled: Bool) {
        guard robotResetButton != nil else {
            return
        }
        robotResetButton.isEnabled = isEnabled
        robotResetButton.setTitle("Reset", for: .normal)
        robotResetButton.setTitleColor(.white.withAlphaComponent(isEnabled ? 1.0 : 0.45), for: .normal)
        robotResetButton.backgroundColor = isEnabled
            ? UIColor.systemRed.withAlphaComponent(0.78)
            : UIColor.black.withAlphaComponent(0.22)
    }

    private func joystickVector(for gesture: UIPanGestureRecognizer) -> CGVector {
        let bounds = joystickControlView.bounds
        let center = CGPoint(x: bounds.midX, y: bounds.midY)
        let location = gesture.location(in: joystickControlView)
        let raw = CGVector(dx: location.x - center.x, dy: location.y - center.y)
        let maxRadius: CGFloat = 48
        let length = hypot(raw.dx, raw.dy)
        let scale = length > maxRadius && length > 0 ? maxRadius / length : 1
        let clamped = CGVector(dx: raw.dx * scale, dy: raw.dy * scale)
        joystickThumbView.transform = CGAffineTransform(translationX: clamped.dx, y: clamped.dy)
        return CGVector(dx: clamped.dx / maxRadius, dy: -clamped.dy / maxRadius)
    }

    private func resetJoystickControl() {
        joystickThumbView?.transform = .identity
    }

    private func updateDirectRobotVelocity(from vector: CGVector) {
        let normalizedForward = Float(vector.dy)
        let forward = normalizedForward >= 0
            ? min(normalizedForward * 0.9, 0.9)
            : max(normalizedForward * 0.25, -0.25)
        let turn = Float(vector.dx)
        let lateral: Float
        let yawRate: Float
        if robotKind == .g1 {
            lateral = min(max(-turn * 0.8, -0.8), 0.8)
            yawRate = min(max(turn * 0.45, -0.45), 0.45)
        } else {
            lateral = 0
            yawRate = min(max(turn * 1.35, -1.35), 1.35)
        }
        Task { [simulation] in
            await simulation?.setRobotVelocityCommand(forward: forward, lateral: lateral, yawRate: yawRate)
        }
    }

    private func updateTPSVelocityCommand(from gesture: UIPanGestureRecognizer) {
        switch gesture.state {
        case .ended, .cancelled, .failed:
            Task { [simulation] in
                await simulation?.setRobotVelocityCommand(forward: 0, lateral: 0, yawRate: 0)
            }
        default:
            let translation = gesture.translation(in: mtkView)
            let forward = min(max(Float(-translation.y / 140), -0.25), 0.9)
            let lateral: Float
            let yawRate: Float
            if robotKind == .g1 {
                lateral = min(max(Float(-translation.x / 160), -0.8), 0.8)
                yawRate = min(max(Float(translation.x / 320), -0.45), 0.45)
            } else {
                lateral = 0
                yawRate = min(max(Float(translation.x / 120), -1.35), 1.35)
            }
            Task { [simulation] in
                await simulation?.setRobotVelocityCommand(forward: forward, lateral: lateral, yawRate: yawRate)
            }
        }
        drawDebugFrame()
    }

    private func applyDebugCameraMode() {
        switch debugCameraMode {
        case .orbit:
            renderer.setDebugCameraMode(.orbit)
        case .tps:
            renderer.setDebugCameraMode(.thirdPerson)
        }
    }

    private func applySceneMode() {
        simulationSceneGeneration += 1
        let generation = simulationSceneGeneration
        switch sceneMode {
        case .debug:
            arCollisionUpdateGeneration += 1
            cancelPendingARMeshRebuild()
            arSession?.pause()
            navigator.clear()
            arMeshChunks.removeAll()
            latestARMeshAnchors.removeAll()
            arMeshChunkLastScheduledTimes.removeAll()
            lastARCollisionCoveragePoint = nil
            arMeshProcessingCount = 0
            arMeshRebuildPending = false
            arHFieldIsGenerating = false
            arMuJoCoIsApplying = false
            arMeshChunkRevision = 0
            debugTerrainPhase = 0
            lastDebugTerrainBuildTime = 0
            setTapMode(.navigate)
            setRobotDriveMode(.navigation)
            applyDebugCameraMode()
            cameraModeControl.isHidden = false
            panelContainerView.isHidden = false
            debugSceneUpdateRow.isHidden = false
            cameraBackgroundRow.isHidden = true
            mapOverlayRow.isHidden = true
            hFieldDetailRow.isHidden = true
            collisionUpdateRow.isHidden = true
            meshUpdateRow.isHidden = true
            processingSectionView.isHidden = false
            updateMapOverlayVisibility()
            updateRobotDriveUI()
            Task { [weak self] in
                guard let self else {
                    return
                }
                await simulation.setRobotKind(robotKind)
                await simulation.setRobotScale(robotScale)
                await simulation.load(environment: .debugDynamicTerrain(phase: debugTerrainPhase))
                await simulation.setRobotVelocityCommand(forward: 0, lateral: 0, yawRate: 0)
                let scene = await simulation.renderScene()
                guard generation == simulationSceneGeneration else {
                    return
                }
                updateRenderedSimulationScene(scene)
                drawDebugFrame()
                updateRobotDriveUI()
            }
        case .ar:
            arCollisionUpdateGeneration += 1
            setTapMode(.spawn)
            setRobotDriveMode(.navigation)
            cancelPendingARMeshRebuild()
            latestARMeshAnchors.removeAll()
            lastARMeshSceneBuildTime = 0
            lastARCollisionCoveragePoint = nil
            navigator.clear()
            cameraModeControl.isHidden = true
            panelContainerView.isHidden = false
            debugSceneUpdateRow.isHidden = true
            cameraBackgroundRow.isHidden = false
            mapOverlayRow.isHidden = false
            hFieldDetailRow.isHidden = false
            collisionUpdateRow.isHidden = false
            meshUpdateRow.isHidden = false
            processingSectionView.isHidden = false
            renderer.setCameraBackgroundEnabled(cameraBackgroundEnabled)
            renderer.setARMeshWireframeEnabled(arMeshWireframeEnabled)
            renderer.setHeightFieldEnabled(heightFieldEnabled)
            updateMapOverlayVisibility()
            startARSession()
            updateRobotDriveUI()
            updateMeshStatusUI()
            Task { [weak self] in
                guard let self else {
                    return
                }
                await simulation.setRobotKind(robotKind)
                await simulation.setRobotScale(robotScale)
                await simulation.setARCollisionHFieldDetailEnabled(arHFieldDetailEnabled)
                await simulation.clearRobotNavigationTarget()
                await simulation.setRobotVelocityCommand(forward: 0, lateral: 0, yawRate: 0)
                if !rebuildARMuJoCoSceneIfNeeded(force: true, preservingState: false) {
                    await simulation.clear()
                    let scene = await simulation.renderScene()
                    guard generation == simulationSceneGeneration else {
                        return
                    }
                    updateRenderedSimulationScene(scene)
                }
                updateRobotDriveUI()
                updateMeshStatusUI()
            }
        }
    }

    private static func robotScaleText(_ scale: Float) -> String {
        String(format: "%.1fx", scale)
    }

    /// Whether the AR session and AR-driven simulation pipeline should be active.
    private var isARSimulationActive: Bool {
        sceneMode == .ar
    }

    private func setTapMode(_ mode: TapMode) {
        tapMode = mode
        tapModeControl?.selectedSegmentIndex = mode.rawValue
    }

    private func cancelPendingARMeshRebuild() {
        pendingARMeshRebuildWorkItem?.cancel()
        pendingARMeshRebuildWorkItem = nil
        pendingARMeshRebuildRequiresForce = false
        arMeshRebuildPending = false
        arHFieldIsGenerating = false
        arMuJoCoIsApplying = false
        arHFieldBuildInProgress = false
        arHFieldBuildGeneration += 1
        resetARHFieldProgress()
    }

    private func pauseARCollisionUpdates() {
        arCollisionUpdateGeneration += 1
        cancelPendingARMeshRebuild()
        arMeshChunkLastScheduledTimes.removeAll()
        lastARCollisionCoveragePoint = nil
    }

    private func resumeARCollisionUpdates() {
        arCollisionUpdateGeneration += 1
        cancelPendingARMeshRebuild()
        arMeshChunks.removeAll()
        arMeshChunkLastScheduledTimes.removeAll()
        lastARCollisionCoveragePoint = nil
        lastARMeshSceneBuildTime = 0

        let anchors = latestARMeshAnchors.values.sorted { lhs, rhs in
            lhs.identifier.uuidString < rhs.identifier.uuidString
        }
        guard !anchors.isEmpty else {
            _ = rebuildARMuJoCoSceneIfNeeded(force: true, preservingState: true)
            return
        }

        for anchor in anchors {
            scheduleMuJoCoMeshChunkUpdate(for: anchor, force: true)
        }
    }

    private func updateMeshStatusUI() {
        guard meshUpdateLabel != nil else {
            return
        }

        let status: String
        let hFieldProgressText: String? = arHFieldTotalCount > 0
            ? "\(min(arHFieldProcessedCount, arHFieldTotalCount))/\(arHFieldTotalCount)"
            : nil
        let isBusy = arCollisionUpdatesEnabled && (
            arMeshProcessingCount > 0 ||
            arMeshRebuildPending ||
            arHFieldIsGenerating ||
            arMuJoCoIsApplying
        )
        if !arCollisionUpdatesEnabled {
            status = "Paused"
        } else if arMuJoCoIsApplying {
            status = hFieldProgressText.map { "MuJoCo \($0)" } ?? "MuJoCo Apply"
        } else if arHFieldIsGenerating {
            status = hFieldProgressText.map { "HField \($0)" } ?? "HField Gen"
        } else if arMeshProcessingCount > 0 {
            status = "Mesh Extract"
        } else if arMeshRebuildPending {
            status = hFieldProgressText.map { "HField Q \($0)" } ?? "HField Queued"
        } else if arMeshChunks.isEmpty {
            status = "Scanning"
        } else {
            status = "HField Ready"
        }

        meshUpdateLabel.text = status
        if isBusy {
            meshUpdateIndicator.startAnimating()
        } else {
            meshUpdateIndicator.stopAnimating()
        }
        updateCollisionUpdateButton()
    }

    private func updateARHFieldProgress(_ progress: MuJoCoSimulation.HeightFieldProgress) {
        let totalCount = max(0, progress.totalCount)
        arHFieldTotalCount = totalCount
        arHFieldProcessedCount = min(max(0, progress.processedCount), totalCount)
    }

    private func estimatedARHFieldProgress() async -> MuJoCoSimulation.HeightFieldProgress {
        let chunks = arMeshChunks.values.sorted { lhs, rhs in
            lhs.identifier.uuidString < rhs.identifier.uuidString
        }
        return await simulation.estimatedARCollisionHFieldProgress(
            chunks: chunks,
            coveragePoints: arCollisionCoveragePoints()
        )
    }

    private func resetARHFieldProgress() {
        arHFieldProcessedCount = 0
        arHFieldTotalCount = 0
    }

    private func updateCollisionUpdateButton() {
        guard collisionUpdateButton != nil else {
            return
        }

        if arCollisionUpdatesEnabled {
            collisionUpdateButton.setTitle("Pause", for: .normal)
            collisionUpdateButton.setTitleColor(.white, for: .normal)
            collisionUpdateButton.backgroundColor = UIColor.systemOrange.withAlphaComponent(0.80)
        } else {
            collisionUpdateButton.setTitle("Resume", for: .normal)
            collisionUpdateButton.setTitleColor(.white, for: .normal)
            collisionUpdateButton.backgroundColor = UIColor.systemGreen.withAlphaComponent(0.80)
        }
    }

    private func startARSession() {
#if !targetEnvironment(simulator)
        guard let arSession else {
            print("ARWorldTrackingConfiguration is not supported")
            return
        }

        let configuration = ARWorldTrackingConfiguration()
        configuration.planeDetection = [.horizontal, .vertical]
        if ARWorldTrackingConfiguration.supportsSceneReconstruction(.meshWithClassification) {
            configuration.sceneReconstruction = .meshWithClassification
        } else if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
            configuration.sceneReconstruction = .mesh
        }

        arSession.run(configuration, options: [.resetTracking, .removeExistingAnchors])
#endif
    }

    private func consumeDebugTerrainPhaseUpdateIfNeeded(at timestamp: CFTimeInterval) -> Double? {
        guard debugSceneUpdatesEnabled else {
            return nil
        }
        guard timestamp - lastDebugTerrainBuildTime > debugTerrainRebuildInterval else {
            return nil
        }

        debugTerrainPhase += 0.55
        lastDebugTerrainBuildTime = timestamp
        return debugTerrainPhase
    }

    @discardableResult
    private func rebuildARMuJoCoSceneIfNeeded(
        force: Bool = false,
        preservingState: Bool = true,
        progress: ((MuJoCoSimulation.LoadPhase) -> Void)? = nil
    ) -> Bool {
        guard isARSimulationActive else {
            return false
        }

        let now = CACurrentMediaTime()
        guard force || now - lastARMeshSceneBuildTime > arMeshSceneRebuildInterval else {
            return false
        }
        let chunks = arMeshChunks.values.sorted { lhs, rhs in
            lhs.identifier.uuidString < rhs.identifier.uuidString
        }
        guard !chunks.isEmpty else {
            if force {
                Task { [weak self] in
                    guard let self else {
                        return
                    }
                    await simulation.clear()
                    let scene = await simulation.renderScene()
                    updateRenderedSimulationScene(scene)
                }
            }
            return false
        }

        let coveragePoints = arCollisionCoveragePoints()
        Task { [weak self] in
            guard let self else {
                return
            }
            _ = await applyARMuJoCoScene(
                chunks: chunks,
                coveragePoints: coveragePoints,
                preservingState: preservingState,
                progress: progress
            )
        }
        return true
    }

    @discardableResult
    private func applyARMuJoCoScene(
        chunks: [EnvironmentMeshChunkDescriptor],
        coveragePoints: [SIMD2<Float>],
        preservingState: Bool,
        progress: ((MuJoCoSimulation.LoadPhase) -> Void)? = nil
    ) async -> Bool {
        guard isARSimulationActive, !chunks.isEmpty else {
            return false
        }

        progress?(.applyingMuJoCo)
        await simulation.load(
            environment: .arMeshChunks(chunks, coveragePoints: coveragePoints),
            preservingState: preservingState
        )
        let scene = await simulation.renderScene()
        updateRenderedSimulationScene(scene)
        lastARMeshSceneBuildTime = CACurrentMediaTime()
        lastARCollisionCoveragePoint = coveragePoints.last
        return true
    }

    private func updateARMuJoCoLoadPhase(_ phase: MuJoCoSimulation.LoadPhase) {
        switch phase {
        case .generatingHeightField(let progress):
            arHFieldIsGenerating = true
            arMuJoCoIsApplying = false
            updateARHFieldProgress(progress)
        case .applyingMuJoCo:
            arHFieldIsGenerating = false
            arMuJoCoIsApplying = true
        }
        updateMeshStatusUI()
        view.layoutIfNeeded()
    }

    private func scheduleARCollisionCoverageRebuildIfNeeded() {
        guard isARSimulationActive,
              arCollisionUpdatesEnabled,
              !arMeshChunks.isEmpty,
              pendingARMeshRebuildWorkItem == nil,
              let coveragePoint = arCollisionCoveragePoints().last else {
            return
        }

        if let lastARCollisionCoveragePoint,
           simd_distance(coveragePoint, lastARCollisionCoveragePoint) < arCollisionCoverageRebuildDistance {
            return
        }

        scheduleARMuJoCoSceneRebuild()
    }

    private func arCollisionCoveragePoints() -> [SIMD2<Float>] {
#if !targetEnvironment(simulator)
        guard isARSimulationActive,
              let frame = arSession?.currentFrame else {
            return []
        }

        let cameraTransform = frame.camera.transform
        let cameraAR = SIMD3<Float>(
            cameraTransform.columns.3.x,
            cameraTransform.columns.3.y,
            cameraTransform.columns.3.z
        )
        let forwardAR = -SIMD3<Float>(
            cameraTransform.columns.2.x,
            cameraTransform.columns.2.y,
            cameraTransform.columns.2.z
        )
        let camera = matrix4x4_muJoCoPosition(fromARPosition: cameraAR)
        let forward3 = matrix4x4_muJoCoDirection(fromARDirection: forwardAR)
        let forward2Raw = SIMD2<Float>(forward3.x, forward3.y)
        let forwardLength = simd_length(forward2Raw)
        let forward = forwardLength > 0.0001 ? forward2Raw / forwardLength : SIMD2<Float>(1, 0)
        let right = SIMD2<Float>(forward.y, -forward.x)
        let origin = SIMD2<Float>(camera.x, camera.y)

        var points: [SIMD2<Float>] = []
        points.reserveCapacity(15)
        for distance in [Float(0), 0.35, 0.75, 1.15, 1.55] {
            let center = origin + forward * distance
            let halfWidth = max(Float(0.35), distance * 0.38)
            points.append(center)
            points.append(center + right * halfWidth)
            points.append(center - right * halfWidth)
        }
        return points
#else
        return []
#endif
    }

    private func scheduleARMuJoCoSceneRebuild(force: Bool = false) {
        guard arCollisionUpdatesEnabled else {
            updateMeshStatusUI()
            return
        }
        pendingARMeshRebuildRequiresForce = pendingARMeshRebuildRequiresForce || force
        arMeshRebuildPending = true
        if !arHFieldIsGenerating && !arMuJoCoIsApplying {
            Task { [weak self] in
                guard let self else {
                    return
                }
                let progress = await estimatedARHFieldProgress()
                guard arMeshRebuildPending, !arHFieldIsGenerating, !arMuJoCoIsApplying else {
                    return
                }
                updateARHFieldProgress(progress)
                updateMeshStatusUI()
            }
        }
        updateMeshStatusUI()
        guard pendingARMeshRebuildWorkItem == nil else {
            return
        }

        let workItem = DispatchWorkItem { [weak self] in
            guard let self else {
                return
            }
            let shouldForce = pendingARMeshRebuildRequiresForce
            pendingARMeshRebuildWorkItem = nil
            pendingARMeshRebuildRequiresForce = false
            arMeshRebuildPending = false
            applyScheduledARMuJoCoSceneRebuild(force: shouldForce)
        }
        pendingARMeshRebuildWorkItem = workItem
        DispatchQueue.main.asyncAfter(deadline: .now() + arMeshSceneRebuildDelay, execute: workItem)
    }

    private func applyScheduledARMuJoCoSceneRebuild(force: Bool) {
        guard isARSimulationActive, arCollisionUpdatesEnabled else {
            updateMeshStatusUI()
            return
        }
        guard !arHFieldBuildInProgress else {
            arMeshRebuildPending = true
            pendingARMeshRebuildRequiresForce = pendingARMeshRebuildRequiresForce || force
            updateMeshStatusUI()
            return
        }

        let now = CACurrentMediaTime()
        guard force || now - lastARMeshSceneBuildTime > arMeshSceneRebuildInterval else {
            arHFieldIsGenerating = false
            arMuJoCoIsApplying = false
            updateMeshStatusUI()
            return
        }

        let chunks = arMeshChunks.values.sorted { lhs, rhs in
            lhs.identifier.uuidString < rhs.identifier.uuidString
        }
        guard !chunks.isEmpty else {
            if force {
                Task { [weak self] in
                    guard let self else {
                        return
                    }
                    await simulation.clear()
                    let scene = await simulation.renderScene()
                    updateRenderedSimulationScene(scene)
                }
            }
            arHFieldIsGenerating = false
            arMuJoCoIsApplying = false
            updateMeshStatusUI()
            return
        }

        let coveragePoints = arCollisionCoveragePoints()
        arHFieldIsGenerating = true
        arMuJoCoIsApplying = false
        updateMeshStatusUI()

        Task { [weak self] in
            guard let self else {
                return
            }

            let buildPlan = await simulation.makeARCollisionHFieldBuildPlan(
                chunks: chunks,
                coveragePoints: coveragePoints
            )
            guard isARSimulationActive, arCollisionUpdatesEnabled else {
                return
            }

            updateARHFieldProgress(buildPlan.initialProgress)
            updateMeshStatusUI()

            guard !buildPlan.requests.isEmpty else {
                _ = await applyARMuJoCoScene(
                    chunks: chunks,
                    coveragePoints: coveragePoints,
                    preservingState: true,
                    progress: updateARMuJoCoLoadPhase
                )
                arHFieldIsGenerating = false
                arMuJoCoIsApplying = false
                updateMeshStatusUI()
                return
            }

            arHFieldBuildInProgress = true
            arHFieldBuildGeneration += 1
            let buildGeneration = arHFieldBuildGeneration
            let collisionGeneration = arCollisionUpdateGeneration
            let initialProcessedCount = buildPlan.initialProgress.processedCount
            let totalCount = buildPlan.initialProgress.totalCount
            let requests = buildPlan.requests

            arHFieldBuildQueue.async { [weak self] in
                var results: [MuJoCoSimulation.ARCollisionHFieldBuildResult] = []
                results.reserveCapacity(requests.count)
                var processedCount = initialProcessedCount

                for request in requests {
                    let result = MuJoCoSimulation.buildARCollisionHField(request)
                    results.append(result)
                    processedCount += 1

                    DispatchQueue.main.async { [weak self, processedCount] in
                        guard let self,
                              isARSimulationActive,
                              arCollisionUpdatesEnabled,
                              arHFieldBuildInProgress,
                              arHFieldBuildGeneration == buildGeneration,
                              arCollisionUpdateGeneration == collisionGeneration else {
                            return
                        }
                        updateARHFieldProgress(MuJoCoSimulation.HeightFieldProgress(
                            processedCount: processedCount,
                            totalCount: totalCount
                        ))
                        updateMeshStatusUI()
                    }
                }

                DispatchQueue.main.async { [weak self] in
                    guard let self,
                          isARSimulationActive,
                          arCollisionUpdatesEnabled,
                          arHFieldBuildInProgress,
                          arHFieldBuildGeneration == buildGeneration,
                          arCollisionUpdateGeneration == collisionGeneration else {
                        return
                    }

                    Task { [weak self] in
                        guard let self else {
                            return
                        }
                        await simulation.installARCollisionHFieldBuildResults(results)
                        _ = await applyARMuJoCoScene(
                            chunks: chunks,
                            coveragePoints: coveragePoints,
                            preservingState: true,
                            progress: updateARMuJoCoLoadPhase
                        )
                        arHFieldBuildInProgress = false
                        arHFieldIsGenerating = false
                        arMuJoCoIsApplying = false
                        updateMeshStatusUI()

                        if arMeshRebuildPending, pendingARMeshRebuildWorkItem == nil {
                            let shouldForce = pendingARMeshRebuildRequiresForce
                            pendingARMeshRebuildRequiresForce = false
                            scheduleARMuJoCoSceneRebuild(force: shouldForce)
                        }
                    }
                }
            }
        }
    }

    private func scheduleMuJoCoMeshChunkUpdate(for anchor: ARMeshAnchor, force: Bool = false) {
        guard arCollisionUpdatesEnabled else {
            updateMeshStatusUI()
            return
        }

        let now = CACurrentMediaTime()
        if !force,
           let lastScheduledTime = arMeshChunkLastScheduledTimes[anchor.identifier],
           now - lastScheduledTime < arMeshChunkUpdateInterval {
            return
        }

        arMeshChunkLastScheduledTimes[anchor.identifier] = now
        arMeshChunkRevision += 1
        let revision = arMeshChunkRevision
        let generation = arCollisionUpdateGeneration
        arMeshProcessingCount += 1
        updateMeshStatusUI()

        arMeshProcessingQueue.async { [weak self, anchor] in
            let chunk = Self.makeMuJoCoMeshChunk(from: anchor, revision: revision)
            DispatchQueue.main.async { [weak self] in
                guard let self else {
                    return
                }
                arMeshProcessingCount = max(0, arMeshProcessingCount - 1)
                guard isARSimulationActive,
                      arCollisionUpdatesEnabled,
                      generation == arCollisionUpdateGeneration else {
                    updateMeshStatusUI()
                    return
                }
                if let chunk {
                    arMeshChunks[anchor.identifier] = chunk
                } else {
                    arMeshChunks.removeValue(forKey: anchor.identifier)
                }
                scheduleARMuJoCoSceneRebuild(force: force)
                updateMeshStatusUI()
            }
        }
    }

    nonisolated private static func makeMuJoCoMeshChunk(
        from anchor: ARMeshAnchor,
        revision: Int
    ) -> EnvironmentMeshChunkDescriptor? {
        let geometry = anchor.geometry
        var vertices: [SIMD3<Float>] = []
        var indices: [UInt32] = []
        var floorFaceMask: [Bool]? = geometry.hasFaceClassification ? [] : nil
        vertices.reserveCapacity(geometry.vertices.count)
        indices.reserveCapacity(geometry.faces.count * geometry.faces.indexCountPerPrimitive)

        for vertexIndex in 0..<geometry.vertices.count {
            let localVertex = geometry.vertex(at: vertexIndex)
            let arPosition4 = anchor.transform * SIMD4<Float>(localVertex.x, localVertex.y, localVertex.z, 1)
            let arPosition = SIMD3<Float>(arPosition4.x, arPosition4.y, arPosition4.z)
            vertices.append(matrix4x4_muJoCoPosition(fromARPosition: arPosition))
        }

        let vertexCount = UInt32(vertices.count)
        let anchorIndices = geometry.faces.triangleIndices()
        for faceStart in stride(from: 0, to: anchorIndices.count - 2, by: 3) {
            let faceIndex = faceStart / 3
            let a = anchorIndices[faceStart]
            let b = anchorIndices[faceStart + 1]
            let c = anchorIndices[faceStart + 2]
            guard a < vertexCount, b < vertexCount, c < vertexCount else {
                continue
            }
            indices.append(contentsOf: [a, b, c])
            floorFaceMask?.append(geometry.isFloorFace(at: faceIndex))
        }

        guard vertices.count >= 3, indices.count >= 3 else {
            return nil
        }
        return EnvironmentMeshChunkDescriptor(
            identifier: anchor.identifier,
            vertices: vertices,
            indices: indices,
            floorFaceMask: floorFaceMask,
            revision: revision
        )
    }
}

extension GameViewController {
    func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer, shouldReceive touch: UITouch) -> Bool {
        guard sceneMode == .debug || sceneMode == .ar else {
            return false
        }
        if touch.view?.isDescendant(of: modeControl) == true {
            return false
        }
        if touch.view?.isDescendant(of: cameraModeControl) == true {
            return false
        }
        if touch.view?.isDescendant(of: panelContainerView) == true {
            return false
        }
        if let heightFieldMapView,
           touch.view?.isDescendant(of: heightFieldMapView) == true {
            return false
        }
        if let joystickControlView,
           touch.view?.isDescendant(of: joystickControlView) == true {
            return false
        }
        return true
    }

    func gestureRecognizer(
        _ gestureRecognizer: UIGestureRecognizer,
        shouldRecognizeSimultaneouslyWith otherGestureRecognizer: UIGestureRecognizer
    ) -> Bool {
        sceneMode == .debug
    }
}

#if !targetEnvironment(simulator)
extension GameViewController: ARSessionDelegate {
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard isARSimulationActive else {
            return
        }
        scheduleARCollisionCoverageRebuildIfNeeded()
        renderer.updateFrame(frame)
        renderer.updateMuJoCoScene(renderSceneForDisplay())
        mtkView.draw()
    }

    func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
        for case let meshAnchor as ARMeshAnchor in anchors {
            latestARMeshAnchors[meshAnchor.identifier] = meshAnchor
            if arCollisionUpdatesEnabled {
                scheduleMuJoCoMeshChunkUpdate(for: meshAnchor, force: true)
            }
            renderer.updateMesh(for: meshAnchor)
        }
        updateMeshStatusUI()
    }

    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        for case let meshAnchor as ARMeshAnchor in anchors {
            latestARMeshAnchors[meshAnchor.identifier] = meshAnchor
            if arCollisionUpdatesEnabled {
                scheduleMuJoCoMeshChunkUpdate(for: meshAnchor)
            }
            renderer.updateMesh(for: meshAnchor)
        }
        updateMeshStatusUI()
    }

    func session(_ session: ARSession, didRemove anchors: [ARAnchor]) {
        for case let meshAnchor as ARMeshAnchor in anchors {
            latestARMeshAnchors.removeValue(forKey: meshAnchor.identifier)
            arMeshChunks.removeValue(forKey: meshAnchor.identifier)
            arMeshChunkLastScheduledTimes.removeValue(forKey: meshAnchor.identifier)
            renderer.removeMesh(for: meshAnchor)
        }
        if arCollisionUpdatesEnabled {
            scheduleARMuJoCoSceneRebuild(force: true)
        }
        updateMeshStatusUI()
    }
}
#endif

private extension ARGeometrySource {
    nonisolated func vector3(at index: Int) -> SIMD3<Float> {
        let pointer = buffer.contents().advanced(by: offset + stride * index)
        let floats = pointer.assumingMemoryBound(to: Float.self)
        return SIMD3<Float>(floats[0], floats[1], floats[2])
    }
}

private extension ARMeshGeometry {
    nonisolated var hasFaceClassification: Bool {
        classification != nil
    }

    nonisolated func vertex(at index: Int) -> SIMD3<Float> {
        vertices.vector3(at: index)
    }

    nonisolated func isFloorFace(at index: Int) -> Bool {
        guard let classification else {
            return false
        }
        let pointer = classification.buffer.contents().advanced(by: classification.offset + classification.stride * index)
        let value = pointer.assumingMemoryBound(to: UInt8.self).pointee
        return ARMeshClassification(rawValue: Int(value)) == .floor
    }
}

private extension ARGeometryElement {
    nonisolated func triangleIndices() -> [UInt32] {
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
