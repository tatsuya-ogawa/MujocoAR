# AR Scene Reconstruction Pipeline

This document describes how MujocoAR turns ARKit scene reconstruction data into
MuJoCo collision terrain, debug visualization, and navigation input.

## Overview

In AR mode, the app runs an `ARWorldTrackingConfiguration` with horizontal and
vertical plane detection. When the device supports ARKit scene reconstruction,
the session prefers `.meshWithClassification`; otherwise it falls back to
`.mesh`.

The pipeline is:

1. ARKit produces `ARMeshAnchor` updates.
2. Each anchor is converted into an `EnvironmentMeshChunkDescriptor`.
3. Mesh chunks are throttled and queued for a MuJoCo scene rebuild.
4. The simulation builds one MuJoCo height field per included chunk.
5. Height fields are combined into one collision terrain description.
6. The resulting terrain is used for MuJoCo contacts, rendering, and Recast
   navigation mesh generation.

The conversion is intentionally based on ARKit-provided geometry values. It does
not apply an arbitrary downward offset to make the terrain sit under the scanned
floor.

## AR Session Setup

AR mode is selected by default on devices where `ARWorldTrackingConfiguration` is
available. The session configuration enables plane detection and requests the
best available scene reconstruction mode:

- `.meshWithClassification` when ARKit can classify reconstructed mesh faces.
- `.mesh` when classification is unavailable but raw reconstruction is supported.

The simulator path does not run ARKit and is not expected to provide collision
terrain.

## Mesh Chunk Ingestion

`GameViewController` receives `ARMeshAnchor` objects from `ARSessionDelegate`.
For each mesh anchor, it builds a chunk descriptor with:

- A stable anchor identifier.
- Vertex positions transformed from anchor-local ARKit coordinates into world
  AR coordinates, then into the app's MuJoCo coordinate convention.
- Triangle indices copied from the AR geometry element.
- An optional `floorFaceMask` when ARKit mesh classification is available.

The app keeps the latest chunk per anchor. Anchor updates are throttled by
`arMeshChunkUpdateInterval` so rapidly changing ARKit mesh data does not trigger
a full MuJoCo rebuild for every frame.

## Rebuild Scheduling

Scene rebuilds are debounced. When chunks change, the controller marks an AR mesh
rebuild as pending and schedules work after `arMeshSceneRebuildDelay`.

Missing height fields are prebuilt on a dedicated background queue. MuJoCo model
loading, `mjData` mutation, stepping, cached height-field installation, and render
snapshot extraction then run through the dedicated `MuJoCoSimulation` actor
instead of the main actor. The main thread still owns UIKit, ARKit session state,
and renderer handoff, but it receives immutable `MuJoCoRenderScene` snapshots
rather than touching live `mjModel` or `mjData` pointers.

The UI label can show several mesh states:

- `Scanning`: AR is active but no collision mesh is ready yet.
- `HField Q x/y`: height field generation is queued.
- `HField x/y`: height fields are being generated.
- `MuJoCo x/y`: height fields are ready and MuJoCo is applying the new XML.
- `HField Ready`: the current height fields are ready.

The `x/y` count is based on the clipped set of chunks that will actually be used
for collision. Cached chunks are counted before generation starts, so a rebuild
that reuses previous height fields should show progress from `n/y` instead of
always starting at `0/y`.

## Chunk Clipping

The collision pipeline caps the amount of AR mesh data included in a rebuild:

- `maxARCollisionVertexCount`
- `maxARCollisionIndexCount`

Chunks are sorted by UUID for deterministic processing. When a chunk must be
partially included, indices are filtered so they only reference retained
vertices. If a floor classification mask exists, it is clipped in the same face
order as the retained triangle indices.

The same clipping helper is used for both progress estimation and actual
height-field generation, which keeps the UI count aligned with the work done by
the simulation.

## Floor Surface Selection

Height fields are generated from floor-like triangles. The preferred source is
ARKit's mesh classification:

1. If at least two triangles are classified as `.floor`, those triangles are used.
2. Otherwise, the app falls back to geometric filtering:
   - Degenerate triangles are ignored.
   - Triangle winding is normalized so upward normals have positive Z in MuJoCo
     space.
   - Triangles with sufficiently upward-facing normals are considered horizontal
     candidates.
   - A floor reference is computed from the lower part of the candidate height
     distribution, and high outliers are filtered away.

Classification is used only as a source selection signal. The generated height
values still come from the actual triangle vertices.

## Height Field Construction

For each included chunk, `MuJoCoSimulation` builds a MuJoCo `hfield`:

1. The selected floor triangles define the XY bounds.
2. Recent coverage points around the camera expand those bounds, so the collision
   surface covers the area where the user is likely moving or spawning a robot.
3. A regular grid is created over the bounds.
4. For each selected floor triangle, the grid points covered by that triangle are
   filled by barycentric interpolation on the triangle plane.
5. Grid points not covered by any triangle fall back to the nearest observed
   floor sample from the same chunk.
6. The resulting heights are normalized into MuJoCo `hfield` elevation values.

If multiple triangles contribute to the same grid point, the lower actual sampled
height is used. This avoids protruding collision caused by overlapping or noisy
mesh faces without adding a global sink offset.

## Detail Mode

The AR overlay includes an `HField Detail` switch. It controls the resolution
used by generated collision height fields:

- Normal mode uses a coarser grid for cheaper rebuilds.
- Detail mode uses a denser grid and a larger fallback sample budget.

Changing this setting clears the AR height-field cache because cached geometry is
resolution-dependent.

## Height Field Cache

Generated height fields are cached per AR mesh anchor. A cached entry is reused
only when all of the following match:

- Anchor revision.
- Retained vertex count.
- Retained index count.
- Coverage-point signature.
- Detail-mode setting.

When a chunk is unchanged, the cached `ARCollisionGeometry` is appended directly
without rebuilding that height field. The progress label uses the same cache
matching rules to estimate how many chunks are already processed.

## MuJoCo XML Generation

Each height field produces:

- An `asset` entry containing the normalized `hfield` elevations.
- A collision `geom` referencing the height field.
- A render mesh used by the Metal renderer.
- A navigation mesh descriptor used by Recast.

When at least one height field is generated, the app uses the combined
height-field collision geometry. If height-field generation fails for all chunks,
the app falls back to raw AR mesh collision assets and builds navigation data from
the selected floor surface when possible.

## Navigation And Controls

The same terrain representation feeds robot navigation:

- Recast builds a navigation mesh from the generated terrain render mesh.
- Tap-to-navigate uses that navigation mesh in the existing nav mode.
- Direct control mode bypasses navigation targets and sends velocity commands
  from the joystick UI.

This keeps AR and debug behavior consistent: both modes can spawn a robot, reset
it after a fall, and switch between navigation and direct drive controls.

## Known Tradeoffs

- ARKit classification quality depends on device support, scanning angle, and
  environment texture. When classification is unavailable or unstable, the app
  falls back to geometry-based floor detection.
- The height field is a grid approximation of AR mesh triangles, not the raw mesh
  itself. Detail mode improves spatial resolution at higher rebuild cost.
- Coverage points intentionally bias the collision bounds toward the camera and
  nearby user interaction area. This keeps rebuilds bounded while the scanned
  world grows.
- Generic iOS builds can verify compilation, but the AR collision pipeline needs
  a physical ARKit-capable device for meaningful runtime validation.
