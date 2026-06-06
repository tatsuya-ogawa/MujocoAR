# Third-Party Notices

This repository bundles generated runtime assets and uses local external packages. Keep this file updated when changing assets, policy models, or package sources.

## Runtime Packages

- MuJoCo iOS package: provided by the remote Swift package `https://github.com/tatsuya-ogawa/mujoco-ios.git`. That package includes MuJoCo notices in `THIRD_PARTY_NOTICES.md`; MuJoCo is distributed under the Apache License 2.0.
- Recast/Detour navigation wrapper: provided by the remote Swift package `https://github.com/tatsuya-ogawa/RecastNavigationKit.git`. That package includes its own `LICENSE`; upstream Recast/Detour notices should remain with that package.
- `mjlab` (retrieved from `https://github.com/mujocolab/mjlab`): used as a source for robot XML/mesh assets and policy training/export workflows. The local checkout is Apache License 2.0 and notes inherited components in its README and source headers.

## Bundled Assets And Models

The app currently includes:

- Optimized Go1 and G1 MuJoCo XML scenes in `MujocoAR/Resources`.
- GLB robot render assets in `MujocoAR/Resources/render_assets`.
- Core ML policy packages in `MujocoAR/Resources`.
- TorchScript policy checkpoints in `coreml`.

Before public distribution, verify that each generated asset and policy checkpoint may be redistributed under the repository's chosen license. If redistribution rights are uncertain, replace the checked-in artifact with a documented download or generation step.
