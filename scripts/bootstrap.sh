#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MJLAB_URL="${MJLAB_URL:-https://github.com/mujocolab/mjlab.git}"

check_only=0
skip_network=0
with_asset_sources=0

usage() {
  cat <<'USAGE'
Usage: scripts/bootstrap.sh [options]

Validates the app checkout and, when network access is allowed, asks Xcode to
resolve Swift package dependencies. MuJoCo and RecastNavigationKit are normal
remote Swift packages; this script does not clone them into the repository.

Options:
  --check-only           Validate the current checkout without resolving packages.
  --skip-network         Do not clone optional asset sources or resolve packages.
  --with-asset-sources   Also prepare/check mjlab for render-asset regeneration.
  -h, --help             Show this help.

Environment:
  MJLAB_URL              Git URL for mjlab when --with-asset-sources is used.
                         Default: https://github.com/mujocolab/mjlab.git

Example:
  ./scripts/bootstrap.sh
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-only)
      check_only=1
      ;;
    --skip-network)
      skip_network=1
      ;;
    --with-asset-sources)
      with_asset_sources=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

log() {
  printf '[bootstrap] %s\n' "$*"
}

warn() {
  printf '[bootstrap] warning: %s\n' "$*" >&2
}

error() {
  printf '[bootstrap] error: %s\n' "$*" >&2
}

require_command() {
  local command_name="$1"

  if ! command -v "$command_name" >/dev/null 2>&1; then
    warn "$command_name was not found on PATH"
  fi
}

clone_if_missing() {
  local path="$1"
  local url="$2"
  local env_name="$3"

  if [[ -d "$path" ]]; then
    log "$path exists"
    return 0
  fi

  if [[ "$check_only" -eq 1 || "$skip_network" -eq 1 ]]; then
    error "$path is missing"
    return 1
  fi

  if [[ -z "$url" ]]; then
    error "$path is missing and $env_name is not set"
    return 1
  fi

  log "cloning optional asset source $path from $url"
  git clone "$url" "$path"
}

missing=0

require_path() {
  local path="$1"
  local message="$2"

  if [[ ! -e "$path" ]]; then
    error "$message: $path"
    missing=1
  fi
}

require_dir() {
  local path="$1"
  local message="$2"

  if [[ ! -d "$path" ]]; then
    error "$message: $path"
    missing=1
  fi
}

log "checking tools"
require_command xcodebuild
if [[ "$with_asset_sources" -eq 1 ]]; then
  require_command git
  require_command node
fi

if [[ "$with_asset_sources" -eq 1 ]]; then
  log "preparing optional asset source checkout"
  if ! clone_if_missing "mjlab" "$MJLAB_URL" "MJLAB_URL"; then
    missing=1
  fi
fi

log "validating Xcode project"
require_path "MujocoAR.xcodeproj/project.pbxproj" "Xcode project is missing"
require_path "MujocoAR/Info.plist" "Info.plist is missing"

if [[ "$with_asset_sources" -eq 1 ]]; then
  log "validating source asset checkout"
  require_dir "mjlab/src/mjlab/asset_zoo/robots/unitree_go1/xmls/assets" "Go1 source meshes are missing"
  require_dir "mjlab/src/mjlab/asset_zoo/robots/unitree_g1/xmls/assets" "G1 source meshes are missing"
fi

if [[ "$with_asset_sources" -eq 1 && "$missing" -eq 0 ]]; then
  log "regenerating render assets"
  if ! node scripts/prepare_ios_render_assets.mjs; then
    error "failed to regenerate render assets"
    missing=1
  fi
fi

log "validating bundled runtime resources"
require_path "MujocoAR/Resources/go1_flat_scene.xml" "Go1 flat scene is missing"
require_path "MujocoAR/Resources/go1_rough_scene.xml" "Go1 rough scene is missing"
require_path "MujocoAR/Resources/g1_flat_scene.xml" "G1 flat scene is missing"
require_path "MujocoAR/Resources/g1_rough_scene.xml" "G1 rough scene is missing"
require_dir "MujocoAR/Resources/render_assets/go1" "Go1 render assets are missing"
require_dir "MujocoAR/Resources/render_assets/g1" "G1 render assets are missing"
require_path "MujocoAR/Resources/render_manifests/go1_render_manifest.json" "Go1 render manifest is missing"
require_path "MujocoAR/Resources/render_manifests/g1_render_manifest.json" "G1 render manifest is missing"
require_path "MujocoAR/Resources/go1_velocity_rough.mlpackage/Manifest.json" "Go1 Core ML package is missing"
require_path "MujocoAR/Resources/g1_velocity_rough.mlpackage/Manifest.json" "G1 Core ML package is missing"

if [[ "$missing" -ne 0 ]]; then
  cat >&2 <<'NEXT'

Bootstrap did not complete.

Common fixes:
  - Run Xcode package resolution for the remote Swift packages.
  - Run with --with-asset-sources to clone mjlab and generate the render assets.
NEXT
  exit 1
fi

if [[ "$check_only" -eq 0 && "$skip_network" -eq 0 ]] && command -v xcodebuild >/dev/null 2>&1; then
  log "resolving Xcode package references"
  xcodebuild -resolvePackageDependencies -project MujocoAR.xcodeproj -scheme MujocoAR
fi

log "setup looks ready"
