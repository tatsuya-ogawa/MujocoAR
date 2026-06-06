#!/usr/bin/env node

import { access, cp, mkdir, rm, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const mjlabRoot = process.env.MJLAB_ROOT
  ? path.resolve(process.env.MJLAB_ROOT)
  : path.join(repoRoot, "mjlab");
const appResources = path.join(repoRoot, "MujocoAR", "Resources");

const mjlabAssetDirs = [
  path.join(mjlabRoot, "src", "mjlab", "asset_zoo", "robots", "unitree_go1", "xmls", "assets"),
  path.join(mjlabRoot, "src", "mjlab", "asset_zoo", "robots", "unitree_g1", "xmls", "assets"),
];

const robotConfigs = [
  {
    id: "g1",
    envIds: ["g1_flat", "g1_rough"],
    renderAssetDir: path.join(appResources, "render_assets", "g1"),
    renderAssetUrlPrefix: "/render_assets/g1",
  },
  {
    id: "go1",
    envIds: ["go1_flat", "go1_rough"],
    renderAssetDir: path.join(appResources, "render_assets", "go1"),
    renderAssetUrlPrefix: "/render_assets/go1",
  },
];

const mjlabRobotAssetDir = (robotId) => {
  return robotId === "g1"
    ? path.join(mjlabRoot, "src", "mjlab", "asset_zoo", "robots", "unitree_g1", "xmls", "assets")
    : path.join(mjlabRoot, "src", "mjlab", "asset_zoo", "robots", "unitree_go1", "xmls", "assets");
};

function parseAttrs(tag) {
  const attrs = {};
  const re = /([A-Za-z0-9_:/.-]+)="([^"]*)"/g;
  for (let match = re.exec(tag); match; match = re.exec(tag)) {
    attrs[match[1]] = match[2];
  }
  return attrs;
}

function parseVec(value, fallback) {
  if (!value) {
    return fallback;
  }
  const values = value.trim().split(/\s+/).map(Number);
  return values.every(Number.isFinite) ? values : fallback;
}

function leafName(name) {
  return name.replace(/^robot\//, "");
}

function glbFileName(stlFileName) {
  return stlFileName.replace(/\.[^.]+$/, ".glb");
}

function materialColor(materials, materialName, rgbaAttr) {
  if (rgbaAttr) {
    return parseVec(rgbaAttr, [0.7, 0.7, 0.7, 1]);
  }
  return materials.get(materialName) ?? [0.7, 0.7, 0.7, 1];
}

function loadSceneMetadata(xml, renderAssetUrlPrefix) {
  const meshes = new Map();
  const materials = new Map();

  for (const match of xml.matchAll(/<mesh\b[^>]*>/g)) {
    const attrs = parseAttrs(match[0]);
    if (attrs.name?.startsWith("robot/") && attrs.file?.toLowerCase().endsWith(".stl")) {
      meshes.set(attrs.name, attrs.file);
    }
  }

  for (const match of xml.matchAll(/<material\b[^>]*>/g)) {
    const attrs = parseAttrs(match[0]);
    if (attrs.name && attrs.rgba) {
      materials.set(attrs.name, parseVec(attrs.rgba, [0.7, 0.7, 0.7, 1]));
    }
  }

  const parts = [];
  const bodyStack = [];
  const tagRe = /<!--[\s\S]*?-->|<[^!?][^>]*>/g;
  for (const match of xml.matchAll(tagRe)) {
    const tag = match[0];
    if (tag.startsWith("<!--")) {
      continue;
    }
    if (/^<\/body\b/.test(tag)) {
      bodyStack.pop();
      continue;
    }

    const openBody = /^<body\b/.test(tag);
    if (openBody) {
      const attrs = parseAttrs(tag);
      if (attrs.name) {
        bodyStack.push(attrs.name);
      }
      if (tag.endsWith("/>")) {
        bodyStack.pop();
      }
      continue;
    }

    if (!/^<geom\b/.test(tag)) {
      continue;
    }
    const attrs = parseAttrs(tag);
    const meshFile = attrs.mesh ? meshes.get(attrs.mesh) : undefined;
    if (!meshFile || attrs.class !== "robot/visual") {
      continue;
    }

    const bodyName = bodyStack.at(-1);
    if (!bodyName) {
      throw new Error(`Visual mesh ${attrs.mesh} has no owning body`);
    }

    const meshLeaf = leafName(attrs.mesh);
    const bodyLeaf = leafName(bodyName);
    const index = parts.length;
    parts.push({
      key: `visual:${index}:${attrs.mesh}`,
      label: meshLeaf === bodyLeaf ? meshLeaf : `${meshLeaf} @ ${bodyLeaf}`,
      bodyName,
      meshName: attrs.mesh,
      url: `${renderAssetUrlPrefix}/${glbFileName(meshFile)}`,
      pos: parseVec(attrs.pos, [0, 0, 0]),
      quat: parseVec(attrs.quat, [1, 0, 0, 0]),
      rgba: materialColor(materials, attrs.material, attrs.rgba),
      materialName: attrs.material ?? "robot/silver",
    });
  }

  return { meshes, parts };
}

function optimizedXml(xml) {
  const keep = xml.split(/\r?\n/).filter((line) => {
    const isRobotStlMesh =
      /<mesh\b/.test(line) &&
      /\bname="robot\/[^"]+"/.test(line) &&
      /\bfile="[^"]+\.stl"/i.test(line);
    const isRobotVisualMeshGeom =
      /<geom\b/.test(line) &&
      /\bclass="robot\/visual"/.test(line) &&
      /\bmesh="robot\/[^"]+"/.test(line);
    return !isRobotStlMesh && !isRobotVisualMeshGeom;
  });
  return `${keep.join("\n")}${xml.endsWith("\n") ? "\n" : ""}`;
}

function parseStl(buffer) {
  if (buffer.length >= 84) {
    const triangleCount = buffer.readUInt32LE(80);
    const binaryLength = 84 + triangleCount * 50;
    if (binaryLength <= buffer.length) {
      return parseBinaryStl(buffer, triangleCount);
    }
  }
  return parseAsciiStl(buffer.toString("utf8"));
}

function parseBinaryStl(buffer, triangleCount) {
  const triangles = [];
  let offset = 84;
  for (let i = 0; i < triangleCount; i += 1) {
    offset += 12;
    const tri = [];
    for (let vertex = 0; vertex < 3; vertex += 1) {
      tri.push([
        buffer.readFloatLE(offset),
        buffer.readFloatLE(offset + 4),
        buffer.readFloatLE(offset + 8),
      ]);
      offset += 12;
    }
    offset += 2;
    triangles.push(tri);
  }
  return triangles;
}

function parseAsciiStl(text) {
  const vertices = [];
  for (const match of text.matchAll(/vertex\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)/gi)) {
    vertices.push([Number(match[1]), Number(match[2]), Number(match[3])]);
  }
  const triangles = [];
  for (let i = 0; i + 2 < vertices.length; i += 3) {
    triangles.push([vertices[i], vertices[i + 1], vertices[i + 2]]);
  }
  return triangles;
}

function normalize(value) {
  const length = Math.hypot(value[0], value[1], value[2]);
  if (length < 1e-12) {
    return [0, 0, 1];
  }
  return [value[0] / length, value[1] / length, value[2] / length];
}

function subtract(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function quantizedKey(vertex) {
  return `${Math.round(vertex[0] * 1e6)},${Math.round(vertex[1] * 1e6)},${Math.round(vertex[2] * 1e6)}`;
}

function buildIndexedMesh(triangles) {
  const vertexMap = new Map();
  const vertices = [];
  const normals = [];
  const indices = [];

  const addVertex = (vertex) => {
    const key = quantizedKey(vertex);
    const existing = vertexMap.get(key);
    if (existing !== undefined) {
      return existing;
    }
    const index = vertices.length / 3;
    vertexMap.set(key, index);
    vertices.push(vertex[0], vertex[1], vertex[2]);
    normals.push(0, 0, 0);
    return index;
  };

  for (const tri of triangles) {
    const a = addVertex(tri[0]);
    const b = addVertex(tri[1]);
    const c = addVertex(tri[2]);
    const normal = normalize(cross(subtract(tri[1], tri[0]), subtract(tri[2], tri[0])));
    for (const index of [a, b, c]) {
      normals[index * 3] += normal[0];
      normals[index * 3 + 1] += normal[1];
      normals[index * 3 + 2] += normal[2];
    }
    indices.push(a, b, c);
  }

  for (let i = 0; i < normals.length; i += 3) {
    const normal = normalize([normals[i], normals[i + 1], normals[i + 2]]);
    normals[i] = normal[0];
    normals[i + 1] = normal[1];
    normals[i + 2] = normal[2];
  }

  return { vertices, normals, indices };
}

function align4(value) {
  return (value + 3) & ~3;
}

function padBuffer(buffer, byte = 0) {
  const paddedLength = align4(buffer.length);
  if (paddedLength === buffer.length) {
    return buffer;
  }
  const padded = Buffer.alloc(paddedLength, byte);
  buffer.copy(padded);
  return padded;
}

function typedBuffer(array) {
  return Buffer.from(array.buffer, array.byteOffset, array.byteLength);
}

function writeGlb(meshName, mesh) {
  const positionArray = new Float32Array(mesh.vertices);
  const normalArray = new Float32Array(mesh.normals);
  const useUint32 = mesh.vertices.length / 3 > 65535;
  const indexArray = useUint32 ? new Uint32Array(mesh.indices) : new Uint16Array(mesh.indices);
  const binChunks = [];
  const bufferViews = [];
  let byteOffset = 0;

  const pushView = (buffer, target) => {
    const alignedOffset = align4(byteOffset);
    if (alignedOffset > byteOffset) {
      binChunks.push(Buffer.alloc(alignedOffset - byteOffset));
      byteOffset = alignedOffset;
    }
    bufferViews.push({
      buffer: 0,
      byteOffset,
      byteLength: buffer.length,
      target,
    });
    binChunks.push(buffer);
    byteOffset += buffer.length;
  };

  pushView(typedBuffer(positionArray), 34962);
  pushView(typedBuffer(normalArray), 34962);
  pushView(typedBuffer(indexArray), 34963);

  const binChunk = padBuffer(Buffer.concat(binChunks));
  const positionCount = positionArray.length / 3;
  const min = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY];
  const max = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];
  for (let i = 0; i < positionArray.length; i += 3) {
    for (let axis = 0; axis < 3; axis += 1) {
      min[axis] = Math.min(min[axis], positionArray[i + axis]);
      max[axis] = Math.max(max[axis], positionArray[i + axis]);
    }
  }

  const json = {
    asset: {
      version: "2.0",
      generator: "MujocoAR prepare-ios-render-assets",
    },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0, name: meshName }],
    meshes: [
      {
        name: meshName,
        primitives: [
          {
            attributes: { POSITION: 0, NORMAL: 1 },
            indices: 2,
            mode: 4,
          },
        ],
      },
    ],
    accessors: [
      {
        bufferView: 0,
        componentType: 5126,
        count: positionCount,
        type: "VEC3",
        min,
        max,
      },
      {
        bufferView: 1,
        componentType: 5126,
        count: positionCount,
        type: "VEC3",
      },
      {
        bufferView: 2,
        componentType: useUint32 ? 5125 : 5123,
        count: indexArray.length,
        type: "SCALAR",
        min: [0],
        max: [positionCount - 1],
      },
    ],
    bufferViews,
    buffers: [{ byteLength: binChunk.length }],
  };

  const jsonChunk = padBuffer(Buffer.from(JSON.stringify(json)), 0x20);
  const length = 12 + 8 + jsonChunk.length + 8 + binChunk.length;
  const output = Buffer.alloc(length);
  let offset = 0;
  output.writeUInt32LE(0x46546c67, offset);
  offset += 4;
  output.writeUInt32LE(2, offset);
  offset += 4;
  output.writeUInt32LE(length, offset);
  offset += 4;
  output.writeUInt32LE(jsonChunk.length, offset);
  offset += 4;
  output.writeUInt32LE(0x4e4f534a, offset);
  offset += 4;
  jsonChunk.copy(output, offset);
  offset += jsonChunk.length;
  output.writeUInt32LE(binChunk.length, offset);
  offset += 4;
  output.writeUInt32LE(0x004e4942, offset);
  offset += 4;
  binChunk.copy(output, offset);
  return output;
}

async function convertStl(config, fileName) {
  const sourcePath = path.join(mjlabRobotAssetDir(config.id), fileName);
  const targetPath = path.join(config.renderAssetDir, glbFileName(fileName));
  const source = await readFile(sourcePath);
  const triangles = parseStl(source);
  const mesh = buildIndexedMesh(triangles);
  const glb = writeGlb(fileName, mesh);
  await writeFile(targetPath, glb);
  return {
    fileName,
    triangles: triangles.length,
    vertices: mesh.vertices.length / 3,
    sourceBytes: source.length,
    glbBytes: glb.length,
  };
}

async function requirePath(target, message) {
  try {
    await access(target);
  } catch {
    throw new Error(`${message}: ${target}`);
  }
}

async function main() {
  for (const assetDir of mjlabAssetDirs) {
    await requirePath(
      assetDir,
      "Missing mjlab robot source assets. Set MJLAB_ROOT or run scripts/bootstrap.sh --with-asset-sources",
    );
  }

  await rm(path.join(appResources, "render_assets"), { recursive: true, force: true });
  await mkdir(path.join(appResources, "render_assets"), { recursive: true });
  await mkdir(path.join(appResources, "render_manifests"), { recursive: true });

  for (const config of robotConfigs) {
    await mkdir(config.renderAssetDir, { recursive: true });

    // Use flat scene XML template to identify meshes to optimize for this robot
    const sourceXmlPath = path.join(repoRoot, "scripts", "raw_scenes", `${config.id}_flat`, "scene.xml");
    await requirePath(sourceXmlPath, `Raw scene file not found`);
    const sourceXml = await readFile(sourceXmlPath, "utf8");
    const { meshes } = loadSceneMetadata(sourceXml, config.renderAssetUrlPrefix);
    const uniqueFiles = [...new Set(meshes.values())].sort();

    const conversionStats = [];
    for (const fileName of uniqueFiles) {
      conversionStats.push(await convertStl(config, fileName));
    }

    for (const envId of config.envIds) {
      const sceneXmlPath = path.join(repoRoot, "scripts", "raw_scenes", envId, "scene.xml");
      await requirePath(sceneXmlPath, `Raw scene file not found`);
      const xml = await readFile(sceneXmlPath, "utf8");
      const metadata = loadSceneMetadata(xml, config.renderAssetUrlPrefix);

      const optimizedXmlPath = path.join(appResources, `${envId}_scene.xml`);
      await writeFile(optimizedXmlPath, optimizedXml(xml));

      if (envId.endsWith("_rough")) {
        const manifestPath = path.join(appResources, "render_manifests", `${config.id}_render_manifest.json`);
        await writeFile(
          manifestPath,
          `${JSON.stringify({ version: 1, parts: metadata.parts }, null, 2)}\n`,
        );
      }
      console.log(`${envId}: ${metadata.parts.length} visual parts`);
    }

    const sourceTotal = conversionStats.reduce((sum, item) => sum + item.sourceBytes, 0);
    const glbTotal = conversionStats.reduce((sum, item) => sum + item.glbBytes, 0);
    console.log(
      `${config.id}: converted ${conversionStats.length} STL files: ${(sourceTotal / 1024 / 1024).toFixed(1)}MB -> ${(glbTotal / 1024 / 1024).toFixed(1)}MB GLB`,
    );
  }

  console.log(`Updated iOS render assets in ${appResources}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
