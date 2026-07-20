const state = {
  image: null,
  actorImage: null,
  scene: null,
  pathsBundle: null,
  animationPlan: null,
  objects: [],
  paths: [],
  mode: "source",
  source: null,
  target: null,
  selectedPathId: "",
  activePath: null,
  manualPath: null,
  animating: false,
  animationStart: 0,
  canvasScale: 1,
  canvasOffset: { x: 0, y: 0 },
};

const els = {
  canvas: document.getElementById("sceneCanvas"),
  statusText: document.getElementById("statusText"),
  sceneTitle: document.getElementById("sceneTitle"),
  pathTitle: document.getElementById("pathTitle"),
  sourceModeBtn: document.getElementById("sourceModeBtn"),
  targetModeBtn: document.getElementById("targetModeBtn"),
  sourceReadout: document.getElementById("sourceReadout"),
  targetReadout: document.getElementById("targetReadout"),
  actionPreset: document.getElementById("actionPreset"),
  customPrompt: document.getElementById("customPrompt"),
  pathSelect: document.getElementById("pathSelect"),
  confidenceReadout: document.getElementById("confidenceReadout"),
  manifoldReadout: document.getElementById("manifoldReadout"),
  contractOutput: document.getElementById("contractOutput"),
};

const ctx = els.canvas.getContext("2d");

function labelForObject(obj) {
  return obj.canonical_name || obj.name || obj.label || obj.id || "object";
}

function shortId(value) {
  if (!value) return "None";
  return String(value).replace(/^grounded_sam2_/, "").replace(/_GroundedSAM2$/, "");
}

function setMode(mode) {
  state.mode = mode;
  els.sourceModeBtn.classList.toggle("selected", mode === "source");
  els.targetModeBtn.classList.toggle("selected", mode === "target");
}

function setStatus(text) {
  els.statusText.textContent = text;
}

function objectCenter(obj) {
  const c = obj.mask_centroid_2d;
  if (Array.isArray(c) && c.length >= 2) return [Number(c[0]), Number(c[1])];
  const b = obj.bbox || [];
  if (b.length >= 4) return [(Number(b[0]) + Number(b[2])) / 2, (Number(b[1]) + Number(b[3])) / 2];
  return [0, 0];
}

function selectionFromObject(obj) {
  const [x, y] = objectCenter(obj);
  return {
    kind: "object",
    id: obj.id || "",
    label: labelForObject(obj),
    xy: [x, y],
  };
}

function selectionFromPoint(x, y) {
  return {
    kind: "point",
    id: "",
    label: `Point ${Math.round(x)}, ${Math.round(y)}`,
    xy: [x, y],
  };
}

function updateReadouts() {
  els.sourceReadout.textContent = state.source ? `${state.source.label} ${shortId(state.source.id)}` : "None";
  els.targetReadout.textContent = state.target ? `${state.target.label} ${shortId(state.target.id)}` : "None";
}

function loadImageFromUrl(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = url;
  });
}

function loadImageFromFile(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);
      resolve(img);
    };
    img.onerror = reject;
    img.src = url;
  });
}

function readJsonFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      try {
        resolve(JSON.parse(String(reader.result || "{}")));
      } catch (err) {
        reject(err);
      }
    };
    reader.onerror = reject;
    reader.readAsText(file);
  });
}

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load ${url}`);
  return res.json();
}

function applyScene(scene) {
  state.scene = scene;
  state.objects = Array.isArray(scene.objects) ? scene.objects.filter((obj) => Array.isArray(obj.bbox)) : [];
  els.sceneTitle.textContent = scene?.metadata?.image_path || scene?.metadata?.track || "CITV scene";
  setStatus(`${state.objects.length} objects loaded`);
  draw();
}

function applyPaths(bundle) {
  state.pathsBundle = bundle;
  state.paths = Array.isArray(bundle.paths) ? bundle.paths.slice() : [];
  state.paths.sort((a, b) => scoreOf(b) - scoreOf(a));
  refreshPathOptions();
  draw();
}

function scoreOf(path) {
  return Number(path?.scores?.overall_confidence ?? path?.scores?.hybrid_overall ?? 0);
}

function refreshPathOptions() {
  els.pathSelect.innerHTML = "";
  const matching = matchingPaths();
  const hasSelectionPair = Boolean(state.source && state.target);
  const list = matching.length ? matching : (hasSelectionPair ? [] : state.paths.slice(0, 30));
  if (!list.length) {
    const opt = document.createElement("option");
    opt.value = "__manual__";
    opt.textContent = "Manual path";
    els.pathSelect.appendChild(opt);
    state.selectedPathId = "__manual__";
    state.activePath = null;
    updatePathReadout();
    return;
  }
  for (const path of list) {
    const opt = document.createElement("option");
    opt.value = path.path_id;
    const source = path.source_entity?.display_label || path.source_entity?.id || "source";
    const target = path.target_entity?.display_label || path.target_entity?.id || "target";
    opt.textContent = `${path.path_num || ""} ${source} -> ${target} (${scoreOf(path).toFixed(2)})`;
    els.pathSelect.appendChild(opt);
  }
  state.selectedPathId = els.pathSelect.value;
  state.activePath = state.paths.find((p) => p.path_id === state.selectedPathId) || null;
  updatePathReadout();
}

function matchingPaths() {
  if (!state.source || !state.target) return [];
  const sid = state.source.id;
  const tid = state.target.id;
  if (!sid || !tid) return [];
  return state.paths.filter((path) => {
    const ps = path.source_entity?.id || "";
    const pt = path.target_entity?.id || "";
    return (ps === sid && pt === tid) || (ps === tid && pt === sid);
  });
}

function activePolyline() {
  if (state.activePath && Array.isArray(state.activePath.polyline_geodesic_2d)) {
    return state.activePath.polyline_geodesic_2d;
  }
  if (state.activePath && Array.isArray(state.activePath.polyline_2d)) {
    return state.activePath.polyline_2d;
  }
  if (state.manualPath) return state.manualPath;
  if (state.source && state.target) {
    return makeCurve(state.source.xy, state.target.xy);
  }
  return [];
}

function makeCurve(a, b) {
  const ax = Number(a[0]);
  const ay = Number(a[1]);
  const bx = Number(b[0]);
  const by = Number(b[1]);
  const mx = (ax + bx) / 2;
  const my = (ay + by) / 2;
  const lift = Math.max(40, Math.hypot(bx - ax, by - ay) * 0.18);
  const cx = mx;
  const cy = my - lift;
  const pts = [];
  for (let i = 0; i <= 48; i += 1) {
    const t = i / 48;
    const x = (1 - t) * (1 - t) * ax + 2 * (1 - t) * t * cx + t * t * bx;
    const y = (1 - t) * (1 - t) * ay + 2 * (1 - t) * t * cy + t * t * by;
    pts.push([x, y]);
  }
  return pts;
}

function inferManifold() {
  const preset = els.actionPreset.value;
  const text = `${preset} ${els.customPrompt.value || ""}`.toLowerCase();
  if (/fly|float|air|smoke|hover/.test(text)) return "volume_path";
  if (/orbit|circle|around|contour|edge/.test(text)) return "contour_path";
  if (/disappear|emerge|enter|exit|portal|window|door|mirror/.test(text)) return "portal_path";
  if (/hide|peek|behind|occlude|reveal/.test(text)) return "occlusion_pulse";
  if (/touch|hold|grab|push|pull|lean|sit|place/.test(text)) return "contact_patch";
  if (/ripple|shimmer|glow|dissolve|magic|light|effect|reflect/.test(text)) return "effect_field";
  if (/run|walk|crawl|drive/.test(text)) return "ribbon_path";
  return "centerline_path";
}

function updatePathReadout() {
  const path = state.activePath;
  const conf = path ? scoreOf(path).toFixed(2) : "-";
  els.confidenceReadout.textContent = conf;
  els.manifoldReadout.textContent = inferManifold();
  els.pathTitle.textContent = path ? path.path_id : "Manual path";
}

function resizeCanvas() {
  const img = state.image;
  const wrap = document.querySelector(".canvas-wrap");
  if (!img || !wrap) {
    els.canvas.width = Math.max(600, els.canvas.clientWidth || 600);
    els.canvas.height = Math.max(420, els.canvas.clientHeight || 420);
    return;
  }
  const maxW = Math.max(320, wrap.clientWidth - 32);
  const maxH = Math.max(320, wrap.clientHeight - 32);
  const scale = Math.min(maxW / img.naturalWidth, maxH / img.naturalHeight, 1);
  state.canvasScale = scale;
  els.canvas.width = Math.round(img.naturalWidth * scale);
  els.canvas.height = Math.round(img.naturalHeight * scale);
}

function toCanvasPoint(x, y) {
  return [x * state.canvasScale, y * state.canvasScale];
}

function fromCanvasPoint(x, y) {
  return [x / state.canvasScale, y / state.canvasScale];
}

function draw() {
  resizeCanvas();
  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
  if (!state.image) {
    ctx.fillStyle = "#101010";
    ctx.fillRect(0, 0, els.canvas.width, els.canvas.height);
    ctx.fillStyle = "#f7f1e8";
    ctx.font = "16px system-ui";
    ctx.fillText("Load a scene image", 24, 34);
    return;
  }

  ctx.drawImage(state.image, 0, 0, els.canvas.width, els.canvas.height);
  drawObjects();
  drawPath(activePolyline());
  drawSelection(state.source, "#0f766e");
  drawSelection(state.target, "#b42318");
}

function drawObjects() {
  ctx.save();
  ctx.lineWidth = 1.5;
  ctx.font = "12px system-ui";
  for (const obj of state.objects) {
    const [x1, y1] = toCanvasPoint(Number(obj.bbox[0]), Number(obj.bbox[1]));
    const [x2, y2] = toCanvasPoint(Number(obj.bbox[2]), Number(obj.bbox[3]));
    ctx.strokeStyle = "rgba(255, 188, 66, 0.95)";
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const label = labelForObject(obj).slice(0, 20);
    const ty = Math.max(14, y1 - 4);
    ctx.fillStyle = "rgba(18, 18, 18, 0.82)";
    ctx.fillRect(x1, ty - 13, Math.min(180, ctx.measureText(label).width + 12), 16);
    ctx.fillStyle = "#fff7e8";
    ctx.fillText(label, x1 + 5, ty);
  }
  ctx.restore();
}

function drawPath(points) {
  if (!Array.isArray(points) || points.length < 2) return;
  ctx.save();
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.strokeStyle = "rgba(5, 150, 105, 0.92)";
  ctx.lineWidth = Math.max(2, 5 * state.canvasScale);
  ctx.beginPath();
  points.forEach((p, i) => {
    const [x, y] = toCanvasPoint(Number(p[0]), Number(p[1]));
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.strokeStyle = "rgba(255, 255, 255, 0.84)";
  ctx.lineWidth = Math.max(1, 1.4 * state.canvasScale);
  ctx.stroke();
  ctx.restore();
}

function drawSelection(sel, color) {
  if (!sel) return;
  const [x, y] = toCanvasPoint(sel.xy[0], sel.xy[1]);
  ctx.save();
  ctx.fillStyle = color;
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(x, y, 7, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function drawActorAt(point, heading = 0) {
  const [x, y] = toCanvasPoint(Number(point[0]), Number(point[1]));
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(heading);
  if (state.actorImage) {
    const size = Math.max(28, Math.min(84, 56 * state.canvasScale));
    ctx.drawImage(state.actorImage, -size / 2, -size / 2, size, size);
  } else {
    ctx.fillStyle = "#0f766e";
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(0, 0, 10, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.moveTo(13, 0);
    ctx.lineTo(2, -5);
    ctx.lineTo(2, 5);
    ctx.closePath();
    ctx.fill();
  }
  ctx.restore();
}

function pathLength(points) {
  let total = 0;
  for (let i = 1; i < points.length; i += 1) {
    total += Math.hypot(Number(points[i][0]) - Number(points[i - 1][0]), Number(points[i][1]) - Number(points[i - 1][1]));
  }
  return total;
}

function samplePolyline(points, t) {
  if (!points.length) return [0, 0, 0];
  if (points.length === 1) return [points[0][0], points[0][1], 0];
  const total = pathLength(points);
  if (total <= 0.0001) return [points[0][0], points[0][1], 0];
  const target = total * Math.max(0, Math.min(1, t));
  let walked = 0;
  for (let i = 1; i < points.length; i += 1) {
    const a = points[i - 1];
    const b = points[i];
    const seg = Math.hypot(Number(b[0]) - Number(a[0]), Number(b[1]) - Number(a[1]));
    if (walked + seg >= target) {
      const local = (target - walked) / Math.max(0.0001, seg);
      const x = Number(a[0]) + (Number(b[0]) - Number(a[0])) * local;
      const y = Number(a[1]) + (Number(b[1]) - Number(a[1])) * local;
      const heading = Math.atan2(Number(b[1]) - Number(a[1]), Number(b[0]) - Number(a[0]));
      return [x, y, heading];
    }
    walked += seg;
  }
  const last = points[points.length - 1];
  return [Number(last[0]), Number(last[1]), 0];
}

function durationForPath(points) {
  const path = state.activePath;
  const anim = state.animationPlan?.paths?.find((p) => p.path_id === path?.path_id);
  if (anim?.duration_s) return Number(anim.duration_s);
  const preset = els.actionPreset.value;
  const pxPerSec = preset === "run" ? 150 : preset === "fly" ? 120 : 80;
  return Math.max(0.8, pathLength(points) / pxPerSec);
}

function animateFrame(now) {
  if (!state.animating) return;
  const points = activePolyline();
  const duration = durationForPath(points) * 1000;
  const elapsed = now - state.animationStart;
  const t = (elapsed % duration) / duration;
  draw();
  const [x, y, heading] = samplePolyline(points, t);
  drawActorAt([x, y], heading);
  requestAnimationFrame(animateFrame);
}

function startPreview() {
  const points = activePolyline();
  if (points.length < 2) return;
  state.animating = true;
  state.animationStart = performance.now();
  requestAnimationFrame(animateFrame);
}

function stopPreview() {
  state.animating = false;
  draw();
}

function hitObject(x, y) {
  for (let i = state.objects.length - 1; i >= 0; i -= 1) {
    const obj = state.objects[i];
    const b = obj.bbox || [];
    if (b.length < 4) continue;
    if (x >= Number(b[0]) && x <= Number(b[2]) && y >= Number(b[1]) && y <= Number(b[3])) {
      return obj;
    }
  }
  return null;
}

function handleCanvasClick(event) {
  const rect = els.canvas.getBoundingClientRect();
  const cx = event.clientX - rect.left;
  const cy = event.clientY - rect.top;
  const [x, y] = fromCanvasPoint(cx, cy);
  const obj = hitObject(x, y);
  const sel = obj ? selectionFromObject(obj) : selectionFromPoint(x, y);
  if (state.mode === "source") {
    state.source = sel;
    setMode("target");
  } else {
    state.target = sel;
    setMode("source");
  }
  state.manualPath = state.source && state.target ? makeCurve(state.source.xy, state.target.xy) : null;
  updateReadouts();
  refreshPathOptions();
  updateContract();
  draw();
}

function contractPayload() {
  const points = activePolyline();
  const path = state.activePath;
  const manifold = inferManifold();
  return {
    schema: "citv_user_action_contract_v0",
    created_by: "ux_demo",
    source: state.source,
    target: state.target,
    action: {
      preset: els.actionPreset.value,
      prompt: els.customPrompt.value.trim(),
      manifold_type: manifold,
    },
    trajectory: {
      polyline_2d: points.map((p) => [Number(p[0]), Number(p[1])]),
      duration_s: Number(durationForPath(points).toFixed(3)),
      fps: Number(state.animationPlan?.fps || 24),
      interpolation: "polyline_constant_speed",
    },
    render: {
      actor_image_loaded: Boolean(state.actorImage),
      scale_with_depth: true,
      visibility_trace_required: ["portal_path", "occlusion_pulse", "contact_patch"].includes(manifold),
    },
    evidence: {
      path_id: path?.path_id || "",
      path_type: path?.path_type || "manual_curve",
      confidence: path ? scoreOf(path) : 0.35,
      source_file: path ? "path_hypotheses.json" : "manual",
      semantic_reasons: path?.semantic_reasons || [],
    },
  };
}

function updateContract() {
  updatePathReadout();
  const payload = contractPayload();
  els.contractOutput.value = JSON.stringify(payload, null, 2);
}

function downloadContract() {
  updateContract();
  const blob = new Blob([els.contractOutput.value], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "citv_action_contract.json";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

async function loadSample() {
  stopPreview();
  setStatus("Loading bundled sample");
  const [image, scene, paths, animationPlan] = await Promise.all([
    loadImageFromUrl("../images/IMG-6392.png"),
    fetchJson("../mps/scene_graph/grounded_sam2/IMG-6392_scene.json"),
    fetchJson("../mps/scene_graph/grounded_sam2/IMG-6392_paths/path_hypotheses.json"),
    fetchJson("../mps/scene_graph/grounded_sam2/IMG-6392_paths/animation_plan.json").catch(() => null),
  ]);
  state.image = image;
  state.animationPlan = animationPlan;
  applyScene(scene);
  applyPaths(paths);
  const firstObjPath = state.paths.find((p) => p.path_level === "object");
  if (firstObjPath) {
    const s = state.objects.find((obj) => obj.id === firstObjPath.source_entity?.id);
    const t = state.objects.find((obj) => obj.id === firstObjPath.target_entity?.id);
    if (s) state.source = selectionFromObject(s);
    if (t) state.target = selectionFromObject(t);
    state.selectedPathId = firstObjPath.path_id;
    els.pathSelect.value = firstObjPath.path_id;
    state.activePath = firstObjPath;
  }
  updateReadouts();
  updateContract();
  setStatus(`${state.objects.length} objects, ${state.paths.length} paths`);
  draw();
}

document.getElementById("loadSampleBtn").addEventListener("click", () => {
  loadSample().catch((err) => setStatus(err.message));
});

document.getElementById("sceneImageInput").addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  state.image = await loadImageFromFile(file);
  els.sceneTitle.textContent = file.name;
  setStatus("Scene image loaded");
  draw();
});

document.getElementById("actorImageInput").addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  state.actorImage = await loadImageFromFile(file);
  setStatus("Actor image loaded");
  draw();
});

document.getElementById("sceneJsonInput").addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  applyScene(await readJsonFile(file));
  updateContract();
});

document.getElementById("pathsJsonInput").addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  applyPaths(await readJsonFile(file));
  updateContract();
});

els.sourceModeBtn.addEventListener("click", () => setMode("source"));
els.targetModeBtn.addEventListener("click", () => setMode("target"));
els.canvas.addEventListener("click", handleCanvasClick);
els.pathSelect.addEventListener("change", () => {
  state.selectedPathId = els.pathSelect.value;
  state.activePath = state.selectedPathId === "__manual__"
    ? null
    : state.paths.find((p) => p.path_id === state.selectedPathId) || null;
  updateContract();
  draw();
});
els.actionPreset.addEventListener("change", updateContract);
els.customPrompt.addEventListener("input", updateContract);
document.getElementById("previewBtn").addEventListener("click", startPreview);
document.getElementById("stopBtn").addEventListener("click", stopPreview);
document.getElementById("buildContractBtn").addEventListener("click", updateContract);
document.getElementById("downloadContractBtn").addEventListener("click", downloadContract);
window.addEventListener("resize", draw);

setMode("source");
draw();
