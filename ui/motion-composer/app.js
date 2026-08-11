const state = {
  tool: "select",
  image: null,
  imageUrl: null,
  actorAssetUrl: null,
  actorImage: null,
  animationUrl: null,
  animationFile: null,
  scene: null,
  startPoint: null,
  endPoint: null,
  drawnPath: [],
  region: [],
  isDrawing: false,
  anchorStep: "start",
  preview: null,
  playing: false,
  selectedRequired: new Set(),
  selectedAvoid: new Set(),
  selectedBehind: new Set(),
};

const els = {
  canvas: document.getElementById("stageCanvas"),
  imageInput: document.getElementById("imageInput"),
  sceneInput: document.getElementById("sceneInput"),
  actorAssetInput: document.getElementById("actorAssetInput"),
  animationInput: document.getElementById("animationInput"),
  actorText: document.getElementById("actorText"),
  actionText: document.getElementById("actionText"),
  manifoldType: document.getElementById("manifoldType"),
  duration: document.getElementById("duration"),
  allowBending: document.getElementById("allowBending"),
  preserveGeometry: document.getElementById("preserveGeometry"),
  corridorRadius: document.getElementById("corridorRadius"),
  maxDeviation: document.getElementById("maxDeviation"),
  corridorValue: document.getElementById("corridorValue"),
  deviationValue: document.getElementById("deviationValue"),
  objectList: document.getElementById("objectList"),
  contractPreview: document.getElementById("contractPreview"),
  previewBtn: document.getElementById("previewBtn"),
  playBtn: document.getElementById("playBtn"),
  timeSlider: document.getElementById("timeSlider"),
  timeLabel: document.getElementById("timeLabel"),
  exportBtn: document.getElementById("exportBtn"),
  copyBtn: document.getElementById("copyBtn"),
  clearGeometryBtn: document.getElementById("clearGeometryBtn"),
  clearAnchorsBtn: document.getElementById("clearAnchorsBtn"),
  clearPathBtn: document.getElementById("clearPathBtn"),
  uploadStatus: document.getElementById("uploadStatus"),
  geometryStatus: document.getElementById("geometryStatus"),
  showObjects: document.getElementById("showObjects"),
  showRegions: document.getElementById("showRegions"),
  showCorridor: document.getElementById("showCorridor"),
  showOcclusion: document.getElementById("showOcclusion"),
  showActor: document.getElementById("showActor"),
  showAnchors: document.getElementById("showAnchors"),
};
const ctx = els.canvas.getContext("2d");

function init() {
  document.querySelectorAll(".tool[data-tool]").forEach((btn) => btn.addEventListener("click", () => setTool(btn.dataset.tool)));
  els.imageInput.addEventListener("change", loadImage);
  els.sceneInput.addEventListener("change", loadScene);
  els.actorAssetInput.addEventListener("change", loadActorAsset);
  els.animationInput.addEventListener("change", loadAnimation);
  els.canvas.addEventListener("pointerdown", onPointerDown);
  els.canvas.addEventListener("pointermove", onPointerMove);
  window.addEventListener("pointerup", onPointerUp);
  [els.actorText, els.actionText, els.manifoldType, els.duration, els.allowBending, els.preserveGeometry].forEach((el) => el.addEventListener("input", refresh));
  [els.corridorRadius, els.maxDeviation].forEach((el) => el.addEventListener("input", () => {
    els.corridorValue.textContent = `${els.corridorRadius.value}px`;
    els.deviationValue.textContent = `${els.maxDeviation.value}px`;
    refresh();
  }));
  [els.showObjects, els.showRegions, els.showCorridor, els.showOcclusion, els.showActor, els.showAnchors].forEach((el) => el.addEventListener("input", draw));
  els.clearGeometryBtn.addEventListener("click", () => { clearAllGeometry(); refresh(); });
  els.clearAnchorsBtn.addEventListener("click", () => { state.startPoint = null; state.endPoint = null; state.anchorStep = "start"; state.preview = null; refresh(); });
  els.clearPathBtn.addEventListener("click", () => { state.drawnPath = []; state.preview = null; refresh(); });
  els.previewBtn.addEventListener("click", () => { state.preview = buildGroundedContract(); refresh(); });
  els.playBtn.addEventListener("click", togglePlay);
  els.timeSlider.addEventListener("input", () => { draw(); updateTimeLabel(); });
  els.exportBtn.addEventListener("click", exportContract);
  els.copyBtn.addEventListener("click", () => navigator.clipboard?.writeText(els.contractPreview.textContent));
  window.__CITV_MOTION_COMPOSER__ = {
    buildLocalContract: buildGroundedContract,
    getScene: () => state.scene,
    setPreview: (grounded) => { state.preview = grounded; refresh(); },
    getState: () => state,
  };
  refresh();
}

function setTool(tool) {
  state.tool = tool;
  if (tool === "anchors" && state.startPoint && !state.endPoint) state.anchorStep = "end";
  document.querySelectorAll(".tool[data-tool]").forEach((btn) => btn.classList.toggle("active", btn.dataset.tool === tool));
  updateStatus();
}
function clearAllGeometry() { state.startPoint = null; state.endPoint = null; state.drawnPath = []; state.region = []; state.anchorStep = "start"; state.preview = null; }

function loadImage(event) { const file = event.target.files?.[0]; if (!file) return; const url = URL.createObjectURL(file); const img = new Image(); img.onload = () => { state.image = img; state.imageUrl = url; const scale = Math.min(1, 1280 / img.naturalWidth); els.canvas.width = Math.round(img.naturalWidth * scale); els.canvas.height = Math.round(img.naturalHeight * scale); refresh(); }; img.src = url; }
function loadActorAsset(event) { const file = event.target.files?.[0]; if (!file) return; state.actorAssetUrl = URL.createObjectURL(file); state.actorImage = null; if (file.type.startsWith("image/")) { const img = new Image(); img.onload = () => { state.actorImage = img; refresh(); }; img.src = state.actorAssetUrl; } refresh(); }
function loadAnimation(event) { const file = event.target.files?.[0]; if (!file) return; state.animationFile = { name: file.name, type: file.type || "unknown", size_bytes: file.size }; state.animationUrl = URL.createObjectURL(file); refresh(); }
async function loadScene(event) { const file = event.target.files?.[0]; if (!file) return; state.scene = JSON.parse(await file.text()); renderObjectList(); refresh(); }
function canvasPoint(event) { const rect = els.canvas.getBoundingClientRect(); return [((event.clientX - rect.left) / rect.width) * els.canvas.width, ((event.clientY - rect.top) / rect.height) * els.canvas.height]; }
function onPointerDown(event) { const pt = canvasPoint(event); if (state.tool === "point") { state.startPoint = pt; state.endPoint = null; state.drawnPath = []; } else if (state.tool === "anchors") { if (state.anchorStep === "start" || !state.startPoint) { state.startPoint = pt; state.anchorStep = "end"; } else { state.endPoint = pt; state.anchorStep = "start"; } } else if (state.tool === "draw") { state.isDrawing = true; state.drawnPath = [pt]; } else if (state.tool === "region") { state.isDrawing = true; state.region = [pt]; } else if (state.tool === "select") selectNearestObject(pt); state.preview = null; refresh(); }
function onPointerMove(event) { if (!state.isDrawing) return; const pt = canvasPoint(event); const target = state.tool === "region" ? state.region : state.drawnPath; const last = target[target.length - 1]; if (!last || distance(last, pt) > 4) target.push(pt); state.preview = null; refresh(); }
function onPointerUp() { state.isDrawing = false; }
function selectNearestObject(pt) { let best = null; for (const obj of getObjects()) { const c = obj.mask_centroid_2d || obj.centroid_2d; if (!Array.isArray(c)) continue; const d = distance(pt, c); if (!best || d < best.d) best = { obj, d }; } if (best && best.d < 60) { state.selectedRequired.has(best.obj.id) ? state.selectedRequired.delete(best.obj.id) : state.selectedRequired.add(best.obj.id); renderObjectList(); } }
function refresh() { updateStatus(); draw(); els.contractPreview.textContent = JSON.stringify(state.preview || buildGroundedContract(), null, 2); }
function updateStatus() { const pieces = []; pieces.push(state.startPoint ? `Start: ${fmtPoint(state.startPoint)}` : "Start: not set"); pieces.push(state.endPoint ? `End: ${fmtPoint(state.endPoint)}` : "End: not set"); pieces.push(state.drawnPath.length ? `Drawn path: ${state.drawnPath.length} points` : "Drawn path: none"); pieces.push(state.region.length ? `Region: ${state.region.length} points` : "Region: none"); els.geometryStatus.textContent = pieces.join(" • "); const uploads = []; if (state.actorAssetUrl) uploads.push("actor asset loaded"); if (state.animationFile) uploads.push(`animation: ${state.animationFile.name}`); els.uploadStatus.textContent = uploads.length ? uploads.join(" • ") : "No uploaded actor or animation yet."; }
function draw() { ctx.clearRect(0, 0, els.canvas.width, els.canvas.height); if (state.image) ctx.drawImage(state.image, 0, 0, els.canvas.width, els.canvas.height); else drawEmptyState(); if (els.showRegions.checked) drawRegions(); if (els.showObjects.checked) drawObjects(); if (els.showCorridor.checked) drawCorridor(); drawUserGeometry(); if (els.showAnchors.checked) drawAnchors(); if (els.showOcclusion.checked) drawOcclusionPreview(); if (els.showActor.checked) drawActorPreview(); }
function drawEmptyState() { ctx.fillStyle = "#10151e"; ctx.fillRect(0, 0, els.canvas.width, els.canvas.height); ctx.fillStyle = "#8e9bae"; ctx.font = "22px system-ui"; ctx.textAlign = "center"; ctx.fillText("Upload an image to begin", els.canvas.width / 2, els.canvas.height / 2); }
function drawObjects() { for (const obj of getObjects()) { const box = obj.bbox; if (Array.isArray(box) && box.length >= 4) { ctx.strokeStyle = state.selectedRequired.has(obj.id) ? "#8ad7ff" : state.selectedAvoid.has(obj.id) ? "#ff7d7d" : state.selectedBehind.has(obj.id) ? "#ffd37d" : "rgba(255,255,255,0.42)"; ctx.lineWidth = state.selectedRequired.has(obj.id) || state.selectedAvoid.has(obj.id) || state.selectedBehind.has(obj.id) ? 3 : 1; ctx.strokeRect(box[0], box[1], box[2] - box[0], box[3] - box[1]); } const c = obj.mask_centroid_2d || obj.centroid_2d; if (Array.isArray(c)) { ctx.fillStyle = "rgba(10,13,18,0.78)"; ctx.fillRect(c[0] - 4, c[1] - 18, 120, 18); ctx.fillStyle = "#eef3fb"; ctx.font = "12px system-ui"; ctx.textAlign = "left"; ctx.fillText(obj.canonical_name || obj.label || obj.id || "object", c[0], c[1] - 5); } } }
function drawRegions() { getRegions().forEach((region, i) => { const c = region.centroid_2d_px || region.centroid_2d; if (!Array.isArray(c)) return; ctx.beginPath(); ctx.arc(c[0], c[1], 22 + (i % 3) * 6, 0, Math.PI * 2); ctx.fillStyle = `rgba(${90 + i * 37 % 120}, ${120 + i * 43 % 100}, 255, 0.11)`; ctx.fill(); ctx.strokeStyle = "rgba(138,215,255,0.35)"; ctx.stroke(); }); }
function drawCorridor() { const pts = fusedPath(); if (pts.length < 2) return; ctx.save(); ctx.lineJoin = "round"; ctx.lineCap = "round"; ctx.lineWidth = Number(els.corridorRadius.value) * 2; ctx.strokeStyle = "rgba(138,215,255,0.12)"; drawPolyline(pts, false); ctx.restore(); }
function drawUserGeometry() { if (state.region.length >= 3) { ctx.save(); ctx.strokeStyle = "rgba(215,184,255,0.9)"; ctx.lineWidth = 3; drawPolyline(state.region, true); ctx.restore(); } if (state.drawnPath.length > 1) { ctx.save(); ctx.lineJoin = "round"; ctx.lineCap = "round"; ctx.strokeStyle = "#8ad7ff"; ctx.lineWidth = 4; drawPolyline(state.drawnPath, false); ctx.restore(); } }
function drawAnchors() { if (state.startPoint) drawAnchor(state.startPoint, "S", "#94f2b5"); if (state.endPoint) drawAnchor(state.endPoint, "E", "#ffb86b"); if (state.startPoint && state.endPoint && state.drawnPath.length < 2) { ctx.save(); ctx.strokeStyle = "rgba(138,215,255,0.85)"; ctx.setLineDash([8, 8]); ctx.lineWidth = 3; drawPolyline([state.startPoint, state.endPoint], false); ctx.restore(); } }
function drawAnchor(p, label, color) { ctx.save(); ctx.fillStyle = color; ctx.beginPath(); ctx.arc(p[0], p[1], 11, 0, Math.PI * 2); ctx.fill(); ctx.fillStyle = "#071019"; ctx.font = "bold 12px system-ui"; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(label, p[0], p[1]); ctx.restore(); }
function drawOcclusionPreview() { const c = state.preview || buildGroundedContract(); const pts = c.grounded_geometry?.adapted_polyline_2d || []; const layers = c.rendering?.render_layers || []; pts.forEach((p, i) => { if (layers[i] === "partially_occluded" || layers[i] === "behind_object") { ctx.beginPath(); ctx.arc(p[0], p[1], 9, 0, Math.PI * 2); ctx.fillStyle = layers[i] === "behind_object" ? "rgba(255,125,125,0.38)" : "rgba(255,211,125,0.32)"; ctx.fill(); } }); }
function drawActorPreview() { const c = state.preview || buildGroundedContract(); const pts = c.grounded_geometry?.adapted_polyline_2d || []; if (!pts.length) return; const t = Number(els.timeSlider.value) / 100; const p = pointAt(pts, t); const scaleHint = c.rendering?.depth_scale_hint || []; const scale = scaleHint[Math.floor(t * (scaleHint.length - 1))] || 1; ctx.save(); ctx.translate(p[0], p[1]); ctx.globalAlpha = 0.92; if (state.actorAssetUrl && state.actorImage?.complete) { const w = 68 * scale; ctx.drawImage(state.actorImage, -w / 2, -w / 2, w, w); } else { ctx.fillStyle = "rgba(215,184,255,0.86)"; ctx.beginPath(); ctx.ellipse(0, 0, 24 * scale, 10 * scale, 0, 0, Math.PI * 2); ctx.fill(); ctx.fillStyle = "#0c0f14"; ctx.font = `${Math.max(10, 12 * scale)}px system-ui`; ctx.textAlign = "center"; ctx.fillText((els.actorText.value || "actor").slice(0, 12), 0, 4); } ctx.restore(); }
function drawPolyline(pts, close) { ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]); for (let i = 1; i < pts.length; i += 1) ctx.lineTo(pts[i][0], pts[i][1]); if (close) ctx.closePath(); ctx.stroke(); }
function renderObjectList() { const objects = getObjects(); if (!objects.length) { els.objectList.className = "object-list empty"; els.objectList.textContent = "No objects found in scene JSON."; return; } els.objectList.className = "object-list"; els.objectList.innerHTML = ""; objects.slice(0, 80).forEach((obj) => { const card = document.createElement("div"); card.className = "object-card"; card.innerHTML = `<strong>${escapeHtml(obj.canonical_name || obj.label || obj.id || "object")}</strong><span>${escapeHtml(obj.id || "no-id")}</span>`; const row = document.createElement("div"); row.className = "constraint-row"; [["required", state.selectedRequired], ["avoid", state.selectedAvoid], ["behind", state.selectedBehind]].forEach(([label, set]) => { const btn = document.createElement("button"); btn.type = "button"; btn.textContent = label; btn.className = set.has(obj.id) ? "active" : ""; btn.addEventListener("click", () => { set.has(obj.id) ? set.delete(obj.id) : set.add(obj.id); renderObjectList(); refresh(); }); row.appendChild(btn); }); card.appendChild(row); els.objectList.appendChild(card); }); }
function buildGroundedContract() { const path = fusedPath(); const mode = state.region.length >= 3 && state.tool === "region" ? "region" : state.startPoint && state.endPoint && state.drawnPath.length > 1 ? "start_end_path" : state.startPoint && state.endPoint ? "start_end" : state.startPoint ? "point" : state.drawnPath.length > 1 ? "polyline" : "polyline"; const manifold = els.manifoldType.value === "auto" ? inferManifold(els.actionText.value, mode) : els.manifoldType.value; const contract = { contract_id: `ui_take_${Date.now()}`, actor: { actor_text: els.actorText.value.trim(), actor_source: state.actorAssetUrl ? "uploaded_asset" : "text", asset_ref: state.actorAssetUrl ? "browser_uploaded_actor_asset" : null, visual_style: state.actorAssetUrl ? "source_preserving" : "photorealistic" }, action_text: els.actionText.value.trim(), uploaded_animation_ref: state.animationFile ? "browser_uploaded_animation" : null, uploaded_animation: state.animationFile ? { ...state.animationFile, retargeting_policy: "preserve_timing_and_style_then_ground_to_scene" } : null, user_geometry: { mode, start_point: state.startPoint ? pointJson(state.startPoint) : null, end_point: state.endPoint ? pointJson(state.endPoint) : null, drawn_path_2d: state.drawnPath.map(pointJson), region_polygon_2d: state.region.map(pointJson), points: path.map(pointJson), source: geometrySource(mode), corridor_radius_px: Number(els.corridorRadius.value) }, manifold_type: manifold, duration_s: Number(els.duration.value), source: "user_authored", policy: { preserve_user_geometry: els.preserveGeometry.checked, allow_path_bending: els.allowBending.checked, max_path_deviation_px: Number(els.maxDeviation.value), required_object_ids: [...state.selectedRequired], avoid_object_ids: [...state.selectedAvoid], must_render_behind_object_ids: [...state.selectedBehind] } }; return adaptContractInBrowser(contract); }
function adaptContractInBrowser(contract) { const pts = resample(contract.user_geometry.points, 48); const objects = getObjects(); const occluderIds = pts.map((p) => objects.filter((o) => pointInBox(p, o.bbox)).map((o) => o.id).filter(Boolean)); const renderLayers = occluderIds.map((ids) => ids.some((id) => contract.policy.must_render_behind_object_ids.includes(id)) ? "behind_object" : ids.length ? "partially_occluded" : "in_front"); const visibility = renderLayers.map((l) => l === "behind_object" ? 0.18 : l === "partially_occluded" ? 0.56 : 1); const supportTrace = pts.map(nearestRegionLabel); const depthTrace = pts.map(estimateDepth); const warnings = []; if (!state.scene) warnings.push("Scene JSON not loaded; grounding uses canvas-only approximations."); if (!state.image) warnings.push("Image not loaded; preview canvas is schematic."); if (!contract.user_geometry.start_point && !contract.user_geometry.drawn_path_2d.length) warnings.push("No start point or drawn path yet; set anchors or draw a path."); if (contract.user_geometry.start_point && !contract.user_geometry.end_point && contract.manifold_type !== "effect_field") warnings.push("Start point is set but end point is missing for a path-like action."); if (!state.actorAssetUrl) warnings.push("No actor asset uploaded; renderer must resolve an open-vocabulary asset before final photoreal export."); if (!state.animationFile) warnings.push("No uploaded animation; renderer should generate motion from the action text or selected manifold."); const firstDepth = depthTrace.find(Boolean) || 1; return { schema_version: "citv.grounded_motion_contract.ui.v2", contract, grounded_geometry: { manifold_type: contract.manifold_type, start_point_2d: contract.user_geometry.start_point, end_point_2d: contract.user_geometry.end_point, user_drawn_path_2d: contract.user_geometry.drawn_path_2d, user_region_polygon_2d: contract.user_geometry.region_polygon_2d, user_polyline_2d: contract.user_geometry.points, adapted_polyline_2d: pts, corridor_radius_px: contract.user_geometry.corridor_radius_px, path_preservation_policy: contract.policy }, traces: { depth_trace_m: depthTrace, support_trace: supportTrace, visibility_profile: visibility, occluder_ids: occluderIds, semantic_trace: supportTrace.map((s) => s === "unknown" ? "semantic_context_unknown" : `${s}:action_context`) }, rendering: { render_layers: renderLayers, alpha_profile: visibility, depth_scale_hint: depthTrace.map((z) => z ? round(firstDepth / z, 3) : null), asset_policy: { actor_source: contract.actor.actor_source, visual_style: contract.actor.visual_style, asset_ref: contract.actor.asset_ref, no_hard_coded_actor_fallback: true }, animation_policy: { uploaded_animation_ref: contract.uploaded_animation_ref, preserve_uploaded_timing: Boolean(contract.uploaded_animation_ref), retarget_to_scene: true } }, report: { status: warnings.length ? "accepted_with_warnings" : "accepted", preserved: ["raw_start_point", "raw_end_point", "raw_drawn_path", "actor_text_or_asset", "action_text", "uploaded_animation_reference", "duration_s"], adapted: ["fused anchors and drawn path", "resampled path for preview", "estimated support trace", "estimated occlusion/render layers"], warnings, scores: { user_geometry_preservation: 1, min_visibility: round(Math.min(...visibility), 3), support_known_ratio: round(supportTrace.filter((s) => s !== "unknown").length / Math.max(1, supportTrace.length), 3) } }, nearest_scene_entities: { objects: nearestObjects(pts[0] || [0, 0]).slice(0, 8) }, alternatives: [] }; }
function fusedPath() { if (state.region.length >= 3 && state.tool === "region") return state.region; if (state.startPoint && state.endPoint && state.drawnPath.length > 1) return [state.startPoint, ...state.drawnPath, state.endPoint]; if (state.startPoint && state.endPoint) return [state.startPoint, state.endPoint]; if (state.startPoint && state.drawnPath.length > 1) return [state.startPoint, ...state.drawnPath]; if (state.drawnPath.length > 1) return state.drawnPath; if (state.startPoint) return [state.startPoint]; return []; }
function geometrySource(mode) { return mode === "start_end_path" ? "user_start_end_plus_drawn_path" : mode === "start_end" ? "user_start_end" : mode === "point" ? "user_tap" : mode === "region" ? "user_region" : "user_drawn"; }
function getObjects() { return Array.isArray(state.scene?.objects) ? state.scene.objects : []; }
function getRegions() { const r = state.scene?.regions; return Array.isArray(r) ? r : Array.isArray(r?.regions) ? r.regions : []; }
function distance(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1]); }
function round(v, n = 2) { const m = 10 ** n; return Math.round(v * m) / m; }
function pointJson(p) { return [round(p[0]), round(p[1])]; }
function fmtPoint(p) { return `[${Math.round(p[0])}, ${Math.round(p[1])}]`; }
function escapeHtml(s) { return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
function pointInBox(p, box) { return Array.isArray(box) && box.length >= 4 && p[0] >= box[0] && p[0] <= box[2] && p[1] >= box[1] && p[1] <= box[3]; }
function nearestObjects(p) { return getObjects().map((o) => ({ id: o.id, label: o.canonical_name || o.label, distance_px: round(distance(p, o.mask_centroid_2d || o.centroid_2d || [9999, 9999])) })).sort((a, b) => a.distance_px - b.distance_px); }
function nearestRegionLabel(p) { const best = getRegions().map((r) => ({ label: r.semantic_label || r.label || r.id || "region", d: distance(p, r.centroid_2d_px || r.centroid_2d || [9999, 9999]) })).sort((a, b) => a.d - b.d)[0]; return best && best.d < 180 ? best.label : "unknown"; }
function estimateDepth(p) { const h = els.canvas.height || 1; return round(1.2 + (p[1] / h) * 3.8, 3); }
function inferManifold(text, mode) { text = (text || "").toLowerCase(); if (/peek|hide|behind/.test(text)) return "occlusion_pulse"; if (/fly|glide|hover/.test(text)) return "volume_path"; if (/swim|float|drift|water/.test(text)) return "blob_path"; if (/hold|touch|sit|bump|push|place/.test(text)) return "contact_patch"; if (/enter|exit|disappear|portal/.test(text)) return "portal_path"; if (/shimmer|glow|ripple|logo|wobble/.test(text)) return "effect_field"; if (/circle|orbit|around/.test(text)) return "contour_path"; if (mode === "point") return "effect_field"; if (mode === "region") return "blob_path"; return "ribbon_path"; }
function resample(points, count) { if (points.length <= 1) return points; const d = [0]; for (let i = 1; i < points.length; i++) d.push(d[i - 1] + distance(points[i - 1], points[i])); const total = d[d.length - 1]; if (!total) return Array(count).fill(points[0]); const out = []; let seg = 0; for (let i = 0; i < count; i++) { const target = total * i / (count - 1); while (seg < d.length - 2 && d[seg + 1] < target) seg++; const a = points[seg], b = points[seg + 1]; const t = (target - d[seg]) / Math.max(1e-6, d[seg + 1] - d[seg]); out.push([round(a[0] + t * (b[0] - a[0])), round(a[1] + t * (b[1] - a[1]))]); } return out; }
function pointAt(points, t) { if (!points.length) return [els.canvas.width / 2, els.canvas.height / 2]; return points[Math.max(0, Math.min(points.length - 1, Math.round(t * (points.length - 1))))]; }
function updateTimeLabel() { els.timeLabel.textContent = `${round((Number(els.timeSlider.value) / 100) * Number(els.duration.value), 1)}s`; }
function togglePlay() { state.playing = !state.playing; els.playBtn.textContent = state.playing ? "Pause" : "Play"; if (state.playing) requestAnimationFrame(playTick); }
function playTick() { if (!state.playing) return; els.timeSlider.value = (Number(els.timeSlider.value) + 1) % 101; updateTimeLabel(); draw(); requestAnimationFrame(playTick); }
function exportContract() { const blob = new Blob([els.contractPreview.textContent], { type: "application/json" }); const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "grounded_motion_contract.json"; a.click(); URL.revokeObjectURL(a.href); }

init();
