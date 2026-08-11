// Backend bridge for the CITV Motion Composer UI.
//
// This script is loaded after app.js. It keeps the existing browser-only preview
// as a fallback while wiring Generate Preview / Export to the local Python API:
//
//   POST /api/motion/ground
//
// The API returns the same GroundedMotionContract shape consumed by the canvas.

(function () {
  const apiStatus = document.getElementById("apiStatus");
  const useBackend = document.getElementById("useBackend");
  const previewBtn = document.getElementById("previewBtn");
  const exportBtn = document.getElementById("exportBtn");

  if (!previewBtn || !exportBtn) return;

  async function groundWithBackend(localContract) {
    const motionContract = localContract.contract || localContract;
    const payload = {
      motion_contract: motionContract,
      scene_graph: window.__CITV_MOTION_COMPOSER__?.getScene?.() || null,
      sample_count: 72,
    };
    const response = await fetch("/api/motion/ground", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || `Backend returned ${response.status}`);
    }
    data.schema_version = data.schema_version || "citv.grounded_motion_contract.backend.v1";
    data.report = data.report || {};
    data.report.adapted = data.report.adapted || [];
    if (!data.report.adapted.includes("grounded by local Python API")) {
      data.report.adapted.push("grounded by local Python API");
    }
    return data;
  }

  async function generatePreviewViaBestAvailable() {
    const local = window.__CITV_MOTION_COMPOSER__?.buildLocalContract?.();
    if (!local) return;
    if (!useBackend?.checked) {
      setStatus("Using browser-only grounding fallback.", "warn");
      window.__CITV_MOTION_COMPOSER__?.setPreview?.(local);
      return;
    }
    setStatus("Calling Python grounding backend...", "warn");
    try {
      const grounded = await groundWithBackend(local);
      window.__CITV_MOTION_COMPOSER__?.setPreview?.(grounded);
      setStatus("Backend grounding succeeded.", "ok");
    } catch (error) {
      local.report = local.report || {};
      local.report.warnings = local.report.warnings || [];
      local.report.warnings.push(`Backend unavailable; using browser fallback: ${error.message}`);
      window.__CITV_MOTION_COMPOSER__?.setPreview?.(local);
      setStatus(`Backend unavailable; using browser fallback. ${error.message}`, "warn");
    }
  }

  function exportCurrentContract() {
    const text = document.getElementById("contractPreview")?.textContent || "{}";
    const blob = new Blob([text], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "grounded_motion_contract.json";
    a.click();
    URL.revokeObjectURL(a.href);
  }

  function setStatus(message, tone) {
    if (!apiStatus) return;
    apiStatus.textContent = message;
    apiStatus.dataset.tone = tone || "neutral";
  }

  previewBtn.addEventListener("click", (event) => {
    event.stopImmediatePropagation();
    event.preventDefault();
    generatePreviewViaBestAvailable();
  }, true);

  exportBtn.addEventListener("click", (event) => {
    event.stopImmediatePropagation();
    event.preventDefault();
    exportCurrentContract();
  }, true);

  setStatus("Ready. Start the local Python server for backend grounding.", "neutral");
})();
