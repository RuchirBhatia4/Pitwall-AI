const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers ?? {}),
    },
    ...options,
  });

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail ?? `Request failed with status ${response.status}`);
  }
  return payload;
}

export function getHealth() {
  return request("/health");
}

export function getDashboardSummary() {
  return request("/api/dashboard-summary");
}

export function getStorageStatus() {
  return request("/api/storage/status");
}

export function getCalibratedDegradation() {
  return request("/api/calibrated-degradation");
}

export function getStrategyRecommendations() {
  return request("/api/strategy-recommendations");
}

export function runCalibrationPipeline() {
  return request("/api/pipeline/calibrated-predictions", {
    method: "POST",
    body: JSON.stringify({}),
  });
}

export function runStrategyPipeline() {
  return request("/api/pipeline/strategy-recommendations", {
    method: "POST",
    body: JSON.stringify({}),
  });
}
