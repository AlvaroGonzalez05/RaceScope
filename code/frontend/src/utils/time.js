export function formatRaceDuration(totalSeconds) {
  const safeSeconds = Math.max(0, Math.round(Number(totalSeconds) || 0));
  const hours = Math.floor(safeSeconds / 3600);
  const minutes = Math.floor((safeSeconds % 3600) / 60);
  const seconds = safeSeconds % 60;

  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  }
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

export function formatLapTime(lapSeconds) {
  const value = Number(lapSeconds);
  if (!Number.isFinite(value)) return "-";
  return `${value.toFixed(3)}s`;
}

/**
 * Format a time delta in seconds relative to the best strategy.
 * Returns "BEST" for zero/negative values, "+Xs" or "+Xm Xs" otherwise.
 */
export function formatDelta(seconds) {
  if (!Number.isFinite(seconds) || seconds <= 0) return "BEST";
  const m = Math.floor(seconds / 60);
  const s = (seconds % 60).toFixed(1);
  return m > 0 ? `+${m}m ${s}s` : `+${s}s`;
}
