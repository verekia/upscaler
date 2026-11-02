export const formatBytes = (bytes: number) => {
  if (bytes < 1e3) return `${bytes} B`
  if (bytes < 1e6) return `${(bytes / 1e3).toFixed(0)} KB`
  return `${(bytes / 1e6).toFixed(bytes < 1e7 ? 1 : 0)} MB`
}

export const formatDuration = (ms: number) => {
  if (ms < 1000) return `${Math.round(ms)} ms`
  const seconds = ms / 1000
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)} s`
  return `${Math.floor(seconds / 60)} min ${Math.round(seconds % 60)} s`
}
