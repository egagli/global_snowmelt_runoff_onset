export const GA_MEASUREMENT_ID = 'G-M59931E34K'

type GtagParams = Record<string, string | number | boolean>

declare global {
  interface Window {
    gtag?: (...args: unknown[]) => void
  }
}

export const track = (name: string, params: GtagParams = {}) => {
  if (typeof window === 'undefined' || typeof window.gtag !== 'function') return
  try {
    window.gtag('event', name, params)
  } catch {}
}

// only send the value the slider settles on
const timers: Record<string, ReturnType<typeof setTimeout>> = {}
export const trackDebounced = (name: string, params: GtagParams = {}, waitMs = 800) => {
  if (typeof window === 'undefined') return
  clearTimeout(timers[name])
  timers[name] = setTimeout(() => track(name, params), waitMs)
}
