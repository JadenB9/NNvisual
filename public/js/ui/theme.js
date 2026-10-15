// Canvas colors come from the same CSS tokens the page is styled with (see
// base.css), read at draw time, so a theme switch repaints every canvas in the
// new palette instead of leaving dark-only pixels on a light page.

const NAMES = ['bg', 'surface', 'surface-2', 'border', 'border-strong', 'text', 'text-bright',
    'muted', 'faint', 'ok', 'err', 'link-blue', 'ink', 'gold'];

let cache = null;
let cacheFor = null;

export function currentTheme() {
    return document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
}

// { bg: '#0e1011', borderStrong: '#333a40', ... } for the theme showing now
export function tokens() {
    const theme = currentTheme();
    if (cache && cacheFor === theme) return cache;
    const style = getComputedStyle(document.documentElement);
    cache = {};
    for (const name of NAMES) {
        const key = name.replace(/-([a-z0-9])/g, (_, c) => c.toUpperCase());
        cache[key] = style.getPropertyValue(`--${name}`).trim();
    }
    cacheFor = theme;
    return cache;
}

// '#0e1011' -> [14, 16, 17]; also accepts the short '#abc' form
export function hexToRgb(hex) {
    let h = hex.trim().replace('#', '');
    if (h.length === 3) h = h.replace(/./g, (c) => c + c);
    const n = parseInt(h, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export function withAlpha(hex, a) {
    const [r, g, b] = hexToRgb(hex);
    return `rgba(${r},${g},${b},${a})`;
}

// one color per digit class for the latent plane, picked to read on each
// theme's background
const CLASS_DARK = ['#6a9955', '#c25b4e', '#86c3d6', '#c2a24e', '#9d7bd8',
    '#4ec2a4', '#d88ab0', '#8fb54e', '#d8873b', '#7b95d8'];
const CLASS_LIGHT = ['#40773d', '#a4402f', '#2f7d94', '#8a6d1d', '#6b4fb3',
    '#1f8a6e', '#b2497f', '#5b7f1f', '#b45c17', '#3f5bb8'];

export function classColors() {
    return currentTheme() === 'light' ? CLASS_LIGHT : CLASS_DARK;
}

export function onThemeChange(fn) {
    window.addEventListener('themechange', fn);
}
