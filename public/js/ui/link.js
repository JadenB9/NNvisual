// Shareable page state in the URL hash, the way diffusion.html's copy link
// works: a few validated keys, nothing the page would not have accepted from
// its own controls.

// spec: { key: rule } where rule is one of
//   { int: [lo, hi] }        an integer in range
//   { num: [lo, hi] }        a finite number in range
//   { oneOf: [...] }         one of the listed strings
//   { ints: [lo, hi, max] }  a comma list of up to max integers in range
//   { nums: [lo, hi, max] }  a comma list of up to max numbers in range
//   { text: max }            any string up to max characters
// Keys that are missing or fail their rule are left out of the result.
export function parseLink(hash, spec) {
    const q = new URLSearchParams(hash.replace(/^#/, ''));
    const out = {};
    for (const [key, rule] of Object.entries(spec)) {
        const raw = q.get(key);
        if (raw == null) continue;
        if (rule.int) {
            const v = parseInt(raw, 10);
            if (Number.isInteger(v) && String(v) === raw.trim() && v >= rule.int[0] && v <= rule.int[1]) out[key] = v;
        } else if (rule.num) {
            const v = parseFloat(raw);
            if (Number.isFinite(v) && v >= rule.num[0] && v <= rule.num[1]) out[key] = v;
        } else if (rule.oneOf) {
            if (rule.oneOf.includes(raw)) out[key] = raw;
        } else if (rule.ints) {
            const [lo, hi, max] = rule.ints;
            const parts = raw.split(',').map((p) => parseInt(p, 10));
            if (parts.length >= 1 && parts.length <= max
                && parts.every((v) => Number.isInteger(v) && v >= lo && v <= hi)) out[key] = parts;
        } else if (rule.nums) {
            const [lo, hi, max] = rule.nums;
            const parts = raw.split(',').map((p) => parseFloat(p));
            if (parts.length >= 1 && parts.length <= max
                && parts.every((v) => Number.isFinite(v) && v >= lo && v <= hi)) out[key] = parts;
        } else if (rule.text != null) {
            if (raw.length <= rule.text) out[key] = raw;
        }
    }
    return out;
}

export function buildLink(values) {
    const q = new URLSearchParams();
    for (const [k, v] of Object.entries(values)) q.set(k, String(v));
    return `#${q}`;
}

// debounced: dragging a slider would otherwise hit the browser's
// replaceState rate limit
let timer = 0;
export function putLinkInBar(hash) {
    clearTimeout(timer);
    timer = setTimeout(() => {
        try { history.replaceState(null, '', hash); } catch { /* rate limited */ }
    }, 250);
}

// copy the page's url with this hash; the button says what happened
export async function copyLink(btn, hash) {
    const url = location.href.split('#')[0] + hash;
    const label = btn.textContent;
    try {
        await navigator.clipboard.writeText(url);
        btn.textContent = 'copied';
    } catch {
        // no clipboard access (old browser, insecure origin) — the address bar has it
        try { history.replaceState(null, '', hash); } catch { /* rate limited */ }
        btn.textContent = 'link is in the address bar';
    }
    setTimeout(() => { btn.textContent = label; }, 1800);
}
