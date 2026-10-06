// Small control helpers shared by the demo pages: segmented button groups,
// the brush picker, and canvases that behave like buttons.

// Mark one button of a segmented group active — for the eye (.is-active)
// and for screen readers (aria-pressed).
export function selectOne(box, btn) {
    box.querySelectorAll('.seg-btn').forEach((x) => {
        x.classList.toggle('is-active', x === btn);
        x.setAttribute('aria-pressed', String(x === btn));
    });
}

// A segmented button inside box; active marks the starting choice.
export function segButton(box, text, active, onPick) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'seg-btn' + (active ? ' is-active' : '');
    b.setAttribute('aria-pressed', String(active));
    b.textContent = text;
    b.addEventListener('click', () => {
        selectOne(box, b);
        onPick();
    });
    box.append(b);
    return b;
}

// brush sigma in cell units, see DrawBox.setBrush
const BRUSHES = [['thin', 0.75], ['medium', 1.15], ['thick', 1.7]];

// thin / medium / thick; onPick receives the sigma
export function buildBrushPicker(box, onPick, hint = 'strokes') {
    box.textContent = '';
    box.setAttribute('role', 'group');
    BRUSHES.forEach(([name, sigma], i) => {
        const b = segButton(box, name, i === 1, () => onPick(sigma));
        b.title = `${name} ${hint}`;
    });
}

// Make a canvas (or any element) act like a button: reachable with Tab,
// announced with a label, and fired by Enter or Space as well as a click.
export function makeClickable(el, label, onActivate) {
    el.tabIndex = 0;
    el.setAttribute('role', 'button');
    el.setAttribute('aria-label', label);
    el.title = label;
    el.addEventListener('click', onActivate);
    el.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            onActivate(e);
        }
    });
}

// true when the visitor asked the OS for less motion
export function prefersReducedMotion() {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

// The loading overlay covers the page, so a failed fetch needs its own way out.
// The message gets its own element: fetches still in flight keep reporting
// progress into textEl, which would otherwise overwrite the error.
export function showLoadError(textEl, err) {
    textEl.hidden = true;
    const msg = document.createElement('span');
    msg.textContent = `failed to load: ${err.message}`;
    const retry = document.createElement('button');
    retry.type = 'button';
    retry.className = 'mini-btn';
    retry.textContent = 'try again';
    retry.addEventListener('click', () => location.reload());
    textEl.after(msg, retry);
}
