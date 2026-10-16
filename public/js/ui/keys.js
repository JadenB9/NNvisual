// Single-key shortcuts for the demo pages. A key does nothing while the
// visitor is typing in a field, and nothing with a modifier held, so browser
// shortcuts keep working.

// true when a keydown should be left alone
export function typingIn(target) {
    if (!target || !target.tagName) return false;
    const tag = target.tagName.toLowerCase();
    return tag === 'input' || tag === 'textarea' || tag === 'select' || target.isContentEditable === true;
}

// map: { 't': fn, ' ': fn, ... } keyed by event.key (lowercase for letters)
export function bindKeys(map) {
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey || e.metaKey || e.altKey || e.repeat) return;
        if (typingIn(e.target)) return;
        const fn = map[e.key.length === 1 ? e.key.toLowerCase() : e.key];
        if (!fn) return;
        e.preventDefault();
        fn(e);
    });
}

// "t" on a button's title so the shortcut is discoverable on hover
export function labelKey(btn, key) {
    const name = key === ' ' ? 'space' : key;
    btn.title = (btn.title ? `${btn.title} · ` : '') + `shortcut: ${name}`;
}
