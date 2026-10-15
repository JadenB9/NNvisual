// Runs in <head> before paint. The site marks light mode with data-theme="light"
// on <html>; dark is the default and carries no attribute. The choice lives in
// localStorage.theme so it carries across all of j4den.com, and with nothing
// saved the page follows the device. A classic script rather than a module so
// it blocks the first paint instead of flashing dark at people who chose light.
(function () {
    const root = document.documentElement;
    const COLOR = { dark: '#141719', light: '#ffffff' };

    function wanted() {
        try {
            const saved = localStorage.getItem('theme');
            if (saved === 'light' || saved === 'dark') return saved;
        } catch { /* private windows refuse localStorage; follow the device */ }
        return window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
    }

    function current() {
        return root.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
    }

    function apply(name) {
        if (name === 'light') root.setAttribute('data-theme', 'light');
        else root.removeAttribute('data-theme');
        document.querySelectorAll('meta[name="theme-color"]').forEach((m) => m.setAttribute('content', COLOR[name]));
        const btn = document.getElementById('theme-toggle');
        if (btn) {
            const label = `switch to ${name === 'light' ? 'dark' : 'light'} theme`;
            btn.setAttribute('aria-label', label);
            btn.title = label;
        }
    }

    // the canvases listen for this and repaint in the new palette
    function announce() {
        window.dispatchEvent(new CustomEvent('themechange', { detail: current() }));
    }

    apply(wanted());

    window.nnTheme = {
        get: current,
        set(name) {
            apply(name);
            try { localStorage.setItem('theme', name); } catch { /* fine, it just won't stick */ }
            announce();
        },
    };

    document.addEventListener('DOMContentLoaded', () => {
        apply(current()); // the button exists now, so label it
        const btn = document.getElementById('theme-toggle');
        if (btn) btn.addEventListener('click', () => window.nnTheme.set(current() === 'light' ? 'dark' : 'light'));
    });

    // nothing saved: keep following the device if it flips during the visit
    if (window.matchMedia) {
        window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', () => {
            try { if (localStorage.getItem('theme')) return; } catch { return; }
            apply(wanted());
            announce();
        });
    }
    // a toggle in another j4den.com tab
    window.addEventListener('storage', (e) => {
        if (e.key !== 'theme') return;
        apply(wanted());
        announce();
    });
})();
