import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { hexToRgb, withAlpha } from '../public/js/ui/theme.js';

test('hex colors decode to rgb triples', () => {
    assert.deepEqual(hexToRgb('#0e1011'), [14, 16, 17]);
    assert.deepEqual(hexToRgb('#eeebe3'), [238, 235, 227]);
    assert.deepEqual(hexToRgb(' #FFF '), [255, 255, 255]);
    assert.equal(withAlpha('#86c3d6', 0.65), 'rgba(134,195,214,0.65)');
});

// Every color a canvas can draw with must exist in both token sets, or a
// light-mode canvas would read an empty string and paint black.
test('both css token sets define every token the canvases read', () => {
    const css = readFileSync(new URL('../public/css/base.css', import.meta.url), 'utf8');
    const block = (sel) => {
        const start = css.indexOf(sel);
        assert.ok(start >= 0, `${sel} block present`);
        return css.slice(start, css.indexOf('}', start));
    };
    const dark = block(':root {');
    const light = block('[data-theme="light"] {');
    const names = [...dark.matchAll(/--([a-z0-9-]+):\s*#[0-9a-f]{6}/g)].map((m) => m[1]);
    assert.ok(names.includes('ink') && names.includes('gold') && names.includes('link-blue'));
    for (const n of names) {
        assert.match(light, new RegExp(`--${n}:\\s*#[0-9a-f]{6}`), `light theme defines --${n}`);
    }
});

// theme-boot.js is a classic script that runs in <head>; every page must load
// it before its stylesheets so the first paint already has the right theme.
test('every page loads theme-boot.js before its css and has a toggle', () => {
    for (const page of ['index', 'mlp', 'cnn', 'transformer', 'diffusion']) {
        const html = readFileSync(new URL(`../public/${page}.html`, import.meta.url), 'utf8');
        const boot = html.indexOf('js/theme-boot.js');
        const css = html.indexOf('css/base.css');
        assert.ok(boot > 0 && boot < css, `${page}: theme-boot before base.css`);
        assert.ok(html.includes('id="theme-toggle"'), `${page}: has the toggle`);
        assert.equal((html.match(/name="theme-color"/g) || []).length, 2, `${page}: theme-color for both schemes`);
    }
});
