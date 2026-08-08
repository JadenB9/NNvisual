// 28x28 drawing surface with a soft MNIST-style brush. The value is a
// Float32Array in [0,1]; every stroke fires onchange (per animation frame).
export class DrawBox {
    constructor(canvas, { size = 28, brush = 1.15, onchange = null } = {}) {
        this.canvas = canvas;
        this.size = size;
        this.brush = brush;
        this.onchange = onchange;
        this.value = new Float32Array(size * size);
        this.undoStack = [];
        this.drawing = false;
        this.last = null;
        this.pendingNotify = false;

        canvas.style.touchAction = 'none';
        canvas.addEventListener('pointerdown', (e) => {
            e.preventDefault();
            try { canvas.setPointerCapture(e.pointerId); } catch { /* synthetic pointers */ }
            this.undoStack.push(this.value.slice());
            if (this.undoStack.length > 30) this.undoStack.shift();
            this.drawing = true;
            this.last = this.toCell(e);
            this.stamp(this.last.x, this.last.y, e.pointerType === 'pen' ? e.pressure + 0.4 : 1);
            this.notify();
        });
        canvas.addEventListener('pointermove', (e) => {
            if (!this.drawing) return;
            const cur = this.toCell(e);
            this.line(this.last, cur, e.pointerType === 'pen' ? e.pressure + 0.4 : 1);
            this.last = cur;
            this.notify();
        });
        const stop = () => { this.drawing = false; this.last = null; };
        canvas.addEventListener('pointerup', stop);
        canvas.addEventListener('pointercancel', stop);

        this.render();
    }

    toCell(e) {
        const r = this.canvas.getBoundingClientRect();
        return {
            x: ((e.clientX - r.left) / r.width) * this.size,
            y: ((e.clientY - r.top) / r.height) * this.size,
        };
    }

    stamp(cx, cy, pressure = 1) {
        const rad = this.brush * 1.9;
        const s2 = 2 * this.brush * this.brush;
        const x0 = Math.max(0, Math.floor(cx - rad));
        const x1 = Math.min(this.size - 1, Math.ceil(cx + rad));
        const y0 = Math.max(0, Math.floor(cy - rad));
        const y1 = Math.min(this.size - 1, Math.ceil(cy + rad));
        for (let y = y0; y <= y1; y++) {
            for (let x = x0; x <= x1; x++) {
                const d2 = (x + 0.5 - cx) ** 2 + (y + 0.5 - cy) ** 2;
                const add = Math.exp(-d2 / s2) * 0.9 * pressure;
                const i = y * this.size + x;
                this.value[i] = Math.min(1, this.value[i] + add);
            }
        }
    }

    line(a, b, pressure) {
        const dist = Math.hypot(b.x - a.x, b.y - a.y);
        const steps = Math.max(1, Math.ceil(dist * 2));
        for (let s = 1; s <= steps; s++) {
            this.stamp(a.x + ((b.x - a.x) * s) / steps, a.y + ((b.y - a.y) * s) / steps, pressure);
        }
    }

    notify() {
        this.render();
        if (this.pendingNotify) return;
        this.pendingNotify = true;
        requestAnimationFrame(() => {
            this.pendingNotify = false;
            if (this.onchange) this.onchange(this.value);
        });
    }

    setValue(arr) {
        this.value.set(arr);
        this.render();
        if (this.onchange) this.onchange(this.value);
    }

    // brush sigma in cell units — thin ~0.75, medium ~1.15, thick ~1.7
    setBrush(sigma) {
        this.brush = sigma;
    }

    clear() {
        this.undoStack.push(this.value.slice());
        this.value.fill(0);
        this.render();
        if (this.onchange) this.onchange(this.value);
    }

    undo() {
        const prev = this.undoStack.pop();
        if (!prev) return;
        this.value.set(prev);
        this.render();
        if (this.onchange) this.onchange(this.value);
    }

    get empty() {
        for (let i = 0; i < this.value.length; i++) if (this.value[i] > 0.02) return false;
        return true;
    }

    render() {
        const ctx = this.canvas.getContext('2d');
        const px = this.canvas.width / this.size;
        ctx.fillStyle = '#0e1011';
        ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        for (let y = 0; y < this.size; y++) {
            for (let x = 0; x < this.size; x++) {
                const v = this.value[y * this.size + x];
                if (v <= 0.01) continue;
                const c = Math.round(14 + v * (238 - 14));
                ctx.fillStyle = `rgb(${c},${c},${Math.round(17 + v * (227 - 17))})`;
                ctx.fillRect(x * px, y * px, px, px);
            }
        }
        // faint grid so the 28x28 structure reads
        ctx.strokeStyle = 'rgba(51,58,64,0.25)';
        ctx.lineWidth = 1;
        for (let i = 0; i <= this.size; i += 7) {
            ctx.beginPath();
            ctx.moveTo(i * px, 0);
            ctx.lineTo(i * px, this.canvas.height);
            ctx.moveTo(0, i * px);
            ctx.lineTo(this.canvas.width, i * px);
            ctx.stroke();
        }
    }
}
