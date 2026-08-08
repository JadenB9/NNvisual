// Horizontal probability bars (DOM, CSS-transitioned). Used for the 10
// digit classes and for top-k token distributions.
export class Bars {
    constructor(container, labels, { format = (v) => `${(v * 100).toFixed(1)}%` } = {}) {
        this.container = container;
        this.format = format;
        this.rows = [];
        container.textContent = '';
        container.classList.add('bars');
        for (const label of labels) {
            const row = document.createElement('div');
            row.className = 'bar-row';
            const name = document.createElement('span');
            name.className = 'bar-label';
            name.textContent = label;
            const track = document.createElement('span');
            track.className = 'bar-track';
            const fill = document.createElement('span');
            fill.className = 'bar-fill';
            track.append(fill);
            const value = document.createElement('span');
            value.className = 'bar-value';
            row.append(name, track, value);
            container.append(row);
            this.rows.push({ row, fill, value });
        }
    }

    update(values, highlight = -1) {
        for (let i = 0; i < this.rows.length; i++) {
            const v = values[i] || 0;
            this.rows[i].fill.style.width = `${Math.max(0, Math.min(1, v)) * 100}%`;
            this.rows[i].value.textContent = this.format(v);
            this.rows[i].row.classList.toggle('is-top', i === highlight);
        }
    }

    setLabels(labels) {
        labels.forEach((l, i) => {
            if (this.rows[i]) this.rows[i].row.firstChild.textContent = l;
        });
    }
}
