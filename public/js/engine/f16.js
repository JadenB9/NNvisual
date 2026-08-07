// IEEE 754 half-precision codec. Weight files ship as fp16 to halve the
// download; the engine computes in fp32.

// Decode one u16 to a float.
export function f16ToF32(h) {
    const sign = (h & 0x8000) ? -1 : 1;
    const exp = (h >> 10) & 0x1f;
    const frac = h & 0x3ff;
    if (exp === 0) return sign * frac * 2 ** -24; // subnormal (frac/1024 * 2^-14)
    if (exp === 31) return frac ? NaN : sign * Infinity;
    return sign * (1 + frac / 1024) * 2 ** (exp - 15);
}

// Encode one float to u16 (round-to-nearest-even).
export function f32ToF16(x) {
    if (Number.isNaN(x)) return 0x7e00;
    const sign = (x < 0 || Object.is(x, -0)) ? 0x8000 : 0;
    x = Math.abs(x);
    if (x === Infinity) return sign | 0x7c00;
    if (x === 0) return sign;
    if (x >= 65520) return sign | 0x7c00; // overflows to inf after rounding
    if (x < 2 ** -24) return sign; // underflows to zero
    if (x < 2 ** -14) {
        // subnormal: value = frac * 2^-24
        const frac = Math.round(x * 2 ** 24);
        return sign | frac;
    }
    let exp = Math.floor(Math.log2(x));
    const mant = x / 2 ** exp; // in [1, 2)
    let frac = Math.round((mant - 1) * 1024);
    if (frac === 1024) { frac = 0; exp += 1; } // mantissa rounded up to 2.0
    if (exp + 15 >= 31) return sign | 0x7c00;
    return sign | ((exp + 15) << 10) | frac;
}

// Bulk decode: Uint16Array (or raw bytes) -> Float32Array.
export function decodeF16(u16) {
    const out = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) out[i] = f16ToF32(u16[i]);
    return out;
}

export function encodeF16(f32) {
    const out = new Uint16Array(f32.length);
    for (let i = 0; i < f32.length; i++) out[i] = f32ToF16(f32[i]);
    return out;
}
