export function toBuffer(bigInts, L) {
    const arr = new Uint32Array(bigInts.length * L);
    bigInts.forEach((bn, i) => {
        let n = bn;
        for (let j = 0; j < L; j++) {
            arr[i * L + j] = Number(n & 0xFFFFFFFFn);
            n >>= 32n;
        }
    });
    return arr;
}

export function fromBuffer(arr, L) {
    const res = [];
    for (let i = 0; i < arr.length / L; i++) {
        let n = 0n;
        for (let j = L - 1; j >= 0; j--) {
            n = (n << 32n) | BigInt(arr[i * L + j]);
        }
        const msb = arr[i * L + L - 1] >>> 31;
        if (msb) {
            n = n - (1n << BigInt(L * 32));
        }
        res.push(n);
    }
    return res;
}

export function floatToBig(v, F) {
    const S = BigInt(F * 32);
    const val = BigInt(Math.round(v * Number(1n << 52n)));
    const shift = S - 52n;
    if (shift >= 0n) return val << shift;
    return val >> (-shift);
}

export function bigToFloatStr(n, F) {
    const S = BigInt(F * 32);
    const scale = 1n << S;
    let v = n;
    let sign = "";
    if (v < 0n) { sign = "-"; v = -v; }

    const intPart = v / scale;
    const fracPart = v % scale;
    const d = (fracPart * (10n ** 10n)) / scale;
    return `${sign}${intPart}.${d.toString().padStart(10, '0')}`;
}

export function hexToFixed(hex, L, F) {
    return toBuffer([BigInt("0x" + hex) << BigInt(F * 32)], L);
}
