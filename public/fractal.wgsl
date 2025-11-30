
const L: u32 = 64u;
const F: u32 = 32u;

struct LargeInt { limbs: array<u32, L>, };
struct Data { values: array<LargeInt>, };

// Buffers
@group(0) @binding(0) var<storage, read> bufA : Data;
@group(0) @binding(1) var<storage, read> bufB : Data;
@group(0) @binding(2) var<storage, read_write> bufR : Data;
@group(0) @binding(3) var<storage, read> bufC : Data;

struct Camera {
    centerX: LargeInt,
    centerY: LargeInt,
    scale: LargeInt,
    resolution: vec2<f32>,
    maxIter: u32,
    padding: u32,
};
@group(0) @binding(0) var<storage, read> cam : Camera;

// --- MATH OPS (BRANCHLESS & OPTIMIZED) ---

// Branchless MAC: (a * b) + c + carry -> (lo, hi)
// Uses decomposition to avoid 64-bit overflow (since WGSL u32*u32=u32)
fn mac(a: u32, b: u32, c: u32, carry: u32) -> vec2<u32> {
    let la = a & 0xFFFFu; let ha = a >> 16u;
    let lb = b & 0xFFFFu; let hb = b >> 16u;

    // Partial products
    let p0 = la * lb; // < 2^32
    let p1 = la * hb; // < 2^32
    let p2 = ha * lb; // < 2^32
    let p3 = ha * hb; // < 2^32

    // Combine to 64-bit result (lo, hi)
    // Result = p0 + (p1<<16) + (p2<<16) + (p3<<32) + c + carry

    // 1. Accumulate p1 into p0's upper half
    let t1 = p0 + (p1 << 16u);
    // Carry from t1 is (p1 >> 16) plus overflow from addition
    // Overflow check: t1 < p0 (or just check high bits)
    // Actually, simple addition of u32s is safer if we track carries explicitly.

    // Robust Column Addition Method:
    // Col 0 (bits 0-15): p0_lo
    // Col 1 (bits 16-31): p0_hi + p1_lo + p2_lo
    // Col 2 (bits 32-47): p1_hi + p2_hi + p3_lo
    // Col 3 (bits 48-63): p3_hi

    // Implementation:
    let col0 = p0;

    // Add p1 aligned at 16
    let s1 = col0 + (p1 << 16u);
    let c1 = u32(s1 < col0) + (p1 >> 16u);

    // Add p2 aligned at 16
    let s2 = s1 + (p2 << 16u);
    let c2 = u32(s2 < s1) + (p2 >> 16u);

    let lo = s2;
    let hi_base = p3 + c1 + c2;

    // Add accumulated 'c' and 'carry' from previous limb
    // lo += c
    let s3 = lo + c;
    let c3 = u32(s3 < lo);
    let lo_final = s3;

    // hi += carry + c3
    // Note: 'carry' input is the carry-in to the MAC operation itself (usually 0 or small)
    // Actually, the 'carry' argument in mac() signature usually acts as an extra addend.
    // In mul_u loop: mac(..., c) -> c is the running carry from previous column.
    // Wait, the signature is `mac(a, b, c, carry)`.
    // c is `t[i+j]`, carry is `c` (running carry).
    // So we add `c` and `carry` to the 64-bit result.

    // lo = lo_final
    // hi = hi_base + c3
    var hi_final = hi_base + c3;

    // Add 'carry' argument
    let s4 = lo_final + carry;
    hi_final += u32(s4 < lo_final);

    return vec2<u32>(s4, hi_final);
}

fn add(a: ptr<function, LargeInt>, b: LargeInt) {
    var c = 0u;
    for(var i=0u; i<L; i++) {
        let v = (*a).limbs[i];
        let s = v + b.limbs[i] + c;
        c = u32(s < v) | (u32(s == v) & c);
        (*a).limbs[i] = s;
    }
}

fn sub(a: ptr<function, LargeInt>, b: LargeInt) {
    var c = 0u;
    for(var i=0u; i<L; i++) {
        let v = (*a).limbs[i];
        let o = b.limbs[i];
        let d = v - o - c;
        c = u32(v < o) | (u32(v == o) & c);
        (*a).limbs[i] = d;
    }
}

fn neg(a: ptr<function, LargeInt>) {
    var c = 1u;
    for(var i=0u; i<L; i++) {
        let v = ~(*a).limbs[i];
        let s = v + c;
        c = u32(s < v) | (u32(s == 0u && v == 0xFFFFFFFFu) & c);
        (*a).limbs[i] = s;
    }
}

fn shl1(a: ptr<function, LargeInt>) {
    var c = 0u;
    for(var i=0u; i<L; i++) {
        let v = (*a).limbs[i];
        (*a).limbs[i] = (v << 1u) | c;
        c = v >> 31u;
    }
}

fn is_neg(a: LargeInt) -> bool { return (a.limbs[L-1u] >> 31u) == 1u; }
fn zero() -> LargeInt { var z: LargeInt; return z; }

fn gte(a: LargeInt, b: LargeInt) -> bool {
    for(var k=0u; k<L; k++) {
        let i = L - 1u - k;
        if(a.limbs[i] > b.limbs[i]){ return true; }
        if(a.limbs[i] < b.limbs[i]){ return false; }
    }
    return true;
}

fn mul_u(a: LargeInt, b: LargeInt) -> LargeInt {
    var t: array<u32, L*2>;
    for(var i=0u; i<L; i++) {
        var c = 0u;
        for(var j=0u; j<L; j++) {
            let r = mac(a.limbs[i], b.limbs[j], t[i+j], c);
            t[i+j] = r.x;
            c = r.y;
        }
        t[i+L] = c;
    }
    var res: LargeInt;
    for(var i=0u; i<L; i++) { res.limbs[i] = t[i+F]; }
    return res;
}

fn mul_fixed(a: LargeInt, b: LargeInt) -> LargeInt {
    var va = a; var vb = b;
    let sa = is_neg(va); let sb = is_neg(vb);
    if(sa) { neg(&va); } if(sb) { neg(&vb); }
    var res = mul_u(va, vb);
    if(sa != sb) { neg(&res); }
    return res;
}

fn sqr_fixed(a: LargeInt) -> LargeInt {
    var va = a;
    if(is_neg(va)) { neg(&va); }
    return mul_u(va, va);
}

fn mul_int(a: LargeInt, b: LargeInt) -> LargeInt {
    var res: LargeInt;
    for(var i=0u; i<L; i++) {
        var c = 0u;
        for(var j=0u; j<L; j++) {
            if (i+j < L) {
                let r = mac(a.limbs[i], b.limbs[j], res.limbs[i+j], c);
                res.limbs[i+j] = r.x;
                c = r.y;
            }
        }
    }
    return res;
}

fn int_to_big(v: i32) -> LargeInt {
    var r = zero(); r.limbs[0] = u32(v);
    if(v < 0) { for(var i=1u; i<L; i++) { r.limbs[i] = 0xFFFFFFFFu; } }
    return r;
}

fn div_scalar(a: ptr<function, LargeInt>, b: u32) {
    var r = 0u;
    for(var k=0u; k<L; k++){
        let i = L - 1u - k;
        let v = (*a).limbs[i];
        var current_r = r;
        var q = 0u;
        for (var bit_idx=0u; bit_idx<32u; bit_idx++) {
            let shift = 31u - bit_idx;
            let bit = (v >> shift) & 1u;
            current_r = (current_r << 1u) | bit;
            q = q << 1u;
            if (current_r >= b) {
                current_r -= b;
                q |= 1u;
            }
        }
        (*a).limbs[i] = q;
        r = current_r;
    }
}

// --- COMPUTE KERNELS ---

@compute @workgroup_size(64) fn op_add(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x; if(i>=arrayLength(&bufA.values)){return;}
    var r=bufA.values[i]; add(&r, bufB.values[i]); bufR.values[i]=r;
}

@compute @workgroup_size(64) fn op_sub(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x; if(i>=arrayLength(&bufA.values)){return;}
    var r=bufA.values[i]; sub(&r, bufB.values[i]); bufR.values[i]=r;
}

@compute @workgroup_size(64) fn op_mul(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x; if(i>=arrayLength(&bufA.values)){return;}
    bufR.values[i] = mul_fixed(bufA.values[i], bufB.values[i]);
}

@compute @workgroup_size(64) fn op_mul_int(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x; if(i>=arrayLength(&bufA.values)){return;}
    bufR.values[i] = mul_int(bufA.values[i], bufB.values[i]);
}

@compute @workgroup_size(64) fn op_div(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x; if(i>=arrayLength(&bufA.values)){return;}
    var rem = zero(); var quo = zero();
    let B = bufB.values[i];
    let total = (L + F) * 32u;
    for(var k=0u; k<total; k++) {
        let bit_idx = total - 1u - k;
        var bit = 0u;
        if(bit_idx >= F*32u) {
            let r = bit_idx - F*32u;
            if(r < L*32u) { bit = (bufA.values[i].limbs[r/32u] >> (r%32u)) & 1u; }
        }
        shl1(&rem); rem.limbs[0] |= bit;
        if (gte(rem, B)) { sub(&rem, B); shl1(&quo); quo.limbs[0] |= 1u; }
        else { shl1(&quo); }
    }
    bufR.values[i] = quo;
}

@compute @workgroup_size(64) fn op_mod(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x; if(i>=arrayLength(&bufA.values)){return;}
    var rem = zero(); let B = bufB.values[i];
    for(var k=0u; k<L*32u; k++) {
        let idx = L*32u - 1u - k;
        let bit = (bufA.values[i].limbs[idx/32u] >> (idx%32u)) & 1u;
        shl1(&rem); rem.limbs[0] |= bit;
        if(gte(rem, B)) { sub(&rem, B); }
    }
    bufR.values[i] = rem;
}

@compute @workgroup_size(64) fn op_sqrt(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x; if(i>=arrayLength(&bufA.values)){return;}
    var rem = zero(); var root = zero();
    for(var k=0u; k<L*16u; k++) {
        let pair = (L*16u) - 1u - k;
        let val = (bufA.values[i].limbs[pair*2u/32u] >> ((pair*2u)%32u)) & 3u;

        var c=0u; for(var z=0u;z<L;z++){let v=rem.limbs[z];let n=(v<<2u)|c;c=v>>30u;rem.limbs[z]=n;}
        rem.limbs[0] |= val;

        var cand = root; shl1(&cand); cand.limbs[0] |= 1u;
        shl1(&root);
        if(gte(rem, cand)) { sub(&rem, cand); root.limbs[0] |= 2u; }
    }
    bufR.values[i] = root;
}

@compute @workgroup_size(64) fn op_exp(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x; if(i>=arrayLength(&bufA.values)){return;}
    let x = bufA.values[i];
    var term = x;
    var sum = zero(); sum.limbs[F] = 1u; add(&sum, x);
    for(var k=2u; k<30u; k++) {
        term = mul_fixed(term, x);
        div_scalar(&term, k);
        add(&sum, term);
        var is_0=true; for(var z=0u;z<L;z++){if(term.limbs[z]!=0u){is_0=false;break;}} if(is_0){break;}
    }
    bufR.values[i] = sum;
}

@compute @workgroup_size(64) fn op_modpow(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x; if(idx >= arrayLength(&bufA.values)){return;}
    var base = bufA.values[idx];
    let exp  = bufB.values[idx];
    let mod_val  = bufC.values[idx];
    var res = zero(); res.limbs[0] = 1u;

    for (var k=0u; k<L*32u; k++) {
        if (((exp.limbs[k/32u] >> (k%32u)) & 1u) == 1u) {
            var p = mul_int(res, base);
            var rem = zero();
            for(var b=0u; b<L*32u; b++) {
                let bi = L*32u - 1u - b;
                let bit = (p.limbs[bi/32u] >> (bi%32u)) & 1u;
                shl1(&rem); rem.limbs[0] |= bit;
                if(gte(rem, mod_val)) { sub(&rem, mod_val); }
            }
            res = rem;
        }
        var p2 = mul_int(base, base);
        var rem2 = zero();
        for(var b=0u; b<L*32u; b++) {
            let bi = L*32u - 1u - b;
            let bit = (p2.limbs[bi/32u] >> (bi%32u)) & 1u;
            shl1(&rem2); rem2.limbs[0] |= bit;
            if(gte(rem2, mod_val)) { sub(&rem2, mod_val); }
        }
        base = rem2;
    }
    bufR.values[idx] = res;
}

@compute @workgroup_size(64) fn op_trig(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x; if(i>=arrayLength(&bufA.values)){return;}
    let is_cos = (bufC.values[0].limbs[0] == 1u);
    var x = bufA.values[i]; let two_pi = bufB.values[0];

    var rem = zero();
    for(var k=0u; k<L*32u; k++) {
       let idx = L*32u - 1u - k;
       let bit = (x.limbs[idx/32u] >> (idx%32u)) & 1u;
       shl1(&rem); rem.limbs[0] |= bit;
       if(gte(rem, two_pi)) { sub(&rem, two_pi); }
    }
    x = rem;

    let x_sq = mul_fixed(x, x);
    var term = x; var sum = x;
    if (is_cos) { term=zero(); term.limbs[F]=1u; sum=term; }

    for(var iter=1u; iter<=16u; iter++) {
        term = mul_fixed(term, x_sq);
        let k = iter * 2u;
        var d = k*(k+1u); if(is_cos){ d=(k-1u)*k; }

        div_scalar(&term, d);

        if(iter%2u==1u){ sub(&sum, term); } else { add(&sum, term); }
    }
    bufR.values[i] = sum;
}

// --- FRACTAL SHADER ---

struct VertexOutput { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };

@vertex fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var pos = array<vec2<f32>, 6>(vec2(-1.,-1.), vec2(1.,-1.), vec2(-1.,1.), vec2(-1.,1.), vec2(1.,-1.), vec2(1.,1.));
    return VertexOutput(vec4(pos[idx], 0., 1.), pos[idx]);
}

@fragment fn fs_main(inp: VertexOutput) -> @location(0) vec4<f32> {
    let px = i32(inp.uv.x * cam.resolution.x * 0.5);
    let py = i32(inp.uv.y * cam.resolution.y * 0.5);
    var dx = mul_int(int_to_big(px), cam.scale);
    var dy = mul_int(int_to_big(py), cam.scale);
    var cx = cam.centerX; add(&cx, dx);
    var cy = cam.centerY; add(&cy, dy);

    var zx = zero(); var zy = zero();
    var iter = 0u;

    for (var i=0u; i<cam.maxIter; i++) {
        var zx_sq = sqr_fixed(zx);
        var zy_sq = sqr_fixed(zy);

        var mag = zx_sq; add(&mag, zy_sq);
        if (mag.limbs[F] >= 4u) { break; }

        var two_xy = mul_fixed(zx, zy);
        shl1(&two_xy);
        add(&two_xy, cy); // New zy

        sub(&zx_sq, zy_sq);
        add(&zx_sq, cx);    // New zx

        zy = two_xy;
        zx = zx_sq;
        iter = i;
    }

    if (iter == cam.maxIter) { return vec4(0.,0.,0.,1.); }
    let t = f32(iter) / f32(cam.maxIter);
    let c = vec3(0.5) + vec3(0.5) * cos(vec3(3., 3.6, 4.) + t*10.);
    return vec4(c, 1.);
}
