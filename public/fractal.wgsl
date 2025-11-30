
const L: u32 = 64u;
const F: u32 = 32u;

struct LargeInt { limbs: array<u32, L>, };
struct Data { values: array<LargeInt>, };

// For Compute Shaders
@group(0) @binding(0) var<storage, read> bufA : Data;
@group(0) @binding(1) var<storage, read> bufB : Data;
@group(0) @binding(2) var<storage, read_write> bufR : Data;
@group(0) @binding(3) var<storage, read> bufC : Data;

// For Fragment Shader
struct Camera {
    centerX: LargeInt,
    centerY: LargeInt,
    scale: LargeInt,
    resolution: vec2<f32>,
    maxIter: u32,
    padding: u32,
};
@group(0) @binding(0) var<storage, read> cam : Camera;

// --- CORE MATH ---

fn mac(a: u32, b: u32, c: u32, carry: u32) -> vec2<u32> {
    let la=a&0xFFFFu; let ha=a>>16u; let lb=b&0xFFFFu; let hi_b=b>>16u;
    let p0=la*lb; let p1=la*hi_b; let p2=ha*lb; let p3=ha*hi_b;
    let t=p1+(p0>>16u);
    var lo=((t<<16u)|(p0&0xFFFFu)); var hi=p3+(t>>16u)+(p2>>16u);
    let t2=lo+(p2<<16u); if(t2<lo){hi++;} lo=t2;
    let t3=lo+c; if(t3<lo){hi++;} lo=t3;
    let t4=lo+carry; if(t4<lo){hi++;} lo=t4;
    return vec2<u32>(lo, hi);
}

fn add(a: ptr<function, LargeInt>, b: LargeInt) {
    var c=0u;
    for(var i=0u; i<L; i++){
        let v=(*a).limbs[i];
        let s=v+b.limbs[i]+c;
        c=u32(s<v||(c==1u&&s==v));
        (*a).limbs[i]=s;
    }
}

fn sub(a: ptr<function, LargeInt>, b: LargeInt) {
    var r=0u;
    for(var i=0u; i<L; i++){
        let v=(*a).limbs[i];
        let o=b.limbs[i];
        let d=v-o-r;
        r=u32(v<o)|(u32(v==o)&r);
        (*a).limbs[i]=d;
    }
}

fn neg(a: ptr<function, LargeInt>) {
    var c=1u;
    for(var i=0u; i<L; i++) {
        let v = ~(*a).limbs[i];
        let s = v + c;
        if (s < v) { c = 1u; } else { c = 0u; }
        if (s == 0u && v == 0xFFFFFFFFu) { c = 1u; }
        (*a).limbs[i] = s;
    }
}

fn is_neg(a: LargeInt) -> bool {
    return (a.limbs[L-1u] >> 31u) == 1u;
}

fn shl1(a: ptr<function, LargeInt>) {
    var c=0u;
    for(var i=0u; i<L; i++){
        let v=(*a).limbs[i];
        let n=(v<<1u)|c;
        c=v>>31u;
        (*a).limbs[i]=n;
    }
}

fn gte(a: ptr<function, LargeInt>, b: LargeInt) -> bool {
    for(var k=0u; k<L; k++) {
        let i=L-1u-k;
        if((*a).limbs[i]>b.limbs[i]){return true;}
        if((*a).limbs[i]<b.limbs[i]){return false;}
    }
    return true;
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

fn mul_fixed_op_unsigned(a: LargeInt, b: LargeInt) -> LargeInt {
    var t: array<u32, L*2>;
    for(var i=0u; i<L; i++){
        var c=0u;
        // Optimization: skip high limbs that will be discarded (>= F+L)
        let limit = min(L, F + L - i);
        for(var j=0u; j<limit; j++){
            let r=mac(a.limbs[i], b.limbs[j], t[i+j], c);
            t[i+j]=r.x;
            c=r.y;
        }
    }
    var res:LargeInt;
    for(var i=0u; i<L; i++){ res.limbs[i]=t[i+F]; }
    return res;
}

fn mul_int_op(a: LargeInt, b: LargeInt) -> LargeInt {
    var res:LargeInt; for(var k=0u;k<L;k++){res.limbs[k]=0u;}
    for(var i=0u;i<L;i++){ var c=0u; for(var j=0u;j<L;j++){ if(i+j<L){ let r=mac(a.limbs[i],b.limbs[j],res.limbs[i+j],c); res.limbs[i+j]=r.x; c=r.y; } } }
    return res;
}

fn mul_fixed(a: LargeInt, b: LargeInt) -> LargeInt {
    var va = a; var vb = b;
    let sa = is_neg(va);
    let sb = is_neg(vb);
    if(sa) { neg(&va); }
    if(sb) { neg(&vb); }

    var res = mul_fixed_op_unsigned(va, vb);

    if (sa != sb) { neg(&res); }
    return res;
}

fn sqr_fixed(a: LargeInt) -> LargeInt {
    return mul_fixed(a, a);
}

fn zero_big() -> LargeInt {
    var res: LargeInt;
    for(var i=0u; i<L; i++) { res.limbs[i] = 0u; }
    return res;
}

fn int_to_big(v: i32) -> LargeInt {
    var res = zero_big();
    let u = u32(v);
    res.limbs[0] = u;
    var fill = 0u;
    if (v < 0) { fill = 0xFFFFFFFFu; }
    for(var i=1u; i<L; i++) { res.limbs[i] = fill; }
    return res;
}

fn to_float_approx(v: LargeInt) -> f32 {
    let sign_bit = v.limbs[L-1u] >> 31u;
    let fill = sign_bit * 0xFFFFFFFFu;
    // Check if high limbs match sign extension
    for(var k=F+1u; k<L; k++) {
        if (v.limbs[k] != fill) { return 999999.0; }
    }
    return f32(i32(v.limbs[F])) + f32(v.limbs[F-1u]) / 4294967296.0;
}

// --- COMPUTE SHADER OPS ---

@compute @workgroup_size(64)
fn op_add(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x; if(i>=arrayLength(&bufA.values)){return;}
    var r=bufA.values[i]; add(&r, bufB.values[i]); bufR.values[i]=r;
}
@compute @workgroup_size(64)
fn op_sub(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x; if(i>=arrayLength(&bufA.values)){return;}
    var r=bufA.values[i]; sub(&r, bufB.values[i]); bufR.values[i]=r;
}
@compute @workgroup_size(64)
fn op_mul(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x; if(i>=arrayLength(&bufA.values)){return;}
    bufR.values[i] = mul_fixed(bufA.values[i], bufB.values[i]);
}
@compute @workgroup_size(64)
fn op_mul_int(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x; if(i>=arrayLength(&bufA.values)){return;}
    bufR.values[i] = mul_int_op(bufA.values[i], bufB.values[i]);
}
@compute @workgroup_size(64)
fn op_div(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x; if(i>=arrayLength(&bufA.values)){return;}
    var rem = zero_big(); var quo = zero_big();
    let total = (L + F) * 32u;
    let B = bufB.values[i];
    for(var k=0u; k<total; k++) {
        let virt_idx = total - 1u - k;
        var bit = 0u;
        if(virt_idx >= F*32u) {
            let r_idx = virt_idx - F*32u;
            if(r_idx < L*32u) { bit = (bufA.values[i].limbs[r_idx/32u] >> (r_idx%32u)) & 1u; }
        }
        shl1(&rem); rem.limbs[0] |= bit;
        if(gte(&rem, B)) { sub(&rem, B); quo.limbs[0] |= 1u; }
        if(k < total - 1u) { shl1(&quo); }
    }
    bufR.values[i] = quo;
}

@compute @workgroup_size(64)
fn op_mod(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x; if(i>=arrayLength(&bufA.values)){return;}
    var rem = zero_big(); let B = bufB.values[i];
    for(var k=0u; k<L*32u; k++) {
        let idx = L*32u - 1u - k;
        let bit = (bufA.values[i].limbs[idx/32u] >> (idx%32u)) & 1u;
        shl1(&rem); rem.limbs[0] |= bit;
        if(gte(&rem, B)) { sub(&rem, B); }
    }
    bufR.values[i] = rem;
}
@compute @workgroup_size(64)
fn op_sqrt(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x; if(i>=arrayLength(&bufA.values)){return;}
    var rem = zero_big(); var root = zero_big();
    for(var k=0u; k<L*16u; k++) {
        let pair = (L*16u) - 1u - k;
        let glob_bit = pair * 2u;
        let val = (bufA.values[i].limbs[glob_bit/32u] >> (glob_bit%32u)) & 3u;
        shl1(&root);
        var c=0u; for(var z=0u;z<L;z++){let v=rem.limbs[z];let n=(v<<2u)|c;c=v>>30u;rem.limbs[z]=n;}
        rem.limbs[0] |= val;
        var cand = root; cand.limbs[0] |= 1u;
        if(gte(&rem, cand)) { sub(&rem, cand); root.limbs[0] |= 2u; }
    }
    bufR.values[i] = root;
}
@compute @workgroup_size(64)
fn op_exp(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x; if(i>=arrayLength(&bufA.values)){return;}
    let x = bufA.values[i];
    var term = x;
    var sum = zero_big(); sum.limbs[F] = 1u; add(&sum, x);
    for(var k=2u; k<30u; k++) {
        term = mul_fixed_op_unsigned(term, x);
        div_scalar(&term, k);
        add(&sum, term);
        var is_0=true; for(var z=0u;z<L;z++){if(term.limbs[z]!=0u){is_0=false;break;}} if(is_0){break;}
    }
    bufR.values[i] = sum;
}
@compute @workgroup_size(64)
fn op_modpow(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x; if(idx >= arrayLength(&bufA.values)){return;}
    var base = bufA.values[idx];
    let exp  = bufB.values[idx];
    let mod_val  = bufC.values[idx];
    var res = zero_big(); res.limbs[0] = 1u;

    for (var k=0u; k<L*32u; k++) {
        if (((exp.limbs[k/32u] >> (k%32u)) & 1u) == 1u) {
            var p = mul_int_op(res, base);
            var rem = zero_big();
            for(var b=0u; b<L*32u; b++) {
                let bi = L*32u - 1u - b;
                let bit = (p.limbs[bi/32u] >> (bi%32u)) & 1u;
                shl1(&rem); rem.limbs[0] |= bit;
                if(gte(&rem, mod_val)) { sub(&rem, mod_val); }
            }
            res = rem;
        }
        var p2 = mul_int_op(base, base);
        var rem2 = zero_big();
        for(var b=0u; b<L*32u; b++) {
            let bi = L*32u - 1u - b;
            let bit = (p2.limbs[bi/32u] >> (bi%32u)) & 1u;
            shl1(&rem2); rem2.limbs[0] |= bit;
            if(gte(&rem2, mod_val)) { sub(&rem2, mod_val); }
        }
        base = rem2;
    }
    bufR.values[idx] = res;
}
@compute @workgroup_size(64)
fn op_trig(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x; if(i>=arrayLength(&bufA.values)){return;}
    let is_cos = (bufC.values[0].limbs[0] == 1u);
    var x = bufA.values[i];
    let two_pi = bufB.values[0];
    var rem = zero_big();
    for(var k=0u; k<L*32u; k++) {
       let idx = L*32u - 1u - k;
       let bit = (x.limbs[idx/32u] >> (idx%32u)) & 1u;
       shl1(&rem); rem.limbs[0] |= bit;
       if(gte(&rem, two_pi)) { sub(&rem, two_pi); }
    }
    x = rem;
    let x_sq = mul_fixed_op_unsigned(x, x);
    var term = x;
    var sum = x;
    if (is_cos) { term=zero_big(); term.limbs[F]=1u; sum=term; }
    for(var iter=1u; iter<=22u; iter++) {
      term = mul_fixed_op_unsigned(term, x_sq);
      let k = iter * 2u;
      var d = k*(k+1u); if(is_cos){ d=(k-1u)*k; }
      div_scalar(&term, d);
      if(iter%2u==1u){ sub(&sum, term); } else { add(&sum, term); }
    }
    bufR.values[i] = sum;
}

// --- FRACTAL RENDERING ---

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var pos = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
        vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0)
    );
    var out: VertexOutput;
    out.pos = vec4<f32>(pos[idx], 0.0, 1.0);
    out.uv = pos[idx];
    return out;
}

@fragment
fn fs_main(inp: VertexOutput) -> @location(0) vec4<f32> {
    let uv = inp.uv;

    let px = i32(uv.x * (cam.resolution.x * 0.5));
    let py = i32(uv.y * (cam.resolution.y * 0.5));

    let offX = int_to_big(px);
    let offY = int_to_big(py);

    var dx = mul_int_op(offX, cam.scale);
    var dy = mul_int_op(offY, cam.scale);

    var cx = cam.centerX; add(&cx, dx);
    var cy = cam.centerY; add(&cy, dy);

    // Performance Optimization: Check for Cardioid and Period-2 Bulb
    let x = to_float_approx(cx);
    let y = to_float_approx(cy);
    let y2 = y*y;
    let q = (x - 0.25) * (x - 0.25) + y2;
    if (q * (q + (x - 0.25)) < 0.25 * y2) { return vec4(0.0, 0.0, 0.0, 1.0); }
    if ((x + 1.0) * (x + 1.0) + y2 < 0.0625) { return vec4(0.0, 0.0, 0.0, 1.0); }

    var zx = zero_big();
    var zy = zero_big();
    var zx2: LargeInt;
    var zy2: LargeInt;

    var iter = 0u;
    let MAX = cam.maxIter;

    for (var i=0u; i<MAX; i++) {
        zx2 = sqr_fixed(zx);
        zy2 = sqr_fixed(zy);

        var mag = zx2; add(&mag, zy2);
        if (mag.limbs[F] >= 4u) { break; }

        var next_zy = mul_fixed(zx, zy);
        shl1(&next_zy);
        add(&next_zy, cy);

        var next_zx = zx2;
        sub(&next_zx, zy2);
        add(&next_zx, cx);

        zx = next_zx;
        zy = next_zy;
        iter = i;
    }

    let t = f32(iter) / f32(MAX);
    if (iter == MAX) { return vec4(0.0, 0.0, 0.0, 1.0); }

    let r = 0.5 + 0.5*cos(3.0 + t*10.0 + 0.0);
    let g = 0.5 + 0.5*cos(3.0 + t*10.0 + 0.6);
    let b = 0.5 + 0.5*cos(3.0 + t*10.0 + 1.0);

    return vec4(r, g, b, 1.0);
}
