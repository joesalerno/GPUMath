class GPUBigNum {
  constructor(limbCount = 64) {
    this.L = limbCount; // 64 * 32-bit = 2048-bit precision
    this.device = null;
    this.pipelines = {};
  }

  async init() {
    if (!navigator.gpu) throw new Error("WebGPU not supported");
    this.device = await (await navigator.gpu.requestAdapter()).requestDevice();

    const shaderCode = `
      override L: u32 = ${this.L}u; 
      
      struct BigInt   { limbs: array<u32, L> }     // 2048-bit
      struct BigIntx2 { limbs: array<u32, L * 2> } // 4096-bit
      
      struct DataIn   { values: array<BigInt> }
      struct DataOut  { values: array<BigIntx2> } // Output is always allocated wide
      
      @group(0) @binding(0) var<storage, read> bufA : DataIn;
      @group(0) @binding(1) var<storage, read> bufB : DataIn;
      @group(0) @binding(2) var<storage, read_write> bufR : DataOut;
      @group(0) @binding(3) var<storage, read> bufM : DataIn; // Optional Modulus

      // --- CORE MATH HELPERS ---

      // 32x32 -> 64-bit Multiply-Accumulate
      fn mac(a: u32, b: u32, c: u32, carry_in: u32) -> vec2<u32> {
        let lo_a = a & 0xFFFFu; let hi_a = a >> 16u;
        let lo_b = b & 0xFFFFu; let hi_b = b >> 16u;
        let p0 = lo_a * lo_b; let p1 = lo_a * hi_b;
        let p2 = hi_a * lo_b; let p3 = hi_a * hi_b;
        let t = p1 + (p0 >> 16u);
        let p0_final = (t << 16u) | (p0 & 0xFFFFu);
        var lo = p0_final;
        var hi = p3 + (t >> 16u) + (p2 >> 16u);
        let t2 = lo + (p2 << 16u); if (t2 < lo) { hi++; } lo = t2;
        let t3 = lo + c; if (t3 < lo) { hi++; } lo = t3;
        let t4 = lo + carry_in; if (t4 < lo) { hi++; } lo = t4;
        return vec2<u32>(lo, hi);
      }

      // Internal Multiplication: A(L) * B(L) -> Res(2L)
      fn mul_core(a: BigInt, b: BigInt) -> BigIntx2 {
        var res: BigIntx2;
        for (var k=0u; k<L*2; k++) { res.limbs[k] = 0u; }
        for (var i=0u; i<L; i++) {
          var carry = 0u;
          let val_a = a.limbs[i];
          for (var j=0u; j<L; j++) {
            let r = mac(val_a, b.limbs[j], res.limbs[i+j], carry);
            res.limbs[i+j] = r.x;
            carry = r.y;
          }
          res.limbs[i+L] = carry;
        }
        return res;
      }

      // Internal Modulo: A(2L) % M(L) -> Res(L) (Restoring Division)
      fn mod_core(val_in: BigIntx2, mod_val: BigInt) -> BigInt {
        var rem: BigInt; // Remainder
        for(var i=0u; i<L; i++) { rem.limbs[i] = 0u; }

        // Iterate bits from MSB of 2L input (size L*2*32)
        for (var k = 0u; k < L*2; k++) {
           let limb_idx = (L*2) - 1u - k;
           let bit = (val_in.limbs[limb_idx] >> (31u - (k % 32u))) & 1u;
           
           // Shift Remainder Left 1 | bit
           var carry = bit;
           for (var i=0u; i<L; i++) {
             let v = rem.limbs[i];
             let n = (v << 1u) | carry;
             carry = v >> 31u;
             rem.limbs[i] = n;
           }

           // If Rem >= Mod, Rem -= Mod
           var gte = true;
           for(var j=0u; j<L; j++) {
             let i = L - 1u - j;
             if (rem.limbs[i] > mod_val.limbs[i]) { break; }
             if (rem.limbs[i] < mod_val.limbs[i]) { gte = false; break; }
           }

           if (gte) {
             var borrow = 0u;
             for (var i=0u; i<L; i++) {
               let a = rem.limbs[i];
               let b = mod_val.limbs[i];
               let diff = a - b - borrow;
               let next_borrow = u32(a < b) | (u32(a == b) & borrow);
               rem.limbs[i] = diff;
               borrow = next_borrow;
             }
           }
        }
        return rem;
      }

      // --- KERNELS ---

      @compute @workgroup_size(64)
      fn main_add(@builtin(global_invocation_id) id : vec3<u32>) {
        let idx = id.x; if (idx >= arrayLength(&bufA.values)) { return; }
        var carry = 0u;
        for (var i=0u; i<L; i++) {
          let a = bufA.values[idx].limbs[i];
          let sum = a + bufB.values[idx].limbs[i] + carry;
          carry = u32(sum < a || (carry == 1u && sum == a));
          bufR.values[idx].limbs[i] = sum;
        }
        for (var i=L; i<L*2; i++) { bufR.values[idx].limbs[i] = 0u; }
      }

      @compute @workgroup_size(64)
      fn main_mul(@builtin(global_invocation_id) id : vec3<u32>) {
        let idx = id.x; if (idx >= arrayLength(&bufA.values)) { return; }
        bufR.values[idx] = mul_core(bufA.values[idx], bufB.values[idx]);
      }

      @compute @workgroup_size(64)
      fn main_mod(@builtin(global_invocation_id) id : vec3<u32>) {
        let idx = id.x; if (idx >= arrayLength(&bufA.values)) { return; }
        // Prepare 2L input from 1L source (zero pad high)
        var wide_a: BigIntx2;
        for(var i=0u; i<L; i++) { wide_a.limbs[i] = bufA.values[idx].limbs[i]; }
        for(var i=L; i<L*2; i++) { wide_a.limbs[i] = 0u; }
        
        let r = mod_core(wide_a, bufB.values[idx]);
        // Write output (padded to 2L)
        for(var i=0u; i<L; i++) { bufR.values[idx].limbs[i] = r.limbs[i]; }
        for(var i=L; i<L*2; i++) { bufR.values[idx].limbs[i] = 0u; }
      }

      @compute @workgroup_size(64)
      fn main_modpow(@builtin(global_invocation_id) id : vec3<u32>) {
        let idx = id.x; if (idx >= arrayLength(&bufA.values)) { return; }
        
        let base_val = bufA.values[idx];
        let exp_val = bufB.values[idx];
        let mod_val = bufM.values[idx];

        // Result = 1
        var res: BigInt;
        res.limbs[0] = 1u;
        for(var i=1u; i<L; i++) { res.limbs[i] = 0u; }
        
        var base_curr = base_val;

        // Iterate Exponent Bits (LSB to MSB)
        for (var i = 0u; i < L; i++) {
          let limb = exp_val.limbs[i];
          for (var b = 0u; b < 32u; b++) {
            if (((limb >> b) & 1u) == 1u) {
              // res = (res * base) % mod
              res = mod_core(mul_core(res, base_curr), mod_val);
            }
            // base = (base * base) % mod
            base_curr = mod_core(mul_core(base_curr, base_curr), mod_val);
          }
        }

        for(var i=0u; i<L; i++) { bufR.values[idx].limbs[i] = res.limbs[i]; }
        for(var i=L; i<L*2; i++) { bufR.values[idx].limbs[i] = 0u; }
      }
    `;

    const module = this.device.createShaderModule({ code: shaderCode });
    const makePipe = (entry) => this.device.createComputePipeline({
      layout: 'auto', compute: { module, entryPoint: entry }
    });

    this.pipelines = {
      add: makePipe('main_add'),
      mul: makePipe('main_mul'),
      mod: makePipe('main_mod'),
      modPow: makePipe('main_modpow')
    };
  }

  async add(listA, listB) { return this._runOp(listA, listB, null, 'add'); }
  async mul(listA, listB) { return this._runOp(listA, listB, null, 'mul'); }
  async mod(listA, listB) { return this._runOp(listA, listB, null, 'mod'); }
  
  // New: modPow (A^B % M)
  async modPow(listBase, listExp, listMod) { 
    return this._runOp(listBase, listExp, listMod, 'modPow'); 
  }

  async _runOp(listA, listB, listM, opName) {
    const count = listA.length;
    const bA = this.toBuffer(listA, this.L);
    const bB = this.toBuffer(listB, this.L);
    // If M is provided, use it, otherwise dummy buffer (for bind group validation)
    const bM = listM ? this.toBuffer(listM, this.L) : new Uint32Array(64); 

    const outSize = count * (this.L * 2) * 4;

    const gA = this._buf(bA, GPUBufferUsage.STORAGE);
    const gB = this._buf(bB, GPUBufferUsage.STORAGE);
    const gM = this._buf(bM, GPUBufferUsage.STORAGE);
    const gR = this._buf(outSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const gRead = this._buf(outSize, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);

    const pipeline = this.pipelines[opName];
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: gA } },
        { binding: 1, resource: { buffer: gB } },
        { binding: 2, resource: { buffer: gR } },
        { binding: 3, resource: { buffer: gM } }
      ]
    });

    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(count / 64));
    pass.end();

    enc.copyBufferToBuffer(gR, 0, gRead, 0, outSize);
    this.device.queue.submit([enc.finish()]);

    await gRead.mapAsync(GPUMapMode.READ);
    const res = this.fromBuffer(new Uint32Array(gRead.getMappedRange()));
    gRead.unmap();
    return res;
  }

  _buf(d, u) {
    const size = d.byteLength || d;
    const b = this.device.createBuffer({ size, usage: u, mappedAtCreation: !!d.byteLength });
    if (d.byteLength) { new Uint32Array(b.getMappedRange()).set(d); b.unmap(); }
    return b;
  }

  toBuffer(bigInts, L) {
    const arr = new Uint32Array(bigInts.length * L);
    bigInts.forEach((n, i) => {
      let v = n;
      for (let j=0; j<L; j++) { arr[i*L+j] = Number(v & 0xFFFFFFFFn); v >>= 32n; }
    });
    return arr;
  }

  fromBuffer(arr) {
    const res = [];
    const L = this.L * 2; // Output is always 2x Width
    for (let i = 0; i < arr.length / L; i++) {
      let n = 0n;
      for (let j = L - 1; j >= 0; j--) { n = (n << 32n) | BigInt(arr[i * L + j]); }
      res.push(n);
    }
    return res;
  }
}