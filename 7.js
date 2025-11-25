class GPUBigNum {
  constructor(limbCount = 64) {
    this.L = limbCount; // Input size (e.g. 64 = 2048 bits)
    this.device = null;
    this.pipelines = {};
  }

  async init() {
    if (!navigator.gpu) throw new Error("WebGPU not supported");
    this.device = await (await navigator.gpu.requestAdapter()).requestDevice();

    const shaderCode = `
      override L: u32 = ${this.L}u; 
      
      struct BigIntIn { limbs: array<u32, ${this.L}> }
      struct BigIntOut { limbs: array<u32, ${this.L * 2}> }
      
      struct DataIn { values: array<BigIntIn> }
      struct DataOut { values: array<BigIntOut> }
      
      @group(0) @binding(0) var<storage, read> bufA : DataIn;
      @group(0) @binding(1) var<storage, read> bufB : DataIn;
      @group(0) @binding(2) var<storage, read_write> bufR : DataOut;

      // --- UTILS ---
      
      // Compare: A >= B (where A is local, B is global)
      fn gte(a: ptr<function, array<u32, L>>, b_idx: u32) -> bool {
        for (var k = 0u; k < L; k++) {
          let i = L - 1u - k; // Iterate High to Low
          let val_a = (*a)[i];
          let val_b = bufB.values[b_idx].limbs[i];
          if (val_a > val_b) { return true; }
          if (val_a < val_b) { return false; }
        }
        return true;
      }

      // Subtract: A = A - B (In-place)
      fn sub_self(a: ptr<function, array<u32, L>>, b_idx: u32) {
        var borrow = 0u;
        for (var i = 0u; i < L; i++) {
          let val_a = (*a)[i];
          let val_b = bufB.values[b_idx].limbs[i];
          let diff = val_a - val_b - borrow;
          let next_borrow = u32(val_a < val_b) | (u32(val_a == val_b) & borrow);
          (*a)[i] = diff;
          borrow = next_borrow;
        }
      }

      // Shift Left 1: A = (A << 1) | bit
      fn shl1(a: ptr<function, array<u32, L>>, bit: u32) {
        var carry = bit;
        for (var i = 0u; i < L; i++) {
          let val = (*a)[i];
          let next = (val << 1u) | carry;
          carry = val >> 31u;
          (*a)[i] = next;
        }
      }

      // Multiply-Accumulate Helper for Mul
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

      // --- KERNELS ---

      @compute @workgroup_size(64)
      fn main_add(@builtin(global_invocation_id) id : vec3<u32>) {
        let idx = id.x;
        if (idx >= arrayLength(&bufA.values)) { return; }
        var carry = 0u;
        for (var i = 0u; i < L; i++) {
          let a = bufA.values[idx].limbs[i];
          let b = bufB.values[idx].limbs[i];
          let sum = a + b + carry;
          carry = u32(sum < a || (carry == 1u && sum == a));
          bufR.values[idx].limbs[i] = sum;
        }
        for (var i = L; i < L*2; i++) { bufR.values[idx].limbs[i] = 0u; }
      }

      @compute @workgroup_size(64)
      fn main_sub(@builtin(global_invocation_id) id : vec3<u32>) {
        let idx = id.x;
        if (idx >= arrayLength(&bufA.values)) { return; }
        var borrow = 0u;
        for (var i = 0u; i < L; i++) {
          let a = bufA.values[idx].limbs[i];
          let b = bufB.values[idx].limbs[i];
          let diff = a - b - borrow;
          let next_borrow = u32(a < b) | (u32(a == b) & borrow);
          bufR.values[idx].limbs[i] = diff;
          borrow = next_borrow;
        }
        for (var i = L; i < L*2; i++) { bufR.values[idx].limbs[i] = 0u; }
      }

      @compute @workgroup_size(64)
      fn main_mul(@builtin(global_invocation_id) id : vec3<u32>) {
        let idx = id.x;
        if (idx >= arrayLength(&bufA.values)) { return; }
        for (var k = 0u; k < L * 2; k++) { bufR.values[idx].limbs[k] = 0u; }
        for (var i = 0u; i < L; i++) {
          var carry = 0u;
          let a_val = bufA.values[idx].limbs[i];
          for (var j = 0u; j < L; j++) {
            let b_val = bufB.values[idx].limbs[j];
            let res = mac(a_val, b_val, bufR.values[idx].limbs[i+j], carry);
            bufR.values[idx].limbs[i+j] = res.x;
            carry = res.y;
          }
          bufR.values[idx].limbs[i + L] = carry;
        }
      }

      @compute @workgroup_size(64)
      fn main_div(@builtin(global_invocation_id) id : vec3<u32>) {
        let idx = id.x;
        if (idx >= arrayLength(&bufA.values)) { return; }

        // Initialize
        var rem : array<u32, L>; // Remainder register
        var quo : array<u32, L>; // Quotient register
        for(var i=0u; i<L; i++) { rem[i] = 0u; quo[i] = 0u; }

        // Bitwise Long Division (Slow but simple)
        // Iterate from MSB (bit L*32 - 1) down to 0
        for (var k = 0u; k < L; k++) {
           let limb_idx = L - 1u - k;
           let limb_val = bufA.values[idx].limbs[limb_idx];
           
           // Process 32 bits of this limb
           for (var b = 0u; b < 32u; b++) {
             let bit_pos = 31u - b;
             let bit = (limb_val >> bit_pos) & 1u;
             
             // R = (R << 1) | bit
             shl1(&rem, bit);
             
             // if (R >= B) { R -= B; Q |= 1 }
             if (gte(&rem, idx)) {
               sub_self(&rem, idx);
               // Set bit in quotient
               quo[limb_idx] |= (1u << bit_pos);
             }
           }
        }

        // Write Result
        for(var i=0u; i<L; i++) { bufR.values[idx].limbs[i] = quo[i]; }
        for(var i=L; i<L*2; i++) { bufR.values[idx].limbs[i] = 0u; }
      }
    `;

    const module = this.device.createShaderModule({ code: shaderCode });
    const makePipe = (entry) => this.device.createComputePipeline({
      layout: 'auto', compute: { module, entryPoint: entry }
    });

    this.pipelines = {
      add: makePipe('main_add'),
      sub: makePipe('main_sub'),
      mul: makePipe('main_mul'),
      div: makePipe('main_div')
    };
  }

  async add(listA, listB) { return this._runOp(listA, listB, 'add'); }
  async sub(listA, listB) { return this._runOp(listA, listB, 'sub'); }
  async mul(listA, listB) { return this._runOp(listA, listB, 'mul'); }
  async div(listA, listB) { return this._runOp(listA, listB, 'div'); }

  async _runOp(listA, listB, opName) {
    const count = listA.length;
    const arrayA = this.toBuffer(listA, this.L);
    const arrayB = this.toBuffer(listB, this.L);
    const outLimbCount = this.L * 2;
    const bufSizeOut = count * outLimbCount * 4;

    const gpuA = this._createBuf(arrayA, GPUBufferUsage.STORAGE);
    const gpuB = this._createBuf(arrayB, GPUBufferUsage.STORAGE);
    const gpuR = this._createBuf(bufSizeOut, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const gpuRead = this._createBuf(bufSizeOut, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);

    const pipeline = this.pipelines[opName];
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: gpuA } },
        { binding: 1, resource: { buffer: gpuB } },
        { binding: 2, resource: { buffer: gpuR } }
      ]
    });

    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(count / 64));
    pass.end();

    enc.copyBufferToBuffer(gpuR, 0, gpuRead, 0, bufSizeOut);
    this.device.queue.submit([enc.finish()]);

    await gpuRead.mapAsync(GPUMapMode.READ);
    const result = this.fromBuffer(new Uint32Array(gpuRead.getMappedRange()), outLimbCount);
    gpuRead.unmap();
    return result;
  }

  _createBuf(data, usage) {
    const size = typeof data === 'number' ? data : data.byteLength;
    const buf = this.device.createBuffer({ size, usage, mappedAtCreation: typeof data !== 'number' });
    if (typeof data !== 'number') { new Uint32Array(buf.getMappedRange()).set(data); buf.unmap(); }
    return buf;
  }

  toBuffer(bigInts, limbCount) {
    const arr = new Uint32Array(bigInts.length * limbCount);
    bigInts.forEach((bn, i) => {
      let n = bn;
      for (let j = 0; j < limbCount; j++) {
        arr[i * limbCount + j] = Number(n & 0xFFFFFFFFn);
        n >>= 32n;
      }
    });
    return arr;
  }

  fromBuffer(arr, limbCount) {
    const res = [];
    for (let i = 0; i < arr.length / limbCount; i++) {
      let n = 0n;
      for (let j = limbCount - 1; j >= 0; j--) {
        n = (n << 32n) | BigInt(arr[i * limbCount + j]);
      }
      res.push(n);
    }
    return res;
  }
}