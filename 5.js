class GPUBigNum {
  constructor(limbCount = 64) {
    this.L = limbCount; // Input size (e.g., 64 limbs = 2048 bits)
    this.device = null;
    this.pipelines = {};
  }

  async init() {
    if (!navigator.gpu) throw new Error("WebGPU not supported");
    this.device = await (await navigator.gpu.requestAdapter()).requestDevice();

    const shaderCode = `
      // Config: Input Limbs (L), Output Limbs for Mul (2*L)
      override L_IN: u32 = ${this.L}u; 
      
      struct BigIntIn { limbs: array<u32, ${this.L}> }
      struct BigIntOut { limbs: array<u32, ${this.L * 2}> } // Support double width result
      
      struct DataIn { values: array<BigIntIn> }
      struct DataOut { values: array<BigIntOut> }
      
      @group(0) @binding(0) var<storage, read> bufA : DataIn;
      @group(0) @binding(1) var<storage, read> bufB : DataIn;
      @group(0) @binding(2) var<storage, read_write> bufR : DataOut;

      // --- HELPER: 32x32 -> 64-bit Multiply Accumulate ---
      // Computes: (a * b) + c + carry_in
      // Returns: vec2(lo, hi)
      fn mac(a: u32, b: u32, c: u32, carry_in: u32) -> vec2<u32> {
        let lo_a = a & 0xFFFFu; let hi_a = a >> 16u;
        let lo_b = b & 0xFFFFu; let hi_b = b >> 16u;
        
        let p0 = lo_a * lo_b;
        let p1 = lo_a * hi_b;
        let p2 = hi_a * lo_b;
        let p3 = hi_a * hi_b;

        let t = p1 + (p0 >> 16u);
        let p0_final = (t << 16u) | (p0 & 0xFFFFu);
        
        // Accumulate the initial values c and carry_in
        var lo = p0_final;
        var hi = p3 + (t >> 16u) + (p2 >> 16u);
        
        // Add p2 lower 16
        let t2 = lo + (p2 << 16u);
        if (t2 < lo) { hi++; } // Carry from add
        lo = t2;

        // Add c
        let t3 = lo + c;
        if (t3 < lo) { hi++; }
        lo = t3;
        
        // Add carry_in
        let t4 = lo + carry_in;
        if (t4 < lo) { hi++; }
        lo = t4;

        return vec2<u32>(lo, hi);
      }

      // --- ADDITION (Preserves logic from v2, mapped to larger output) ---
      @compute @workgroup_size(64)
      fn main_add(@builtin(global_invocation_id) id : vec3<u32>) {
        let idx = id.x;
        if (idx >= arrayLength(&bufA.values)) { return; }
        var carry = 0u;
        for (var i = 0u; i < L_IN; i++) {
          let a = bufA.values[idx].limbs[i];
          let b = bufB.values[idx].limbs[i];
          let sum = a + b + carry;
          carry = u32(sum < a || (carry == 1u && sum == a));
          bufR.values[idx].limbs[i] = sum;
        }
        // Zero out remaining high limbs (for consistency with mul output format)
        for (var i = L_IN; i < L_IN*2; i++) { bufR.values[idx].limbs[i] = 0u; }
      }

      // --- SUBTRACTION ---
      @compute @workgroup_size(64)
      fn main_sub(@builtin(global_invocation_id) id : vec3<u32>) {
        let idx = id.x;
        if (idx >= arrayLength(&bufA.values)) { return; }
        var borrow = 0u;
        for (var i = 0u; i < L_IN; i++) {
          let a = bufA.values[idx].limbs[i];
          let b = bufB.values[idx].limbs[i];
          let diff = a - b - borrow;
          let next_borrow = u32(a < b) | (u32(a == b) & borrow);
          bufR.values[idx].limbs[i] = diff;
          borrow = next_borrow;
        }
        for (var i = L_IN; i < L_IN*2; i++) { bufR.values[idx].limbs[i] = 0u; }
      }

      // --- MULTIPLICATION (O(N^2)) ---
      @compute @workgroup_size(64)
      fn main_mul(@builtin(global_invocation_id) id : vec3<u32>) {
        let idx = id.x;
        if (idx >= arrayLength(&bufA.values)) { return; }

        // Initialize result to 0
        for (var k = 0u; k < L_IN * 2; k++) { bufR.values[idx].limbs[k] = 0u; }

        // Schoolbook Multiplication: A[i] * B[j] -> R[i+j]
        for (var i = 0u; i < L_IN; i++) {
          var carry = 0u;
          let a_val = bufA.values[idx].limbs[i];
          
          for (var j = 0u; j < L_IN; j++) {
            let b_val = bufB.values[idx].limbs[j];
            let res_idx = i + j;
            
            // Current value at R[i+j]
            let current_res = bufR.values[idx].limbs[res_idx];
            
            // Calculate: (A[i] * B[j]) + current_res + carry
            let result = mac(a_val, b_val, current_res, carry);
            
            bufR.values[idx].limbs[res_idx] = result.x; // Low 32
            carry = result.y; // High 32 (New Carry)
          }
          // Propagate final carry for this row
          bufR.values[idx].limbs[i + L_IN] = carry;
        }
      }
    `;

    const module = this.device.createShaderModule({ code: shaderCode });
    const makePipe = (entry) => this.device.createComputePipeline({
      layout: 'auto', compute: { module, entryPoint: entry }
    });

    this.pipelines = {
      add: makePipe('main_add'),
      sub: makePipe('main_sub'),
      mul: makePipe('main_mul')
    };
  }

  async add(listA, listB) { return this._runOp(listA, listB, 'add'); }
  async sub(listA, listB) { return this._runOp(listA, listB, 'sub'); }
  async mul(listA, listB) { return this._runOp(listA, listB, 'mul'); }

  async _runOp(listA, listB, opName) {
    const count = listA.length;
    const arrayA = this.toBuffer(listA, this.L);
    const arrayB = this.toBuffer(listB, this.L);
    
    // Output is always 2*L size in memory (Add/Sub use half, Mul uses all)
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