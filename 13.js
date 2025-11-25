class GPUBigNum {
  constructor(limbCount = 64) {
    this.L = limbCount; // 64 limbs = 2048-bit
    this.device = null;
    this.pipelines = {};
  }

  async init() {
    if (!navigator.gpu) throw new Error("WebGPU not supported");
    this.device = await (await navigator.gpu.requestAdapter()).requestDevice();

    const shaderCode = `
      override L: u32 = ${this.L}u; 
      
      struct BigInt   { limbs: array<u32, L> }     
      struct BigIntx2 { limbs: array<u32, L * 2> } 
      
      struct DataIn   { values: array<BigInt> }
      struct DataOut  { values: array<BigIntx2> } 
      
      @group(0) @binding(0) var<storage, read> bufA : DataIn;
      @group(0) @binding(1) var<storage, read> bufB : DataIn;
      @group(0) @binding(2) var<storage, read_write> bufR : DataOut;
      @group(0) @binding(3) var<storage, read> bufM : DataIn;

      // --- LOCAL MATH HELPERS (Register-based) ---

      // Shift Left 1: A <<= 1
      fn shl1(a: ptr<function, BigInt>) {
        var carry = 0u;
        for (var i = 0u; i < L; i++) {
          let val = (*a).limbs[i];
          let next = (val << 1u) | carry;
          carry = val >> 31u;
          (*a).limbs[i] = next;
        }
      }

      // Shift Left 2: A <<= 2
      fn shl2(a: ptr<function, BigInt>) {
        var carry = 0u;
        for (var i = 0u; i < L; i++) {
          let val = (*a).limbs[i];
          let next = (val << 2u) | carry;
          carry = val >> 30u;
          (*a).limbs[i] = next;
        }
      }

      // Greater Than or Equal (Local vs Local): A >= B
      fn gte(a: ptr<function, BigInt>, b: ptr<function, BigInt>) -> bool {
        for (var k = 0u; k < L; k++) {
          let i = L - 1u - k;
          let va = (*a).limbs[i]; let vb = (*b).limbs[i];
          if (va > vb) { return true; }
          if (va < vb) { return false; }
        }
        return true;
      }

      // Subtract (Local vs Local): A -= B
      fn sub(a: ptr<function, BigInt>, b: ptr<function, BigInt>) {
        var borrow = 0u;
        for (var i = 0u; i < L; i++) {
          let va = (*a).limbs[i]; let vb = (*b).limbs[i];
          let diff = va - vb - borrow;
          let next_borrow = u32(va < vb) | (u32(va == vb) & borrow);
          (*a).limbs[i] = diff;
          borrow = next_borrow;
        }
      }

      // Helper: 32x32 -> 64 mul
      fn mac(a: u32, b: u32, c: u32, carry: u32) -> vec2<u32> {
        let lo_a=a&0xFFFFu; let hi_a=a>>16u; let lo_b=b&0xFFFFu; let hi_b=b>>16u;
        let p0=lo_a*lo_b; let p1=lo_a*hi_b; let p2=hi_a*lo_b; let p3=hi_a*hi_b;
        let t=p1+(p0>>16u);
        var lo=((t<<16u)|(p0&0xFFFFu)); var hi=p3+(t>>16u)+(p2>>16u);
        let t2=lo+(p2<<16u); if(t2<lo){hi++;} lo=t2;
        let t3=lo+c; if(t3<lo){hi++;} lo=t3;
        let t4=lo+carry; if(t4<lo){hi++;} lo=t4;
        return vec2<u32>(lo, hi);
      }

      // --- KERNELS ---

      @compute @workgroup_size(64)
      fn main_mul(@builtin(global_invocation_id) id : vec3<u32>) {
        let idx = id.x; if (idx >= arrayLength(&bufA.values)) { return; }
        var res: BigIntx2;
        for (var k=0u; k<L*2; k++) { res.limbs[k] = 0u; }
        for (var i=0u; i<L; i++) {
          var carry = 0u;
          let va = bufA.values[idx].limbs[i];
          for (var j=0u; j<L; j++) {
            let r = mac(va, bufB.values[idx].limbs[j], res.limbs[i+j], carry);
            res.limbs[i+j] = r.x; carry = r.y;
          }
          res.limbs[i+L] = carry;
        }
        bufR.values[idx] = res;
      }

      @compute @workgroup_size(64)
      fn main_sqrt(@builtin(global_invocation_id) id : vec3<u32>) {
        let idx = id.x; if (idx >= arrayLength(&bufA.values)) { return; }
        
        // Integer Square Root Algorithm (Binary Restoration)
        var rem: BigInt;  // Running Remainder
        var root: BigInt; // Accumulating Root
        // Init logic 0
        for(var i=0u; i<L; i++){ rem.limbs[i]=0u; root.limbs[i]=0u; }

        // Iterate 2 bits at a time, from MSB to LSB
        for (var k = 0u; k < L * 16u; k++) {
          let pair_idx = (L * 16u) - 1u - k; // Index of bit pair
          
          // 1. root <<= 1
          shl1(&root);
          
          // 2. rem <<= 2
          shl2(&rem);
          
          // 3. Bring down next 2 bits from input
          let glob_bit = pair_idx * 2u;
          let val = (bufA.values[idx].limbs[glob_bit / 32u] >> (glob_bit % 32u)) & 3u;
          rem.limbs[0] |= val;

          // 4. Test Candidate: (root << 1) | 1;
          // Since we already did root << 1, we just test (root | 1)
          var cand = root;
          cand.limbs[0] |= 1u;

          if (gte(&rem, &cand)) {
             sub(&rem, &cand);
             root.limbs[0] |= 1u; // Result bit is 1
          }
        }
        
        // Write result (padded)
        for(var i=0u; i<L; i++) { bufR.values[idx].limbs[i] = root.limbs[i]; }
        for(var i=L; i<L*2; i++) { bufR.values[idx].limbs[i] = 0u; }
      }
      
      // (Simplified placeholders for add/sub/modPow to save space in this view)
      @compute @workgroup_size(64) fn main_add(@builtin(global_invocation_id) id: vec3<u32>) { /*...*/ }
    `;
    
    // Compile & Cache
    const module = this.device.createShaderModule({ code: shaderCode });
    const pipe = (ep) => this.device.createComputePipeline({ layout:'auto', compute:{ module, entryPoint:ep }});
    
    this.pipelines = { 
      mul: pipe('main_mul'), 
      sqrt: pipe('main_sqrt') 
      // Add others here...
    };
  }

  async mul(listA, listB) { return this._op(listA, listB, null, 'mul'); }
  async sqrt(listA) { return this._op(listA, listA, null, 'sqrt'); } // B is ignored

  async _op(listA, listB, listM, op) {
    const count = listA.length;
    const bA = this._buf(this._toArr(listA), GPUBufferUsage.STORAGE);
    const bB = this._buf(this._toArr(listB), GPUBufferUsage.STORAGE);
    const bR = this._buf(count*this.L*8, GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
    const bRead = this._buf(count*this.L*8, GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
    
    // Dummy buffer for binding 3 if not used
    const bM = listM ? this._buf(this._toArr(listM), GPUBufferUsage.STORAGE) : this._buf(64, GPUBufferUsage.STORAGE);

    const bg = this.device.createBindGroup({
      layout: this.pipelines[op].getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bA } },
        { binding: 1, resource: { buffer: bB } },
        { binding: 2, resource: { buffer: bR } },
        { binding: 3, resource: { buffer: bM } }
      ]
    });

    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipelines[op]);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(count/64));
    pass.end();
    enc.copyBufferToBuffer(bR, 0, bRead, 0, count*this.L*8);
    this.device.queue.submit([enc.finish()]);

    await bRead.mapAsync(GPUMapMode.READ);
    const res = this._fromArr(new Uint32Array(bRead.getMappedRange()));
    bRead.unmap();
    return res;
  }

  _buf(d, u) {
    const s = d.byteLength||d; const b=this.device.createBuffer({size:s, usage:u, mappedAtCreation:!!d.byteLength});
    if(d.byteLength){new Uint32Array(b.getMappedRange()).set(d);b.unmap();} return b;
  }
  _toArr(l) { const a=new Uint32Array(l.length*this.L); l.forEach((n,i)=>{let v=n;for(let j=0;j<this.L;j++){a[i*this.L+j]=Number(v&0xFFFFFFFFn);v>>=32n}}); return a; }
  _fromArr(a) { const r=[]; const L=this.L*2; for(let i=0;i<a.length/L;i++){let n=0n;for(let j=L-1;j>=0;j--)n=(n<<32n)|BigInt(a[i*L+j]);r.push(n)} return r; }
}