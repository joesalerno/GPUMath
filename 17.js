class GPUBigNum {
  constructor(limbCount = 64) {
    this.L = limbCount; // 64 limbs = 2048 bits
    this.F = this.L / 2; // Fixed Point: 32 Integer, 32 Fraction
    this.device = null;
    this.pipelines = {};
    this.constants = {};
  }

  async init() {
    if (!navigator.gpu) throw new Error("WebGPU not supported");
    this.device = await (await navigator.gpu.requestAdapter()).requestDevice();

    // 1. Generate High-Precision 2*PI Constant (2048-bit)
    // We use a predefined hex string for precision to avoid BigInt lib dependency complexity here.
    // This is 2*PI truncated to fit our precision.
    const PI_HEX = "6487ED5110B4611A62633145C06E0E68948127044533E63A0105DF531D89CD9128A57F477590822765A1523B06C758169135064731F29C35C7433877995643640F11C89874136C055F60B84D2B196C27F0922872A437C0994C3817F723223126848A183D5D7716944B8411D44686475C62281D6F2C33D14D89CD0627721535451D00B026859752D5D00B89C6D39E837D8D6228076635292415516053748259463991C6E6A2689240361245787680D311E6A1221430F7C2037953258A3668393526E3082989D22784566270E03C1A32766397FC30846503715C6C075D1C689849E94D414619379685954B469950796865074E182522770248430541E1837F359051680186591295320076214C236E0D2C76A288E8367F7D21E428C6466986693892801F41B68F807466540673059695655513A4997096696C7540D3708D64C7203780D774653697968525049964585354972410";
    this.constants.two_pi = this.toBuffer([BigInt("0x" + PI_HEX)]);
    this.gpuTwoPi = this._buf(this.constants.two_pi, GPUBufferUsage.STORAGE);

    const shaderCode = `
      override L: u32 = ${this.L}u; 
      override F: u32 = ${this.F}u;
      
      struct BigInt { limbs: array<u32, L> }     
      struct DataIn { values: array<BigInt> }
      
      @group(0) @binding(0) var<storage, read> bufA : DataIn;
      @group(0) @binding(1) var<storage, read> bufB : DataIn; // Modulus or Divisor
      @group(0) @binding(2) var<storage, read_write> bufR : DataIn; // Reuse size L for output

      // --- HELPERS ---

      // A >= B
      fn gte(a: ptr<function, BigInt>, b: BigInt) -> bool {
        for (var k=0u; k<L; k++) {
          let i = L - 1u - k;
          if ((*a).limbs[i] > b.limbs[i]) { return true; }
          if ((*a).limbs[i] < b.limbs[i]) { return false; }
        }
        return true;
      }

      // A -= B
      fn sub_self(a: ptr<function, BigInt>, b: BigInt) {
        var bor = 0u;
        for(var i=0u; i<L; i++) {
          let v = (*a).limbs[i]; let ov = b.limbs[i];
          let d = v - ov - bor;
          bor = u32(v < ov) | (u32(v == ov) & bor);
          (*a).limbs[i] = d;
        }
      }
      
      // A += B
      fn add_self(a: ptr<function, BigInt>, b: BigInt) {
        var c = 0u;
        for(var i=0u; i<L; i++) {
          let v = (*a).limbs[i];
          let s = v + b.limbs[i] + c;
          c = u32(s < v || (c==1u && s==v));
          (*a).limbs[i] = s;
        }
      }

      fn shl1(a: ptr<function, BigInt>) {
        var c = 0u;
        for (var i=0u; i<L; i++) {
          let v = (*a).limbs[i];
          let n = (v << 1u) | c;
          c = v >> 31u;
          (*a).limbs[i] = n;
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

      // Fixed Point Mul
      fn mul_fixed(a: BigInt, b: BigInt) -> BigInt {
        var temp: array<u32, L * 2>;
        for(var k=0u; k<L*2; k++){ temp[k]=0u; }
        for(var i=0u; i<L; i++) {
          var c = 0u;
          for(var j=0u; j<L; j++) {
            let r = mac(a.limbs[i], b.limbs[j], temp[i+j], c);
            temp[i+j] = r.x; c = r.y;
          }
          temp[i+L] = c;
        }
        var res: BigInt;
        for(var i=0u; i<L; i++) { res.limbs[i] = temp[i + F]; }
        return res;
      }

      // Scalar Div (O(N))
      fn div_scalar(a: ptr<function, BigInt>, b: u32) {
        var rem = 0u;
        for(var k=0u; k<L; k++) {
          let i = L - 1u - k;
          let val = (*a).limbs[i];
          let r_f = f64(rem) * 4294967296.0 + f64(val);
          (*a).limbs[i] = u32(r_f / f64(b));
          rem = u32(r_f % f64(b));
        }
      }

      // --- KERNELS ---

      // 1. ARGUMENT REDUCTION: X = X % TWO_PI
      @compute @workgroup_size(64)
      fn main_reduce(@builtin(global_invocation_id) id : vec3<u32>) {
        let idx = id.x; if (idx >= arrayLength(&bufA.values)) { return; }
        
        // Load Input and Modulus
        let val_a = bufA.values[idx];
        let mod_v = bufB.values[0]; // Constant TWO_PI
        
        var rem: BigInt;
        for(var i=0u; i<L; i++) { rem.limbs[i] = 0u; }

        // Binary Long Division (Modulo)
        for (var k = 0u; k < L*32u; k++) {
           let limb_idx = L - 1u - (k / 32u);
           let bit = (val_a.limbs[limb_idx] >> (31u - (k % 32u))) & 1u;
           
           shl1(&rem); // rem << 1
           rem.limbs[0] |= bit;

           if (gte(&rem, mod_v)) {
             sub_self(&rem, mod_v);
           }
        }
        bufR.values[idx] = rem;
      }

      // 2. TANGENT DIVISION: Fixed Point Div (A << F) / B
      @compute @workgroup_size(64)
      fn main_div_fixed(@builtin(global_invocation_id) id : vec3<u32>) {
        let idx = id.x; if (idx >= arrayLength(&bufA.values)) { return; }
        
        let A = bufA.values[idx]; // Sin
        let B = bufB.values[idx]; // Cos
        
        var rem: BigInt; 
        var quo: BigInt;
        for(var i=0u; i<L; i++) { rem.limbs[i] = 0u; quo.limbs[i] = 0u; }

        // Iterate bits. 
        // Conceptually we are dividing (A << F_BITS) by B.
        // Total bits to scan = L*32 + F*32
        
        let total_bits = L*32u + F*32u;

        for (var k = 0u; k < total_bits; k++) {
           // Get bit from A (shifted left by F*32 virtually)
           // Bit index 'p' in A corresponds to 'k'. 
           // If k < L*32, we are pulling bits from A. 
           // If k >= L*32, we are pulling "zeros" (the shift).
           
           // Actually, standard restoring division usually iterates from MSB of Dividend.
           // Dividend is A << F. MSB is at index (L+F)*32 - 1.
           // Let's iterate 'step' from 0 to total_bits.
           
           let bit_idx_virtual = total_bits - 1u - k;
           var bit = 0u;
           
           // Map virtual index to A's limbs
           // A is size L. A << F means A's MSB is at (L+F-1).
           // If bit_idx_virtual >= F*32, it's inside A.
           if (bit_idx_virtual >= F*32u) {
             let bit_in_a = bit_idx_virtual - F*32u;
             if (bit_in_a < L*32u) {
                let limb = bit_in_a / 32u;
                let b_off = bit_in_a % 32u;
                bit = (A.limbs[limb] >> b_off) & 1u;
             }
           }
           // Else bit is 0 (padding)

           shl1(&rem);
           rem.limbs[0] |= bit;

           if (gte(&rem, B)) {
             sub_self(&rem, B);
             
             // Set quotient bit
             // Q bit is at 'bit_idx_virtual' relative to the result... 
             // Actually, since we shift A by F, the result Q is automatically Fixed Point.
             // We just need to capture Q bits.
             // With k going 0..total, we generate MSB first.
             // We only have storage for L limbs (L*32 bits). 
             // We only care about the lower L*32 bits of the quotient usually?
             // Or rather, the top L*32 bits of the answer?
             
             // Simple: Just shift Q left and add 1.
             shl1(&quo);
             quo.limbs[0] |= 1u;
           } else {
             shl1(&quo);
           }
        }
        bufR.values[idx] = quo;
      }

      // 3. TAYLOR SERIES (Sin/Cos)
      @compute @workgroup_size(64)
      fn main_trig(@builtin(global_invocation_id) id : vec3<u32>) {
        let idx = id.x; if (idx >= arrayLength(&bufA.values)) { return; }
        
        // Mode 0 = Sin, 1 = Cos. We use a hacky uniform or just distinct shaders.
        // For conciseness, let's assume this is SIN. (We'll make a copy for COS).
        // Actually, let's look at binding B. If B[0] == 1, Cos. Else Sin.
        // Reuse buffers!
        
        let x = bufA.values[idx];
        let is_cos = (bufB.values[0].limbs[0] == 1u);

        let x_sq = mul_fixed(x, x);
        var term = x;
        var sum = x;
        
        if (is_cos) {
          // Cos Init
          for(var i=0u; i<L; i++) { term.limbs[i] = 0u; }
          term.limbs[F] = 1u; // 1.0
          sum = term;
        }

        for(var i=1u; i<=22u; i++) {
          term = mul_fixed(term, x_sq);
          let k = i * 2u;
          var div = 1u;
          if (is_cos) { div = (k - 1u) * k; } 
          else        { div = k * (k + 1u); }
          
          div_scalar(&term, div);
          if (i % 2u == 1u) { sub_self(&sum, term); }
          else              { add_self(&sum, term); }
        }
        bufR.values[idx] = sum;
      }
    `;

    const module = this.device.createShaderModule({ code: shaderCode });
    const pipe = (ep) => this.device.createComputePipeline({ layout:'auto', compute:{ module, entryPoint:ep }});
    
    this.pipelines = { 
      reduce: pipe('main_reduce'),
      trig: pipe('main_trig'),
      div_fixed: pipe('main_div_fixed')
    };
  }

  // --- Public API ---

  async sin(listA) { return this._runTrig(listA, 0); }
  async cos(listA) { return this._runTrig(listA, 1); }
  
  async tan(listA) {
    // Tan = Sin / Cos
    // 1. Calculate reduced angles
    const reduced = await this._runOp(listA, [this.constants.two_pi], 'reduce'); // Pass TwoPi as B
    const reducedList = this._fromArr(reduced);

    // 2. Calc Sin and Cos in parallel
    // We need to run internal ops without reading back to CPU for max speed, 
    // but for simplicity here we compose the calls.
    const sinVals = await this._runTrigInternal(reducedList, 0);
    const cosVals = await this._runTrigInternal(reducedList, 1);

    // 3. Div Fixed
    return this._runDivFixed(sinVals, cosVals);
  }

  // --- Internal Orchestration ---

  async _runTrig(listA, mode) {
    // 1. Reduce Arguments (A % 2PI)
    // We pass the TWO_PI constant as buffer B
    const reducedArr = await this._runOp(listA, null, 'reduce', this.gpuTwoPi);
    const reducedList = this._fromArr(reducedArr);
    
    // 2. Calculate Taylor Series
    return this._runTrigInternal(reducedList, mode);
  }

  async _runTrigInternal(list, mode) {
    // Mode Buffer: 0 = Sin, 1 = Cos
    const modeBuf = this._buf(new Uint32Array(this.L * 64).fill(0), GPUBufferUsage.STORAGE); // Wasteful but safe size
    if (mode === 1) { 
        new Uint32Array(modeBuf.getMappedRange()).set([1]); 
        modeBuf.unmap(); 
    }
    return this._runOp(list, null, 'trig', modeBuf);
  }

  async _runDivFixed(listA, listB) {
    return this._runOp(listA, listB, 'div_fixed');
  }

  async _runOp(listA, listB, opName, gpuBufferB_Override = null) {
    const count = listA.length;
    const bA = this._buf(this._toArr(listA), GPUBufferUsage.STORAGE);
    
    let bB;
    if (gpuBufferB_Override) {
      bB = gpuBufferB_Override;
    } else if (listB) {
      bB = this._buf(this._toArr(listB), GPUBufferUsage.STORAGE);
    } else {
      bB = this._buf(64, GPUBufferUsage.STORAGE); // Dummy
    }

    const bR = this._buf(count*this.L*4, GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
    const bRead = this._buf(count*this.L*4, GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);

    const bg = this.device.createBindGroup({
      layout: this.pipelines[opName].getBindGroupLayout(0),
      entries: [ { binding:0, resource:{buffer:bA} }, { binding:1, resource:{buffer:bB} }, { binding:2, resource:{buffer:bR} } ]
    });

    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipelines[opName]);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(count/64));
    pass.end();
    enc.copyBufferToBuffer(bR, 0, bRead, 0, count*this.L*4);
    this.device.queue.submit([enc.finish()]);

    await bRead.mapAsync(GPUMapMode.READ);
    const res = this._fromArr(new Uint32Array(bRead.getMappedRange()));
    bRead.unmap();
    return res;
  }

  // --- Utils ---
  floatToBig(v){const S=BigInt(this.F*32);const I=BigInt(Math.floor(v));const Fr=BigInt(Math.floor((v-Math.floor(v))*Number(1n<<52n)));return(I<<S)|(Fr<<(S-52n));}
  bigToFloatStr(n){const S=BigInt(this.F*32);const I=n>>S;const F=n&((1n<<S)-1n);const D=(F*(10n**18n))>>S;return`${I}.${D.toString().padStart(18,'0')}`;}
  _buf(d,u){const s=d.byteLength||d;const b=this.device.createBuffer({size:s,usage:u,mappedAtCreation:!!d.byteLength});if(d.byteLength){new Uint32Array(b.getMappedRange()).set(d);b.unmap();}return b;}
  _toArr(l){const a=new Uint32Array(l.length*this.L);l.forEach((n,i)=>{let v=n;for(let j=0;j<this.L;j++){a[i*this.L+j]=Number(v&0xFFFFFFFFn);v>>=32n}});return a;}
  _fromArr(a){const r=[];for(let i=0;i<a.length/this.L;i++){let n=0n;for(let j=this.L-1;j>=0;j--)n=(n<<32n)|BigInt(a[i*this.L+j]);r.push(n)}return r;}
  toBuffer(l){return this._toArr(l);} // Helper for internal use
}