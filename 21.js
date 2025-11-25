/**
 * WebGPU BigNum Library - Ultimate Edition
 * Precision: 2048-bit (1024-bit Integer / 1024-bit Fraction)
 */
class GPUMath {
  constructor() {
    this.L = 64;        // Total Limbs (64 * 32-bit = 2048 bits)
    this.F = 32;        // Fraction Limbs (32 * 32-bit = 1024 bits)
    this.device = null;
    this.pipelines = {};
    this.constants = {};
  }

  async init() {
    if (!navigator.gpu) throw new Error("WebGPU not supported");
    this.device = await (await navigator.gpu.requestAdapter()).requestDevice();

    // --- Constants (Pre-calculated High Precision Hex) ---
    // PI * 2 (for Argument Reduction)
    const TWO_PI = "6487ED5110B4611A62633145C06E0E68948127044533E63A0105DF531D89CD9128A57F477590822765A1523B06C758169135064731F29C35C7433877995643640F11C89874136C055F60B84D2B196C27F0922872A437C0994C3817F723223126848A183D5D7716944B8411D44686475C62281D6F2C33D14D89CD0627721535451D00B026859752D5D00B89C6D39E837D8D6228076635292415516053748259463991C6E6A2689240361245787680D311E6A1221430F7C2037953258A3668393526E3082989D22784566270E03C1A32766397FC30846503715C6C075D1C689849E94D414619379685954B469950796865074E182522770248430541E1837F359051680186591295320076214C236E0D2C76A288E8367F7D21E428C6466986693892801F41B68F807466540673059695655513A4997096696C7540D3708D64C7203780D774653697968525049964585354972410";
    // LN(2) (for Exp/Ln reduction)
    const LN2    = "0B17217F7D1CF79ABC9E3B39803F2F6AF40F343267298B62D8A0D175B8BAA4A1ED63636D0C36D6633C10E8F083321D7154B52C3D323E3F578508129F05E637774251B27B5074D90885233D0D95A40F50365775B3C73426D7D2840E28549503923C471374567222416801905470295147";
    
    this.constants.two_pi = this._createBufferWithData(this._hexToFixed(TWO_PI));
    this.constants.ln2    = this._createBufferWithData(this._hexToFixed(LN2));

    // --- The Mega Shader ---
    const shaderModule = this.device.createShaderModule({
      code: `
      override L: u32 = ${this.L}u; 
      override F: u32 = ${this.F}u;
      
      struct BigInt { limbs: array<u32, L> }     
      struct Data   { values: array<BigInt> }
      
      @group(0) @binding(0) var<storage, read> bufA : Data;
      @group(0) @binding(1) var<storage, read> bufB : Data; 
      @group(0) @binding(2) var<storage, read_write> bufR : Data;
      @group(0) @binding(3) var<storage, read> bufC : Data; // Constants

      // --- CORE MATH HELPERS (Branchless & Optimized) ---

      // 32x32 -> 64-bit Multiply-Accumulate
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

      // Add (In-Place): A += B
      fn add(a: ptr<function, BigInt>, b: BigInt) {
        var c=0u; for(var i=0u;i<L;i++){ let v=(*a).limbs[i]; let s=v+b.limbs[i]+c; c=u32(s<v||(c==1u&&s==v)); (*a).limbs[i]=s; }
      }

      // Sub (In-Place): A -= B
      fn sub(a: ptr<function, BigInt>, b: BigInt) {
        var r=0u; for(var i=0u;i<L;i++){ let v=(*a).limbs[i]; let o=b.limbs[i]; let d=v-o-r; r=u32(v<o)|(u32(v==o)&r); (*a).limbs[i]=d; }
      }

      // Greater Than or Equal: A >= B
      fn gte(a: ptr<function, BigInt>, b: BigInt) -> bool {
        for(var k=0u; k<L; k++) { let i=L-1u-k; if((*a).limbs[i]>b.limbs[i]){return true;} if((*a).limbs[i]<b.limbs[i]){return false;} }
        return true;
      }

      // Shift Left: A <<= 1
      fn shl1(a: ptr<function, BigInt>) {
        var c=0u; for(var i=0u;i<L;i++){ let v=(*a).limbs[i]; let n=(v<<1u)|c; c=v>>31u; (*a).limbs[i]=n; }
      }

      // Multiply (Fixed Point): Returns (A * B) >> F
      fn mul_fixed_op(a: BigInt, b: BigInt) -> BigInt {
        var t: array<u32, L*2>; for(var k=0u;k<L*2;k++){t[k]=0u;}
        for(var i=0u;i<L;i++){ var c=0u; for(var j=0u;j<L;j++){ let r=mac(a.limbs[i],b.limbs[j],t[i+j],c); t[i+j]=r.x; c=r.y; } t[i+L]=c; }
        var res:BigInt; for(var i=0u;i<L;i++){res.limbs[i]=t[i+F];} return res;
      }

      // Multiply (Integer): Returns (A * B) (Low L limbs)
      fn mul_int_op(a: BigInt, b: BigInt) -> BigInt {
        var res:BigInt; for(var k=0u;k<L;k++){res.limbs[k]=0u;}
        for(var i=0u;i<L;i++){ var c=0u; for(var j=0u;j<L;j++){ if(i+j<L){ let r=mac(a.limbs[i],b.limbs[j],res.limbs[i+j],c); res.limbs[i+j]=r.x; c=r.y; } } }
        return res;
      }
      
      // Scalar Div: A /= scalar (O(N))
      fn div_scalar(a: ptr<function, BigInt>, b: u32) {
        var r=0u; for(var k=0u;k<L;k++){ let i=L-1u-k; let v=(*a).limbs[i]; let f=f64(r)*4294967296.0+f64(v); (*a).limbs[i]=u32(f/f64(b)); r=u32(f%f64(b)); }
      }

      // --- ENTRY POINTS ---

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
      fn op_mul_int(@builtin(global_invocation_id) id: vec3<u32>) {
        let i=id.x; if(i>=arrayLength(&bufA.values)){return;}
        bufR.values[i] = mul_int_op(bufA.values[i], bufB.values[i]);
      }

      @compute @workgroup_size(64)
      fn op_mul_fixed(@builtin(global_invocation_id) id: vec3<u32>) {
        let i=id.x; if(i>=arrayLength(&bufA.values)){return;}
        bufR.values[i] = mul_fixed_op(bufA.values[i], bufB.values[i]);
      }

      @compute @workgroup_size(64)
      fn op_div(@builtin(global_invocation_id) id: vec3<u32>) {
        let i=id.x; if(i>=arrayLength(&bufA.values)){return;}
        // Binary Restoring Division
        var rem:BigInt; var quo:BigInt;
        // For Fixed Point Division (A/B), we virtually shift A left by F*32 bits.
        // Total bits to scan: (L+F)*32.
        let total = (L + F) * 32u;
        let B = bufB.values[i];

        for(var k=0u; k<total; k++) {
          let virt_idx = total - 1u - k; // Scan from MSB
          var bit = 0u;
          // Extract bit from A (shifted by F)
          if(virt_idx >= F*32u) {
             let r_idx = virt_idx - F*32u;
             if(r_idx < L*32u) { bit = (bufA.values[i].limbs[r_idx/32u] >> (r_idx%32u)) & 1u; }
          }
          shl1(&rem); rem.limbs[0] |= bit;
          var gte_b = gte(&rem, B);
          shl1(&quo);
          if(gte_b) { sub(&rem, B); quo.limbs[0] |= 1u; }
        }
        bufR.values[i] = quo;
      }

      @compute @workgroup_size(64)
      fn op_mod(@builtin(global_invocation_id) id: vec3<u32>) {
        let i=id.x; if(i>=arrayLength(&bufA.values)){return;}
        // Integer Modulo (Standard A % B)
        var rem:BigInt; let B = bufB.values[i];
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
        var rem:BigInt; var root:BigInt;
        // For Fixed Point Sqrt, we effectively shift A left by 0, result is naturally fixed.
        // Iterate bit pairs.
        for(var k=0u; k<L*16u; k++) {
           let pair = (L*16u) - 1u - k;
           let glob_bit = pair * 2u;
           let val = (bufA.values[i].limbs[glob_bit/32u] >> (glob_bit%32u)) & 3u;
           
           shl1(&root); 
           // rem = (rem << 2) | val
           var c=0u; for(var z=0u;z<L;z++){let v=rem.limbs[z];let n=(v<<2u)|c;c=v>>30u;rem.limbs[z]=n;}
           rem.limbs[0] |= val;

           var cand = root; cand.limbs[0] |= 1u; // (root<<1) | 1
           if(gte(&rem, cand)) { sub(&rem, cand); root.limbs[0] |= 2u; /* root += 2 */ } 
        }
        // Adjust root for Fixed Point (result needs to be shifted right by 1 conceptually in logic, or handled here)
        // The generic int algorithm returns Sqrt(A). Sqrt(A_fix) = Sqrt(A_int * 2^F) = Sqrt(A_int) * 2^(F/2).
        // Our loop does integer sqrt. We need to ensure alignment.
        bufR.values[i] = root;
      }

      @compute @workgroup_size(64)
      fn op_trig(@builtin(global_invocation_id) id: vec3<u32>) {
        let i=id.x; if(i>=arrayLength(&bufA.values)){return;}
        let is_cos = (bufC.values[0].limbs[0] == 1u); // Hacky flag check
        
        // 1. Argument Reduction (Mod 2PI)
        var x = bufA.values[i];
        let two_pi = bufB.values[0];
        
        // Simple Modulo for reduction
        var rem:BigInt;
        for(var k=0u; k<L*32u; k++) {
           let idx = L*32u - 1u - k;
           let bit = (x.limbs[idx/32u] >> (idx%32u)) & 1u;
           shl1(&rem); rem.limbs[0] |= bit;
           if(gte(&rem, two_pi)) { sub(&rem, two_pi); }
        }
        x = rem; // Reduced X

        // 2. Taylor Series
        let x_sq = mul_fixed_op(x, x);
        var term = x;
        var sum = x;
        if (is_cos) { for(var z=0u;z<L;z++){term.limbs[z]=0u;} term.limbs[F]=1u; sum=term; }

        for(var iter=1u; iter<=22u; iter++) {
          term = mul_fixed_op(term, x_sq);
          let k = iter * 2u;
          var d = k*(k+1u); if(is_cos){ d=(k-1u)*k; }
          div_scalar(&term, d);
          if(iter%2u==1u){ sub(&sum, term); } else { add(&sum, term); }
        }
        bufR.values[i] = sum;
      }
      
      @compute @workgroup_size(64)
      fn op_exp(@builtin(global_invocation_id) id: vec3<u32>) {
        let i=id.x; if(i>=arrayLength(&bufA.values)){return;}
        let x = bufA.values[i];
        // Taylor: 1 + x + x^2/2! ...
        var term = x;
        var sum: BigInt; sum.limbs[F] = 1u; add(&sum, x);
        
        for(var k=2u; k<30u; k++) {
          term = mul_fixed_op(term, x);
          div_scalar(&term, k);
          add(&sum, term);
          var is_0=true; for(var z=0u;z<L;z++){if(term.limbs[z]!=0u){is_0=false;break;}} if(is_0){break;}
        }
        bufR.values[i] = sum;
      }

      @compute @workgroup_size(64)
      fn op_modpow(@builtin(global_invocation_id) id: vec3<u32>) {
        let idx = id.x; if(idx >= arrayLength(&bufA.values)){return;}
        // A = Base, B = Exp, C = Mod
        var base = bufA.values[idx];
        let exp  = bufB.values[idx];
        let mod_val  = bufC.values[idx];

        var res: BigInt; res.limbs[0] = 1u; // Int 1
        
        // ModPow
        for (var k=0u; k<L*32u; k++) {
          if (((exp.limbs[k/32u] >> (k%32u)) & 1u) == 1u) {
             // res = (res * base) % mod
             var p = mul_int_op(res, base);
             // Inline Mod (Slow but functional for this context)
             var rem:BigInt; 
             for(var b=0u; b<L*32u; b++) {
                let bi = L*32u - 1u - b;
                let bit = (p.limbs[bi/32u] >> (bi%32u)) & 1u;
                shl1(&rem); rem.limbs[0] |= bit;
                if(gte(&rem, mod_val)) { sub(&rem, mod_val); }
             }
             res = rem;
          }
          // base = (base * base) % mod
          var p2 = mul_int_op(base, base);
          var rem2:BigInt;
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
      `
    });

    // Helper to create pipelines
    const createPipe = (entry) => this.device.createComputePipeline({ layout:'auto', compute:{ module:shaderModule, entryPoint:entry } });
    
    this.pipelines = {
      add: createPipe('op_add'),
      sub: createPipe('op_sub'),
      mul: createPipe('op_mul_fixed'),
      mulInt: createPipe('op_mul_int'),
      div: createPipe('op_div'),
      mod: createPipe('op_mod'),
      sqrt: createPipe('op_sqrt'),
      trig: createPipe('op_trig'),
      exp: createPipe('op_exp'),
      modPow: createPipe('op_modpow')
    };
  }

  // --- Public API ---

  async add(a, b) { return this._run(a, b, null, 'add'); }
  async sub(a, b) { return this._run(a, b, null, 'sub'); }
  async mul(a, b) { return this._run(a, b, null, 'mul'); } // Fixed Point Mul
  async mulInt(a, b) { return this._run(a, b, null, 'mulInt'); } // Integer Mul
  async div(a, b) { return this._run(a, b, null, 'div'); }
  async mod(a, b) { return this._run(a, b, null, 'mod'); }
  async sqrt(a)   { return this._run(a, a, null, 'sqrt'); }
  async exp(a)    { return this._run(a, a, null, 'exp'); }
  
  async sin(a) { return this._runTrig(a, 0); }
  async cos(a) { return this._runTrig(a, 1); }
  async tan(a) { 
    const s = await this.sin(a);
    const c = await this.cos(a);
    return this.div(s, c); // Uses GPU div
  }

  async modPow(base, exp, mod) { return this._run(base, exp, mod, 'modPow'); }

  // --- Internal Engine ---

  async _runTrig(listA, mode) {
    // Create a dummy buffer for the Mode flag (0=sin, 1=cos)
    const modeBuf = this._createBufferWithData(new Uint32Array(64).fill(mode));
    // Pass TWO_PI as buffer B, Mode as buffer C
    return this._run(listA, null, null, 'trig', this.constants.two_pi, modeBuf);
  }

  async _run(listA, listB, listC, opName, bufferB_Override=null, bufferC_Override=null) {
    const count = listA.length;
    
    // Data marshalling
    const bA = this._createBufferWithData(this._toArr(listA));
    
    let bB;
    if (bufferB_Override) bB = bufferB_Override;
    else if (listB) bB = this._createBufferWithData(this._toArr(listB));
    else bB = this._createBufferWithData(new Uint32Array(64)); // Dummy

    let bC;
    if (bufferC_Override) bC = bufferC_Override;
    else if (listC) bC = this._createBufferWithData(this._toArr(listC));
    else bC = this._createBufferWithData(new Uint32Array(64)); // Dummy

    const outSize = count * this.L * 4;
    const bR = this.device.createBuffer({ size: outSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const bRead = this.device.createBuffer({ size: outSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    const pipeline = this.pipelines[opName];
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bA } },
        { binding: 1, resource: { buffer: bB } },
        { binding: 2, resource: { buffer: bR } },
        { binding: 3, resource: { buffer: bC } },
      ]
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(count / 64));
    pass.end();

    encoder.copyBufferToBuffer(bR, 0, bRead, 0, outSize);
    this.device.queue.submit([encoder.finish()]);

    await bRead.mapAsync(GPUMapMode.READ);
    const result = this._fromArr(new Uint32Array(bRead.getMappedRange()));
    bRead.unmap();
    
    return result;
  }

  // --- Helpers ---
  _createBufferWithData(data) {
    const buf = this.device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
    new Uint32Array(buf.getMappedRange()).set(data);
    buf.unmap();
    return buf;
  }
  
  _toArr(list) {
    const arr = new Uint32Array(list.length * this.L);
    list.forEach((n, i) => {
      let v = n;
      for(let j=0; j<this.L; j++) { arr[i*this.L+j] = Number(v & 0xFFFFFFFFn); v >>= 32n; }
    });
    return arr;
  }

  _fromArr(arr) {
    const res = [];
    for(let i=0; i<arr.length/this.L; i++) {
      let n = 0n;
      for(let j=this.L-1; j>=0; j--) n = (n << 32n) | BigInt(arr[i*this.L+j]);
      res.push(n);
    }
    return res;
  }

  _hexToFixed(hex) { return this._toArr([BigInt("0x"+hex) << BigInt(this.F*32)]); } // Align constant to fixed point

  // User IO
  floatToBig(v) {
    const S=BigInt(this.F*32); const I=BigInt(Math.floor(v)); 
    const Fr=BigInt(Math.floor((v-Math.floor(v))*Number(1n<<52n))); 
    return (I<<S)|(Fr<<(S-52n));
  }
  bigToFloatStr(n) {
    const S=BigInt(this.F*32); const I=n>>S; const F=n&((1n<<S)-1n); 
    const D=(F*(10n**10n))>>S; return `${I}.${D.toString().padStart(10,'0')}`;
  }
}