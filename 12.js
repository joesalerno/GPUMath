(async () => {
  console.log("⚡ Initializing GPU BigNum (v6) with MOD POW...");
  const math = new GPUBigNum(64); // 2048-bit
  await math.init();

  // --- RSA SIMULATION ---
  const COUNT = 100; // Number of parallel decryptions
  console.log(`🔐 Simulating ${COUNT} parallel 2048-bit RSA operations...`);

  const bases = [];
  const exponents = [];
  const moduli = [];

  // Generate random test data
  // Formula: Result = (Base ^ Exp) % Mod
  for(let i=0; i<COUNT; i++) {
    // Base (Message)
    bases.push(BigInt("0x" + Array(100).fill("C").join(""))); 
    // Exponent (Private Key - typically huge)
    exponents.push(BigInt("0x10001")); 
    // Modulus (Public Key)
    moduli.push(BigInt("0x" + Array(110).fill("F").join(""))); 
  }

  // 1. Run on GPU
  console.log("🚀 Running modPow...");
  const start = performance.now();
  
  // The magic happens here
  const results = await math.modPow(bases, exponents, moduli);
  
  const duration = performance.now() - start;
  console.log(`   Completed in ${duration.toFixed(2)}ms`);
  console.log(`   Throughput: ${(COUNT / (duration/1000)).toFixed(1)} RSA/sec`);

  // 2. Verify one result using native BigInt (slow on CPU)
  console.log("\n🔎 Verification (CPU vs GPU):");
  const idx = 0;
  
  // JS BigInt native modPow wrapper
  const nativeModPow = (b, e, m) => {
    let r = 1n;
    b = b % m;
    while (e > 0n) {
      if (e % 2n === 1n) r = (r * b) % m;
      b = (b * b) % m;
      e /= 2n;
    }
    return r;
  };

  const expected = nativeModPow(bases[idx], exponents[idx], moduli[idx]);
  const actual = results[idx];

  console.log(`   Base: ${bases[idx].toString().slice(0,10)}...`);
  console.log(`   Exp:  ${exponents[idx].toString()}`);
  console.log(`   Mod:  ${moduli[idx].toString().slice(0,10)}...`);
  console.log(`   Exp:  ${expected.toString().slice(0,20)}...`);
  console.log(`   Act:  ${actual.toString().slice(0,20)}...`);

  if (expected === actual) {
    console.log("✅ modPow MATCH!");
  } else {
    console.error("❌ modPow MISMATCH");
  }
})();