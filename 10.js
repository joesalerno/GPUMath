(async () => {
  console.log("⚡ Initializing GPU BigNum (v5) with MODULO...");
  const math = new GPUBigNum(64); // 2048-bit
  await math.init();

  const COUNT = 50; 
  const inputsA = [];
  const inputsB = [];

  console.log(`Generating ${COUNT} pairs...`);
  for(let i=0; i<COUNT; i++) {
    // A = Random large number
    const a = BigInt("0x" + Array(64).fill("F").join("")); 
    // B = Slightly smaller random number
    const b = BigInt("0x" + Array(32).fill("A").join(""));
    inputsA.push(a);
    inputsB.push(b);
  }

  // 1. Test Modulo
  console.log("🚀 Running Batch MOD...");
  const start = performance.now();
  const results = await math.mod(inputsA, inputsB);
  console.log(`   Done in ${(performance.now() - start).toFixed(2)}ms`);

  // 2. Verify
  const idx = 0;
  const expected = inputsA[idx] % inputsB[idx];
  const actual = results[idx];

  console.log("\n🔎 Verification:");
  console.log(`   A: ${inputsA[idx].toString().slice(0,15)}...`);
  console.log(`   B: ${inputsB[idx].toString().slice(0,15)}...`);
  console.log(`   Expected (Mod): ${expected.toString()}`);
  console.log(`   Actual   (Mod): ${actual.toString()}`);

  if (actual === expected) {
    console.log("✅ MOD SUCCESS!");
  } else {
    console.error("❌ MOD FAILED");
  }
})();