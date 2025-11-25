(async () => {
  console.log("⚡ Initializing GPU BigNum (v4) with DIVISION...");
  const math = new GPUBigNum(64); // 2048-bit precision
  await math.init();

  // 1. Setup Data
  const inputsA = [];
  const inputsB = [];
  const COUNT = 100; 

  console.log(`Generate ${COUNT} random pairs...`);
  for(let i=0; i<COUNT; i++) {
    // Generate random huge numbers
    const a = BigInt("0x" + Array(100).fill("A").join(""));
    const b = BigInt("0x" + Array(50).fill("B").join("")); // Smaller than A
    inputsA.push(a);
    inputsB.push(b);
  }

  // 2. Run Division
  console.log("🚀 Running Batch DIV...");
  const start = performance.now();
  const results = await math.div(inputsA, inputsB);
  console.log(`   Done in ${(performance.now() - start).toFixed(2)}ms`);

  // 3. Verify
  console.log("\n🔎 Verification:");
  const idx = 0;
  const expected = inputsA[idx] / inputsB[idx];
  const actual = results[idx];

  console.log(`   A: ${inputsA[idx].toString().slice(0,15)}...`);
  console.log(`   B: ${inputsB[idx].toString().slice(0,15)}...`);
  console.log(`   Expected (Q): ${expected.toString().slice(0,15)}...`);
  console.log(`   Actual   (Q): ${actual.toString().slice(0,15)}...`);

  if (actual === expected) {
    console.log("✅ DIV SUCCESS!");
  } else {
    console.error("❌ DIV FAILED");
    console.log("Exp:", expected);
    console.log("Act:", actual);
  }
})();