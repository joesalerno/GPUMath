(async () => {
  console.log("⚡ Initializing GPU BigNum Engine...");
  const math = new GPUBigNum(64); // 2048-bit precision
  await math.init();

  // 1. Setup Data
  const a = BigInt("0x" + "9".repeat(500)); // Huge number
  const b = BigInt("0x" + "1".repeat(500)); // Huge number
  
  const batchA = Array(10_000).fill(a);
  const batchB = Array(10_000).fill(b);

  // 2. Perform ADDITION
  console.log("🚀 Running Batch ADD...");
  const startAdd = performance.now();
  const addResults = await math.add(batchA, batchB);
  console.log(`   Added 10k numbers in ${(performance.now() - startAdd).toFixed(2)}ms`);
  
  // 3. Perform SUBTRACTION
  console.log("🚀 Running Batch SUB...");
  const startSub = performance.now();
  const subResults = await math.sub(batchA, batchB);
  console.log(`   Subtracted 10k numbers in ${(performance.now() - startSub).toFixed(2)}ms`);

  // 4. Verification
  const jsSum = a + b;
  const jsDiff = a - b;

  console.log("\n🔎 Verification:");
  console.log(`   Add Valid: ${addResults[0] === jsSum ? "✅ Yes" : "❌ No"}`);
  console.log(`   Sub Valid: ${subResults[0] === jsDiff ? "✅ Yes" : "❌ No"}`);
  
  // 5. Edge Case (Negative Result / Underflow)
  // In modular arithmetic, 5 - 10 results in a huge unsigned number (like size_t in C++)
  const smallA = [5n];
  const smallB = [10n];
  const underflowRes = await math.sub(smallA, smallB);
  console.log(`\n⚠️ Underflow Check (5 - 10):`);
  console.log(`   Result (Modular): ${underflowRes[0]}`); 
  // This will be a massive number near 2^2048, which is correct for unsigned GPU math.
})();