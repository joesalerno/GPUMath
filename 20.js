(async () => {
  console.log("⚡ Initializing GPU BigNum (v10) - EXP & LN...");
  const math = new GPUBigNum(64); 
  await math.init();

  const inputs = [];
  // 1. e^1.0 = 2.718...
  inputs.push(math.floatToBig(1.0));
  // 2. e^10.0 = 22026.46...
  inputs.push(math.floatToBig(10.0));
  // 3. ln(100.0) = 4.605...
  inputs.push(math.floatToBig(100.0));
  // 4. ln(Huge) -> ln(2^100) = 100 * 0.693 = 69.3
  const huge = 1n << (100n + BigInt(32*32)); // 2^100 in fixed point
  inputs.push(huge);

  console.log("🚀 Running EXP...");
  const expRes = await math.exp(inputs);
  
  console.log("🚀 Running LN...");
  const lnRes = await math.ln(inputs);

  console.log("\n🔎 Results:");
  console.log(`e^1.0   = ${math.bigToFloatStr(expRes[0])} (Exp: 2.71828)`);
  console.log(`e^10.0  = ${math.bigToFloatStr(expRes[1])} (Exp: 22026.46)`);
  
  console.log(`ln(100) = ${math.bigToFloatStr(lnRes[2])} (Exp: 4.60517)`);
  console.log(`ln(2^100)= ${math.bigToFloatStr(lnRes[3])} (Exp: 69.3147)`);
})();