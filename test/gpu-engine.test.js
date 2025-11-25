import { describe, it, expect } from '@jest/globals';
import { GPUEngine } from '../src/gpu-engine.js';

describe('GPUEngine', () => {
    let engine;
    beforeEach(() => {
        engine = new GPUEngine(64, 32);
    });

    it('should convert a positive float to a BigInt and back', () => {
        const big = engine.floatToBig(1.5);
        const str = engine.bigToFloatStr(big);
        expect(str).toBe("1.5000000000");
    });

    it('should convert a negative float to a BigInt and back', () => {
        const big = engine.floatToBig(-123.456);
        const str = engine.bigToFloatStr(big);
        // Using startsWith because of potential minor precision differences
        expect(str.startsWith("-123.456")).toBe(true);
    });
});
