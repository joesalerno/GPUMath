import { describe, it, expect } from 'vitest';
import * as math from '../src/math-utils.js';

describe('Math Utilities', () => {
    const L = 64;
    const F = 32;

    it('should convert a positive float to a BigInt and back', () => {
        const big = math.floatToBig(1.5, F);
        const str = math.bigToFloatStr(big, F);
        expect(str).toBe("1.5");
    });

    it('should convert a negative float to a BigInt and back', () => {
        const big = math.floatToBig(-123.456, F);
        const str = math.bigToFloatStr(big, F);
        expect(str.startsWith("-123.456")).toBe(true);
    });

    it('should convert a BigInt to a buffer and back', () => {
        const bigInts = [12345678901234567890n, -98765432109876543210n];
        const buffer = math.toBuffer(bigInts, L);
        const converted = math.fromBuffer(buffer, L);
        expect(converted).toEqual(bigInts);
    });
});
