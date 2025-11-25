import { describe, it, expect, beforeEach } from 'vitest';
import { GPUEngine } from '../src/gpu-engine.js';

describe('GPUEngine', () => {
    let engine;

    beforeEach(() => {
        engine = new GPUEngine(64, 32);
    });

    it('should have math helper functions', () => {
        expect(engine.floatToBig).toBeInstanceOf(Function);
        expect(engine.bigToFloatStr).toBeInstanceOf(Function);
        expect(engine.toBuffer).toBeInstanceOf(Function);
        expect(engine.fromBuffer).toBeInstanceOf(Function);
    });
});
