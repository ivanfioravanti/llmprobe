import { afterEach, describe, expect, test } from "vitest";

import { ADAPTERS } from "../conformance/index";
import type { SurfaceAdapter } from "../core/adapter";
import { EngineClient, type RunConfig } from "../core/client";
import { createContext } from "../core/context";
import { normalizeRoot } from "../core/probe";
import { type MockEngine, startMockEngine } from "../fixtures/mock-engine";
import { REASONING_CASES, runReasoning, selectCases } from "./index";

let engine: MockEngine | null = null;
afterEach(() => {
  engine?.stop();
  engine = null;
});

describe("runReasoning against the mock", () => {
  test("grades every selected case and sums by source", async () => {
    engine = await startMockEngine();
    const config: RunConfig = {
      baseUrl: `${normalizeRoot(engine.url)}/v1`,
      apiKey: "",
      model: "mock-model-12b",
      timeoutMs: 15_000,
      depth: "default",
      reasoningHeadroom: 0,
    };
    const ctx = createContext({
      config,
      client: new EngineClient(config),
      adapters: new Map<string, SurfaceAdapter>(ADAPTERS.map((a) => [a.id, a])),
      present: new Set(["chat"]),
      evalSurface: "chat",
    });

    // First GPQA case (answer B), first AIME case, first COMPSEC case.
    const aime = REASONING_CASES.findIndex((c) => c.source === "AIME2025") + 1;
    const compsec =
      REASONING_CASES.findIndex((c) => c.source === "COMPSEC") + 1;
    const report = await runReasoning(ctx, {
      maxTokens: 64,
      temperature: 0,
      sequence: `1,${aime},${compsec}`,
    });

    expect(report.total).toBe(3);
    expect(report.scopeNote).toMatch(/3 of/);
    // The mock answers "Answer: B" to multiple choice, "Answer: 0" otherwise.
    expect(report.cases.map((c) => c.status)).toEqual([
      "passed",
      "failed",
      "failed",
    ]);
    expect(report.passed).toBe(1);
    expect(
      report.bySource.map((s) => `${s.source}:${s.passed}/${s.total}`),
    ).toEqual(["GPQA Diamond:1/1", "AIME2025:0/1", "COMPSEC:0/1"]);
  });
});

describe("selectCases", () => {
  test("limit and sequence", () => {
    expect(selectCases(REASONING_CASES, { limit: 2 })).toHaveLength(2);
    expect(
      selectCases(REASONING_CASES, { sequence: "3,1" }).map((c) => c.id),
    ).toEqual([REASONING_CASES[2]!.id, REASONING_CASES[0]!.id]);
    expect(() => selectCases(REASONING_CASES, { sequence: "999" })).toThrow();
  });
});
