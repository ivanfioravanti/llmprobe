import { BudgetExceededError, TargetUnreachableError } from "../core/client";
import type { RunContext } from "../core/context";
import {
  answerMatches,
  extractAnswer,
  isCompsec,
  isMultipleChoice,
} from "./grade";
import { REASONING_CASES } from "./cases";
import type {
  ReasoningCase,
  ReasoningCaseResult,
  ReasoningReport,
  ReasoningSourceSummary,
} from "./types";

export { REASONING_CASES } from "./cases";
export * from "./types";

export const SYSTEM_PROMPT =
  "You are solving a hard benchmark question. Reason carefully. " +
  "The final answer must follow the requested format exactly.";

const TAIL =
  "At the end, write exactly one final line in this format and do not write anything after it:\n";

export function buildPrompt(tc: ReasoningCase): string {
  if (isMultipleChoice(tc)) {
    const choices = tc
      .choices!.map((c, i) => `${String.fromCharCode(65 + i)}. ${c}`)
      .join("\n");
    return `${tc.question}\n\nChoices:\n${choices}\n\nSolve the question. ${TAIL}Answer: <letter>`;
  }
  if (isCompsec(tc)) {
    return `${tc.question}\n\n${TAIL}Answer: <line number or comma-separated line numbers>`;
  }
  return `${tc.question}\n\nSolve the problem. ${TAIL}Answer: <integer>`;
}

const SOURCE_ALIASES: Record<string, string[]> = {
  gpqa: ["GPQA Diamond", "GPQA Diamond (modified)"],
  gpqadiamond: ["GPQA Diamond", "GPQA Diamond (modified)"],
  aime2025: ["AIME2025"],
  supergpqa: ["SuperGPQA"],
  aime: ["AIME2025"],
  compsec: ["COMPSEC"],
};

/** "1,5,9" (1-based), case ids, or sources (gpqa, supergpqa, aime, compsec), in the order given. */
export function selectCases(
  all: ReasoningCase[],
  opts: { limit?: number; sequence?: string },
): ReasoningCase[] {
  let picked = all;
  if (opts.sequence) {
    picked = opts.sequence.split(",").flatMap((raw) => {
      const s = raw.trim();
      const sources = SOURCE_ALIASES[s.toLowerCase().replace(/[^a-z]/g, "")];
      if (sources) return all.filter((c) => sources.includes(c.source));
      const n = Number(s);
      const tc =
        Number.isInteger(n) && n >= 1 && n <= all.length
          ? all[n - 1]
          : all.find((c) => c.id === s);
      if (!tc) {
        throw new Error(
          `--eval-cases: unknown case '${s}' (1..${all.length}, a case id, or ${Object.keys(SOURCE_ALIASES).join("/")})`,
        );
      }
      return [tc];
    });
  }
  if (opts.limit !== undefined && opts.limit > 0)
    picked = picked.slice(0, opts.limit);
  return picked;
}

export interface ReasoningOptions {
  maxTokens: number;
  temperature: number;
  topP?: number;
  limit?: number;
  sequence?: string;
  onCase?: (result: ReasoningCaseResult, index: number, total: number) => void;
}

export async function runReasoning(
  ctx: RunContext,
  opts: ReasoningOptions,
): Promise<ReasoningReport> {
  const surface = ctx.evalSurface;
  if (!surface) throw new Error("no chat-shaped surface available for --eval");

  const cases = selectCases(REASONING_CASES, opts);
  const results: ReasoningCaseResult[] = [];

  for (const [i, tc] of cases.entries()) {
    const started = Date.now();
    const base = {
      id: tc.id,
      source: tc.source,
      domain: tc.domain,
      title: tc.title,
      expected: tc.answer,
    };
    let result: ReasoningCaseResult;
    try {
      const res = await ctx.send(
        surface,
        {
          system: SYSTEM_PROMPT,
          turns: [{ type: "user", text: buildPrompt(tc) }],
          maxTokens: opts.maxTokens,
          temperature: opts.temperature,
          ...(opts.topP !== undefined ? { topP: opts.topP } : {}),
          allowReasoning: false,
        },
        // A 16k-token think runs for minutes; the token cap is the bound here,
        // and a clock would grade our patience rather than the model.
        { timeoutMs: null },
      );
      const text = res.reply.text ?? "";
      const got = extractAnswer(tc, text);
      const passed = answerMatches(tc, got);
      const truncated = res.reply.finishReason === "length";
      result = {
        ...base,
        status: passed ? "passed" : truncated ? "stopped" : "failed",
        got,
        text: text.replace(/[\s\S]*<\/think>/, "").trim(),
        finishReason: res.reply.finishReason,
        outputTokens: res.reply.usage.outputTokens,
        reasoningTokens: res.reply.usage.reasoningTokens,
        durationMs: Date.now() - started,
      };
    } catch (err) {
      if (
        err instanceof TargetUnreachableError ||
        err instanceof BudgetExceededError
      ) {
        throw err;
      }
      result = {
        ...base,
        status: "error",
        got: "?",
        text: "",
        finishReason: null,
        outputTokens: null,
        reasoningTokens: null,
        durationMs: Date.now() - started,
        error: err instanceof Error ? err.message : String(err),
      };
    }
    results.push(result);
    opts.onCase?.(result, i, cases.length);
  }

  const bySource: ReasoningSourceSummary[] = [];
  for (const r of results) {
    let s = bySource.find((x) => x.source === r.source);
    if (!s) {
      s = {
        source: r.source,
        passed: 0,
        failed: 0,
        stopped: 0,
        error: 0,
        total: 0,
      };
      bySource.push(s);
    }
    s[r.status] += 1;
    s.total += 1;
  }
  const count = (st: ReasoningCaseResult["status"]) =>
    results.filter((r) => r.status === st).length;

  const scopeNote =
    cases.length !== REASONING_CASES.length
      ? `${cases.length} of ${REASONING_CASES.length} questions — not comparable to a full run`
      : null;

  return {
    passed: count("passed"),
    total: results.length,
    stopped: count("stopped"),
    error: count("error"),
    maxTokens: opts.maxTokens,
    temperature: opts.temperature,
    bySource,
    cases: results,
    scopeNote,
  };
}
