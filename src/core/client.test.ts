import { createServer, type ServerResponse } from "node:http";
import { AddressInfo } from "node:net";

import { describe, expect, test } from "vitest";
import { Agent, fetch as undiciFetch } from "undici";

import { isTransportFailure, TargetUnreachableError } from "./client";

describe("isTransportFailure", () => {
  test("a refused connection is the target being gone", () => {
    const err = Object.assign(new TypeError("fetch failed"), {
      cause: Object.assign(new Error("connect ECONNREFUSED 127.0.0.1:8080"), {
        code: "ECONNREFUSED",
      }),
    });
    expect(isTransportFailure(err)).toBe(true);
  });

  test("a socket that hangs up mid-response counts", () => {
    expect(isTransportFailure(new Error("socket hang up"))).toBe(true);
  });

  test("a timeout is a slow engine, not a dead one", () => {
    const timeout = Object.assign(new TypeError("fetch failed"), {
      cause: Object.assign(new Error("The operation timed out."), {
        name: "TimeoutError",
      }),
    });
    expect(isTransportFailure(timeout)).toBe(false);
  });

  test("an ordinary error is not a transport failure", () => {
    expect(isTransportFailure(new Error("unexpected end of JSON input"))).toBe(
      false,
    );
  });
});

describe("TargetUnreachableError", () => {
  test("names the errno from the cause chain, undici nests included", () => {
    // The shape undici actually throws: TypeError("fetch failed") wrapping an
    // AggregateError whose members carry the codes.
    const err = new TargetUnreachableError("/chat/completions", {
      name: "TypeError",
      message: "fetch failed",
      cause: {
        name: "AggregateError",
        errors: [
          { name: "Error", code: "ECONNRESET", message: "read ECONNRESET" },
        ],
      },
    });

    expect(err.message).toContain("ECONNRESET");
  });
});

describe("the no-deadline dispatcher", () => {
  test("headersTimeout 0 outlives undici's built-in ceiling", async () => {
    // A non-streaming request to a thinking model gets no headers until
    // generation ends; undici's default headersTimeout kills it. The knob our
    // dispatcher sets to 0 is the only lever. No artificial sleeps here: the
    // impatient fetch dies on undici's own 100ms clock, and the patient one —
    // pending on the same silent server the whole time — must outlive it.
    const held: ServerResponse[] = [];
    const server = createServer((req, res) => {
      held.push(res);
    });
    await new Promise<void>((resolve) =>
      server.listen(0, "127.0.0.1", resolve),
    );
    const url = `http://127.0.0.1:${(server.address() as AddressInfo).port}/x`;

    try {
      const impatient = undiciFetch(url, {
        dispatcher: new Agent({ headersTimeout: 100 }),
      });
      const patient = undiciFetch(url, {
        dispatcher: new Agent({ headersTimeout: 0, bodyTimeout: 0 }),
      });

      await expect(impatient).rejects.toThrow();

      for (const res of held) {
        res.writeHead(200, { "content-type": "application/json" });
        res.end("{}");
      }
      const response = await patient;
      expect(response.status).toBe(200);
    } finally {
      server.close();
    }
  });
});
