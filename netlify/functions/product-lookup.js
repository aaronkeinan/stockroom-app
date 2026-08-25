// This function runs on Netlify's servers, never in the visitor's browser.
// It holds your Gemini API key (set as the GEMINI_API_KEY environment
// variable in Netlify site settings) and translates the app's requests into
// Gemini API calls, so the key itself is never exposed to visitors.
//
// The frontend still sends requests shaped like Anthropic's Messages API
// (model/system/messages/tools) since that's how the app was originally
// built — this function converts that shape to Gemini's format, calls
// Gemini, then converts the response back into the shape the frontend
// already knows how to read: { content: [{ type: "text", text: "..." }] }.
//
// Free-tier note: Gemini's free API keys don't always include access to the
// Google Search grounding tool, and on some models the search tool itself is
// unreliable (seen returning "MALFORMED_FUNCTION_CALL" with no text at all
// on gemini-3.6-flash). This function tries WITH search grounding first; if
// that fails outright OR comes back with no usable text, it automatically
// retries WITHOUT grounding (explicitly telling the model not to attempt any
// tool call) so the app still returns a best-effort answer instead of
// failing outright. Ungrounded answers rely on the model's training data
// rather than a live search, so they'll be less current and the app's own
// "needs a real source URL" checks will likely blank out fields like
// rating/price that it can't back up.
//
// Trustworthy product links: an LLM can produce a perfectly well-formed
// https://www.amazon.com/.../dp/XXXXXXXXXX link for a listing that's been
// removed or never existed (dead), or — worse — a link that's genuinely
// LIVE but resolves to a completely different product (seen on Home Depot,
// whose /p/.../<id> pages resolve purely by the numeric id and silently
// ignore a mismatched slug). Neither failure shows up in the URL's shape
// alone, which is all the app's own frontend check (looksLikeRealUrl in
// index.html) can see.
//
// The fix uses what Gemini's Google Search grounding tool actually gives us:
// when grounding succeeds, the response includes groundingMetadata citing
// the real pages Google's search found for this query (groundingChunks[].web
// .uri/.title) — these aren't the model's own account of what it read, they're
// Google's. This function follows each citation's redirect to see where it
// really lands, then tries to match one to the product's named store (by
// domain) or to each alternative's name (by matching the citation's title).
// A match found this way is preferred over whatever URL the model wrote for
// that field, since an independently-resolved citation is generally more
// trustworthy than model-generated text — but it is NOT trusted blindly: see
// checkUrlAgainstProduct below. A citation being reachable when Google's
// index was built (or even a moment ago, when this function resolved its
// redirect) doesn't mean the page still shows the right product now —
// listings get delisted, ids get reused, and some stores (Amazon included)
// serve a soft "page not found" screen with an HTTP 200 rather than a real
// 404. So every grounded match is run back through checkUrlAgainstProduct
// just like a model-claimed URL would be, before it's trusted.
//
// This falls back to fetching the model's own claimed URL directly via
// checkUrlAgainstProduct when no grounded match is found for a field
// (grounding wasn't available — e.g. this key hit its free-tier grounding
// quota and fell back to an ungrounded answer — or nothing in the citations
// matched) OR when the grounded match itself fails that same check.
// checkUrlAgainstProduct goes a step further than a plain liveness check: it
// also reads the fetched page's own <title> and compares it against the
// product's name, so a live-but-wrong page (the Home Depot case above) gets
// caught even though it never 404s. The link is only stripped (set to null)
// on a confirmed dead page or a clear title mismatch — a blocked/ambiguous
// response (403/405/429, a timeout, or a page whose title looks like an
// anti-bot challenge) is left alone, since that usually means bot defenses
// rather than a genuinely wrong page, and we'd rather risk keeping a real
// link than wrongly deleting one. If a field ends up null after all of this,
// the app's frontend falls back to a guaranteed-real retailer search link
// rather than showing nothing.

const GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-3.6-flash";

function anthropicMessagesToGeminiContents(messages) {
  return (messages || []).map((m) => {
    const role = m.role === "assistant" ? "model" : "user";
    if (typeof m.content === "string") {
      return { role, parts: [{ text: m.content }] };
    }
    const parts = (m.content || []).map((block) => {
      if (block.type === "text") {
        return { text: block.text };
      }
      if (block.type === "image") {
        return {
          inline_data: {
            mime_type: block.source && block.source.media_type,
            data: block.source && block.source.data,
          },
        };
      }
      return { text: "" };
    });
    return { role, parts };
  });
}

async function callGemini(apiKey, { system, contents, maxOutputTokens, withSearch }) {
  const url =
    "https://generativelanguage.googleapis.com/v1beta/models/" +
    GEMINI_MODEL +
    ":generateContent?key=" +
    encodeURIComponent(apiKey);

  // Without search we explicitly tell the model not to attempt any tool or
  // function call. Some models (seen on gemini-3.6-flash) reflexively try to
  // invoke a search-style function on their own even when no tool is
  // declared, which the API rejects as a "MALFORMED_FUNCTION_CALL" and
  // returns with no usable text — this instruction heads that off so the
  // no-search fallback reliably produces an answer instead of another dead end.
  const effectiveSystem = withSearch
    ? system || ""
    : (system || "") +
      "\n\nDo not call any tools or functions. Answer directly using your own knowledge, without performing a search.";

  const body = {
    systemInstruction: { parts: { text: effectiveSystem } },
    contents,
    generationConfig: { maxOutputTokens: maxOutputTokens || 8192 },
  };
  if (withSearch) {
    body.tools = [{ googleSearch: {} }];
  }

  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  const data = await res.json().catch(() => ({}));
  return { ok: res.ok, status: res.status, data };
}

// Mirrors the frontend's own JSON-extraction (strip markdown fences, take
// the first {...} block) so this function can inspect and patch the same
// object the app will eventually parse, without changing what counts as
// valid output.
function extractJsonObject(text) {
  const cleaned = (text || "").replace(/```json/gi, "```").replace(/```/g, "");
  const start = cleaned.indexOf("{");
  const end = cleaned.lastIndexOf("}");
  if (start === -1 || end === -1 || end < start) return null;
  try {
    return JSON.parse(cleaned.slice(start, end + 1));
  } catch (e) {
    return null;
  }
}

// Phrases that show up on bot-block / CAPTCHA / access-denied interstitials
// rather than a real product page. When a fetched title matches one of these
// we treat the result as inconclusive rather than as evidence the product
// doesn't match — anti-bot pages are common and shouldn't cost a real link.
const BOT_BLOCK_SIGNALS = [
  "are you a human",
  "access denied",
  "captcha",
  "robot check",
  "unusual traffic",
  "verify you are a human",
  "just a moment",
  "attention required",
];

// Checks a claimed product URL two ways: is it even live, and — when we can
// tell — does the page it actually resolves to look like the right product.
// This second check exists because of a real failure mode: some retailers
// (Home Depot's /p/.../<id> pages are the confirmed case) resolve purely by
// a numeric id and silently ignore a mismatched slug, so a guessed id can
// return a genuinely live 200 for a completely different item — no 404, no
// error, just the wrong page. A liveness check alone can't catch that; only
// looking at what the page actually says it is can.
//
// Returns:
//   "dead"     — confirmed 404/410, caller should drop the link
//   "mismatch" — page loaded fine but its title shares none of the
//                product's significant words, caller should drop the link
//   "ok"       — either the title matches, or GET didn't return anything
//                specific enough to compare
//   "unknown"  — timeout, network error, or a blocked/ambiguous status
//                (403/405/429) — usually anti-bot defenses, not a dead or
//                wrong page, so the caller should leave the link untouched
async function checkUrlAgainstProduct(url, productName) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 4500);
  const headers = {
    "User-Agent":
      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
  };
  try {
    let res;
    try {
      res = await fetch(url, { method: "GET", redirect: "follow", signal: controller.signal, headers });
    } catch (getErr) {
      return "unknown";
    }
    if (res.status === 404 || res.status === 410) return "dead";
    if (!res.ok) return "unknown";

    const words = String(productName || "")
      .toLowerCase()
      .split(/\s+/)
      .filter((w) => w.length > 2);
    if (!words.length) return "ok";

    // Only read enough of the body to find <title> — these pages can be
    // large, and we don't need the rest.
    let html = "";
    try {
      const reader = res.body && res.body.getReader ? res.body.getReader() : null;
      if (reader) {
        const decoder = new TextDecoder();
        while (html.length < 20000) {
          const { done, value } = await reader.read();
          if (done) break;
          html += decoder.decode(value, { stream: true });
        }
        reader.cancel().catch(() => {});
      } else {
        html = (await res.text()).slice(0, 20000);
      }
    } catch (readErr) {
      return "ok"; // couldn't read the body — don't penalize the link for that
    }

    const titleMatch = html.match(/<title[^>]*>([^<]*)<\/title>/i);
    if (!titleMatch) return "ok";
    const title = titleMatch[1].toLowerCase();
    if (!title.trim()) return "ok";
    if (BOT_BLOCK_SIGNALS.some((signal) => title.includes(signal))) return "ok";

    const score = words.filter((w) => title.includes(w)).length;
    if (score >= Math.ceil(words.length / 2)) return "ok";
    return "mismatch";
  } catch (e) {
    return "unknown";
  } finally {
    clearTimeout(timeout);
  }
}

// Pulls the real search citations out of a grounded Gemini response. Each
// citation's uri is always an opaque https://vertexaisearch.cloud.google.com
// redirect link (Google never exposes the raw destination up front) — never
// the real page itself — so these are only useful once resolved (see
// resolveGroundingCitations below). Returns [] whenever grounding wasn't
// used or the API didn't include any (e.g. this key/model combo has no
// grounding access, or nothing was found).
function extractGroundingCitations(data) {
  const candidate = data.candidates && data.candidates[0];
  const meta = candidate && candidate.groundingMetadata;
  const chunks = (meta && meta.groundingChunks) || [];
  return chunks
    .map((c) => c && c.web)
    .filter((w) => w && typeof w.uri === "string" && w.uri.trim())
    .map((w) => ({ uri: w.uri.trim(), title: (w.title || "").trim() }));
}

// Follows each citation's redirect (in parallel, with a short per-citation
// timeout so one slow/dead citation can't stall the whole lookup) and
// records where it actually lands. A citation that fails to resolve is
// simply dropped — it's never treated as evidence either way.
async function resolveGroundingCitations(citations) {
  const resolved = [];
  await Promise.allSettled(
    citations.map(async (citation) => {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 4000);
      try {
        const res = await fetch(citation.uri, {
          method: "GET",
          redirect: "follow",
          signal: controller.signal,
          headers: {
            "User-Agent":
              "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
          },
        });
        if (res.ok && res.url) {
          resolved.push({ url: res.url, title: citation.title });
        }
      } catch (e) {
        // unreachable or timed out — not usable as a trusted link
      } finally {
        clearTimeout(timeout);
      }
    })
  );
  return resolved;
}

function hostnameOf(url) {
  try {
    return new URL(url).hostname.replace(/^www\./, "").toLowerCase();
  } catch (e) {
    return "";
  }
}

// Small store-name -> domain map so a resolved citation's hostname can be
// matched back to the store name Gemini named (e.g. "Home Depot" ->
// homedepot.com).
const STORE_DOMAINS = {
  amazon: "amazon.com",
  "home depot": "homedepot.com",
  homedepot: "homedepot.com",
  lowe: "lowes.com",
  target: "target.com",
  walmart: "walmart.com",
  "best buy": "bestbuy.com",
  bestbuy: "bestbuy.com",
  costco: "costco.com",
  wayfair: "wayfair.com",
  "ace hardware": "acehardware.com",
};

function domainForStore(storeName) {
  const s = String(storeName || "").toLowerCase();
  for (const key in STORE_DOMAINS) {
    if (s.includes(key)) return STORE_DOMAINS[key];
  }
  return null;
}

// Finds the best grounded (Google-verified) page for the named store among
// already-resolved citations, preferring one whose path looks like an actual
// product-detail page (e.g. "/dp/", "/p/") over a bare category or search
// page, since that's the one useful as a direct "buy this" link.
function bestGroundedMatchForStore(resolved, storeName) {
  const wantDomain = domainForStore(storeName);
  if (!wantDomain) return null;
  const matches = resolved.filter((r) => {
    const h = hostnameOf(r.url);
    return h === wantDomain || h.endsWith("." + wantDomain);
  });
  if (!matches.length) return null;
  const productLike = matches.find((r) => /\/(dp|p|product|ip|pd)\//i.test(r.url));
  return (productLike || matches[0]).url;
}

// Alternatives don't come with a store name to match on, so instead this
// matches a citation's title against the alternative's own product name — if
// Google's search actually surfaced a page whose title contains most of the
// significant words in that name, that's good evidence it's a real, findable
// listing for that exact product (not a coincidental one-word match).
function bestGroundedMatchForName(resolved, name) {
  if (!name) return null;
  const words = String(name)
    .toLowerCase()
    .split(/\s+/)
    .filter((w) => w.length > 2);
  if (!words.length) return null;
  let best = null;
  let bestScore = 0;
  resolved.forEach((r) => {
    const title = r.title.toLowerCase();
    if (!title) return;
    const score = words.filter((w) => title.includes(w)).length;
    if (score > bestScore) {
      bestScore = score;
      best = r.url;
    }
  });
  return bestScore >= Math.ceil(words.length / 2) ? best : null;
}

// Reconciles listing_url and every alternatives[].url against the grounded
// citations first — a match there is a strong signal, since it's an
// independently-resolved page Google's own search actually surfaced — but
// it still gets run through checkUrlAgainstProduct before being trusted,
// because a citation resolving with an HTTP-ok status when Google indexed it
// (or even moments ago, during resolveGroundingCitations above) doesn't
// guarantee the page still shows that product now. If the grounded match
// fails that check (dead or mismatched title), this falls back to checking
// whatever URL the model itself claimed, exactly as it would if no grounded
// match had been found at all. checkUrlAgainstProduct drops the link on
// either a confirmed-dead page OR a live page whose own title doesn't match
// the product (the Home Depot wrong-id failure mode, or the Amazon
// soft-404-with-200 failure mode) — stricter than a plain liveness check,
// but still fails open (keeps the link) on anything ambiguous like a
// bot-block page. Never throws — any unexpected failure here just leaves the
// data as Gemini returned it, so a bug in this safety net can't take down
// product lookups entirely.
async function reconcileLinks(obj, resolvedCitations) {
  try {
    if (obj.listing_url || obj.listing_url === undefined) {
      const grounded = bestGroundedMatchForStore(resolvedCitations, obj.store || obj.price_source);
      let groundedOk = false;
      if (grounded) {
        const groundedStatus = await checkUrlAgainstProduct(grounded, obj.name);
        groundedOk = groundedStatus !== "dead" && groundedStatus !== "mismatch";
      }
      if (groundedOk) {
        obj.listing_url = grounded;
      } else if (obj.listing_url && typeof obj.listing_url === "string") {
        const status = await checkUrlAgainstProduct(obj.listing_url.trim(), obj.name);
        if (status === "dead" || status === "mismatch") obj.listing_url = null;
      } else if (grounded) {
        // Had a grounded candidate and nothing else to fall back on, but the
        // candidate itself didn't hold up — don't hand back a link we just
        // proved is wrong.
        obj.listing_url = null;
      }
    }

    if (Array.isArray(obj.alternatives)) {
      await Promise.allSettled(
        obj.alternatives.map(async (alt) => {
          if (!alt) return;
          const grounded = bestGroundedMatchForName(resolvedCitations, alt.name);
          let groundedOk = false;
          if (grounded) {
            const groundedStatus = await checkUrlAgainstProduct(grounded, alt.name);
            groundedOk = groundedStatus !== "dead" && groundedStatus !== "mismatch";
          }
          if (groundedOk) {
            alt.url = grounded;
          } else if (typeof alt.url === "string" && alt.url.trim()) {
            const status = await checkUrlAgainstProduct(alt.url.trim(), alt.name);
            if (status === "dead" || status === "mismatch") alt.url = null;
          } else if (grounded) {
            alt.url = null;
          }
        })
      );
    }
  } catch (e) {
    // swallow — see comment above
  }
  return obj;
}

exports.handler = async function (event) {
  if (event.httpMethod !== "POST") {
    return { statusCode: 405, body: "Method Not Allowed" };
  }

  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    return {
      statusCode: 500,
      body: JSON.stringify({
        error:
          "Server is missing GEMINI_API_KEY. Add it in Netlify: Site settings > Environment variables.",
      }),
    };
  }

  let body;
  try {
    body = JSON.parse(event.body || "{}");
  } catch (e) {
    return { statusCode: 400, body: JSON.stringify({ error: "Invalid JSON body" }) };
  }

  const contents = anthropicMessagesToGeminiContents(body.messages);

  function extractText(data) {
    const candidate = data.candidates && data.candidates[0];
    const parts = (candidate && candidate.content && candidate.content.parts) || [];
    return parts
      .map((p) => p.text || "")
      .join("\n")
      .trim();
  }

  let result = await callGemini(apiKey, {
    system: body.system,
    contents,
    maxOutputTokens: body.max_tokens,
    withSearch: true,
  });

  // Grab the real search citations now, before the possible ungrounded
  // retry below overwrites `result` — only the search-grounded call ever
  // has any.
  const groundingCitations = result.ok ? extractGroundingCitations(result.data) : [];

  let text = result.ok ? extractText(result.data) : "";

  // Retry once without search grounding if the first attempt either failed
  // outright (common on a free-tier key with no grounding access — usually
  // a 400 naming the tool) or came back with an HTTP-level success but no
  // usable text. The latter happens when the model's search-tool call itself
  // gets malformed server-side (seen as finishReason "MALFORMED_FUNCTION_CALL"
  // on some models/versions) — a 200 with nothing we can use, which the
  // original !result.ok check alone wouldn't catch.
  if (!result.ok || !text) {
    result = await callGemini(apiKey, {
      system: body.system,
      contents,
      maxOutputTokens: body.max_tokens,
      withSearch: false,
    });
    text = result.ok ? extractText(result.data) : "";
  }

  if (!result.ok) {
    return {
      statusCode: result.status,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        error:
          (result.data && result.data.error && result.data.error.message) ||
          "Gemini API request failed",
      }),
    };
  }

  if (!text) {
    return {
      statusCode: 502,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        error: "Gemini returned an empty response for that item — try again.",
      }),
    };
  }

  // Best-effort link verification. If we can cleanly parse the JSON Gemini
  // produced, replace/check the URLs it named against real search citations
  // (falling back to a plain liveness check where no citation matches), then
  // hand back the patched JSON instead of the raw text. If parsing fails for
  // any reason, fall through and return Gemini's text untouched — the
  // frontend has its own (shape-only) URL checks as a last line of defense
  // either way.
  const parsed = extractJsonObject(text);
  if (parsed) {
    const resolvedCitations = groundingCitations.length
      ? await resolveGroundingCitations(groundingCitations)
      : [];
    await reconcileLinks(parsed, resolvedCitations);
    text = JSON.stringify(parsed);
  }

  return {
    statusCode: 200,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content: [{ type: "text", text }] }),
  };
};
