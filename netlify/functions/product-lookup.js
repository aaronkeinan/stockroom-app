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
// Link-liveness check: neither the search-grounded nor the ungrounded model
// call actually confirms the product URL it names is still a real, working
// page — an LLM can produce a perfectly well-formed
// https://www.amazon.com/.../dp/XXXXXXXXXX link for a listing that's been
// removed or never existed, and the app's own frontend checks (see
// looksLikeRealUrl in index.html) only check the URL's shape, not whether it
// resolves. Since this function runs on Netlify's servers, it CAN make its
// own outbound request to check — so after Gemini responds, this function
// fetches the claimed listing_url (and each alternative's url) itself and
// strips out any that come back 404/410 (confirmed dead) before handing the
// result to the app. A blocked/ambiguous response (403/405/429, or a
// timeout) is left alone rather than stripped, since that usually means
// anti-bot defenses rather than a genuinely dead page, and we'd rather risk
// keeping a real link than wrongly deleting one.

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

// Returns "dead", "alive", or "unknown" for a given URL. Only "dead" causes
// the caller to strip the link — "unknown" (timeout, network error, or an
// ambiguous status like 403/405/429 that usually means bot-blocking rather
// than a missing page) leaves the link untouched.
async function checkUrlLiveness(url) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 3500);
  const fetchOpts = (method) => ({
    method,
    redirect: "follow",
    signal: controller.signal,
    headers: {
      "User-Agent":
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    },
  });
  try {
    let res;
    try {
      res = await fetch(url, fetchOpts("HEAD"));
    } catch (headErr) {
      // Some sites reject HEAD outright (not the same as the page being
      // gone) — retry once with GET before giving up on this URL.
      res = await fetch(url, fetchOpts("GET"));
    }
    if (res.status === 404 || res.status === 410) return "dead";
    return "alive";
  } catch (e) {
    return "unknown";
  } finally {
    clearTimeout(timeout);
  }
}

// Checks listing_url and every alternatives[].url in parallel and blanks out
// (sets to null) any confirmed-dead one. Never throws — any unexpected
// failure here just leaves the data as Gemini returned it, so a bug in this
// safety net can't take down product lookups entirely.
async function stripDeadLinks(obj) {
  try {
    const checks = [];

    if (obj.listing_url && typeof obj.listing_url === "string") {
      checks.push(
        checkUrlLiveness(obj.listing_url.trim()).then((status) => {
          if (status === "dead") obj.listing_url = null;
        })
      );
    }

    if (Array.isArray(obj.alternatives)) {
      obj.alternatives.forEach((alt) => {
        if (alt && typeof alt.url === "string" && alt.url.trim()) {
          checks.push(
            checkUrlLiveness(alt.url.trim()).then((status) => {
              if (status === "dead") alt.url = null;
            })
          );
        }
      });
    }

    await Promise.allSettled(checks);
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
  // produced, check the URLs it named and drop any that are confirmed dead,
  // then hand back the patched JSON instead of the raw text. If parsing
  // fails for any reason, fall through and return Gemini's text untouched —
  // the frontend has its own (shape-only) URL checks as a last line of
  // defense either way.
  const parsed = extractJsonObject(text);
  if (parsed) {
    await stripDeadLinks(parsed);
    text = JSON.stringify(parsed);
  }

  return {
    statusCode: 200,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content: [{ type: "text", text }] }),
  };
};
