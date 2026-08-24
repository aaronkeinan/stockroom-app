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

  return {
    statusCode: 200,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content: [{ type: "text", text }] }),
  };
};
