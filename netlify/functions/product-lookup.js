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
// Google Search grounding tool. This function tries WITH search grounding
// first; if Gemini rejects that (common on a brand-new free key), it
// automatically retries WITHOUT grounding so the app still returns a
// best-effort answer instead of failing outright. Ungrounded answers rely on
// the model's training data rather than a live search, so they'll be less
// current and the app's own "needs a real source URL" checks will likely
// blank out fields like rating/price that it can't back up.

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

  const body = {
    systemInstruction: { parts: { text: system || "" } },
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

  let result = await callGemini(apiKey, {
    system: body.system,
    contents,
    maxOutputTokens: body.max_tokens,
    withSearch: true,
  });

  // A free-tier key without search-grounding access typically comes back as
  // a 400 naming the tool. Retry once without it rather than failing.
  if (!result.ok) {
    result = await callGemini(apiKey, {
      system: body.system,
      contents,
      maxOutputTokens: body.max_tokens,
      withSearch: false,
    });
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

  const candidate = result.data.candidates && result.data.candidates[0];
  const parts = (candidate && candidate.content && candidate.content.parts) || [];
  const text = parts
    .map((p) => p.text || "")
    .join("\n")
    .trim();

  return {
    statusCode: 200,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content: [{ type: "text", text }] }),
  };
};
