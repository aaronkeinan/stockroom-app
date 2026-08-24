# Deploying your Stockroom app (Gemini free-tier prototype)

This folder is a ready-to-deploy version of your app. It's a plain static
site (no build step needed) plus one small server function that keeps your
Gemini API key private while still powering the "look up this product"
feature — using Google's Gemini free API tier so you can try it before
spending anything.

## What's in here

- `index.html` — the whole app (loads React from a CDN, no install needed)
- `netlify/functions/product-lookup.js` — a server function that holds your
  Gemini API key and forwards product-lookup requests to Gemini on the app's
  behalf, translating between the app's original request format and
  Gemini's
- `netlify.toml` — tells Netlify where to find the function

## Steps

### 1. Get a free Gemini API key
Go to **aistudio.google.com**, sign in with a Google account, and create an
API key (usually under "Get API key" in the left sidebar). No billing setup
is required for the free tier. Copy the key — you'll paste it into Netlify
in step 4, never into the code itself.

**Heads up on a limitation:** this app's lookup feature relies on live web
search to research products. Google's free-tier keys don't always include
access to the Search grounding tool that makes this work — Google's own
docs list it as unavailable on the free tier for most models. The function
in this project tries with search first and automatically falls back to a
plain (non-searching) answer if that's rejected, so the app won't just
break — but without grounding, results will lean on the model's general
knowledge rather than live product pages, so prices/ratings/links may be
less accurate or missing (the app already has guards that blank out
fields it can't back up with a real source URL). If you want reliable
grounded results, that's the point where enabling billing on the Google
Cloud project behind your key (which still comes with its own free monthly
credit) unlocks it — no code changes needed, it's the same key.

### 2. Put this code on GitHub
Create a free account at **github.com** if you don't have one, then create a
new repository (public or private, either works). Use **Add file > Upload
files** and drag in this entire folder (including the `netlify` subfolder) —
GitHub preserves the folder structure.

### 3. Deploy on Netlify
Create a free account at **netlify.com**, then **Add new site > Import an
existing project**, connect your GitHub account, and pick the repository you
just created. Netlify will read `netlify.toml` automatically and detect both
the site and the function — just click deploy.

### 4. Add your API key
In your new Netlify site, go to **Site configuration > Environment
variables**, add a variable named `GEMINI_API_KEY` with the key you copied
in step 1, then trigger a redeploy (**Deploys > Trigger deploy**) so the
function picks it up.

Optional: add a second variable `GEMINI_MODEL` if you want to point at a
different free-tier model later (e.g. a newer Flash release) — it defaults
to `gemini-2.5-flash` if you don't set one.

### 5. Get your link
Your site is now live at something like `https://random-name-12345.netlify.app`.
You can rename it to something nicer under **Site configuration > Change
site name** (still a free `netlify.app` address, just with a name you pick,
e.g. `my-stockroom-app.netlify.app`).

**This is the link to submit to Amazon Associates** — it's hosted under your
own Netlify account, not claude.ai, so it should pass their ownership check.

## Notes

- If you skip steps 1 and 4 (no API key configured), the app itself still
  works fully for manual entry — only the "look up this product" button
  will show an error when clicked.
- A custom domain (e.g. `mystockroomapp.com`) is optional and not required
  for Amazon — a free `netlify.app` address is enough. You can add one later
  under **Domain management** in Netlify (domains typically cost $10–15/year
  from a registrar).
- Later, if you decide to switch to Anthropic's Claude API instead of
  Gemini (e.g. for more reliable grounded search), that just means writing
  a different version of `product-lookup.js` — the frontend doesn't need to
  change since it already talks to `/.netlify/functions/product-lookup`
  either way.
