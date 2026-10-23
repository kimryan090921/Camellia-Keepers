// Netlify Function: the Camellia Keepers question-answering API.
//
//   GET  /api/ask?site=bukchon      → the site's name, intro and suggested questions (no OpenAI call)
//   POST /api/ask {question, lang?, site?}
//        → { answer, sources: [{n, title, page, official, url}], lang }
//
// Needs OPENAI_API_KEY (Netlify: Site configuration → Environment variables).
// The search index in ./data/index.mjs is built by rag/ingest.mjs.

import index from './data/index.mjs';
import { SITES, GENERAL } from './sites.mjs';
import { embed, chat, search, groupHits, buildPrompt, sourceList, detectLang, isMock, ServiceError } from './lib.mjs';

export const config = { path: '/api/ask' };

const MAX_QUESTION = 500;
const LIMIT_PER_MINUTE = 6;
const LIMIT_PER_DAY = 100;

// Per-visitor limits. Kept in memory, so they reset when Netlify starts a new instance;
// the OpenAI spending limit is the real backstop.
const hits = new Map();
function rateLimited(ip) {
  if (process.env.DISABLE_RATE_LIMIT === '1') return false; // local testing only
  const now = Date.now();
  const list = (hits.get(ip) || []).filter((t) => now - t < 86_400_000);
  const lastMinute = list.filter((t) => now - t < 60_000).length;
  if (lastMinute >= LIMIT_PER_MINUTE || list.length >= LIMIT_PER_DAY) { hits.set(ip, list); return true; }
  list.push(now);
  hits.set(ip, list);
  if (hits.size > 5000) hits.clear();
  return false;
}

const json = (status, body) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' },
  });

export default async (req, context) => {
  const url = new URL(req.url);

  if (req.method === 'GET') {
    if (url.searchParams.has('sites')) {
      return json(200, { sites: Object.fromEntries(Object.entries(SITES).map(([k, s]) => [k, s.name])) });
    }
    const slug = url.searchParams.get('site');
    const site = slug && SITES[slug];
    const s = site || GENERAL;
    return json(200, {
      site: site ? slug : null,
      name: s.name, intro: s.intro, suggestions: s.suggestions,
      ready: index.chunks.length > 0,
    });
  }

  if (req.method !== 'POST') return json(405, { error: 'method_not_allowed' });

  let body;
  try { body = await req.json(); } catch { return json(400, { error: 'bad_request' }); }
  const question = String(body?.question || '').trim();
  if (question.length < 2) return json(400, { error: 'empty_question' });
  if (question.length > MAX_QUESTION) return json(400, { error: 'question_too_long' });
  const lang = body?.lang === 'ko' || body?.lang === 'en' ? body.lang : detectLang(question);
  const slug = typeof body?.site === 'string' ? body.site : null;
  const site = slug && SITES[slug] ? SITES[slug] : null;

  const ip = context?.ip || req.headers.get('x-nf-client-connection-ip') || 'local';
  if (rateLimited(ip)) return json(429, { error: 'rate_limited' });

  if (!index.chunks.length) return json(503, { error: 'index_not_built' });
  if (index.embed_model === 'mock' && !isMock()) return json(503, { error: 'index_is_mock' });

  const started = Date.now();
  try {
    const [qvec] = await embed([question]);
    const found = groupHits(search(index, qvec));
    const { system, user } = buildPrompt({ question, lang, site, hits: found, index });
    const { text: answer, web } = await chat(system, user, found.length);
    const sources = [...sourceList(found, index), ...web.map((w) => ({ n: w.n, title: w.title, page: null, official: false, web: true, url: w.url }))];
    console.log(JSON.stringify({ ok: true, lang, site: slug, ms: Date.now() - started, webSources: web.length, q: question.slice(0, 200) }));
    return json(200, { answer, sources, lang });
  } catch (e) {
    const code = e instanceof ServiceError ? e.code : 'internal';
    console.error(JSON.stringify({ ok: false, code, message: e.message, site: slug }));
    return json(502, { error: 'upstream', code });
  }
};
