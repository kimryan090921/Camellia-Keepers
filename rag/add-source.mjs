// Adds a document to what the 4·3 bot knows, then rebuilds its search index.
//
//   node rag/add-source.mjs <file.pdf|.txt|.md> [options]     add (or replace) a document
//   node rag/add-source.mjs --list                             show what the bot knows
//   node rag/add-source.mjs --remove <id>                      take a document out
//
// Options:
//   --title "..."       title visitors see in the source list (default: the file name)
//   --official          an official source, e.g. the government's 4·3 report. The bot trusts
//                       these over everything else. Leave it off for essays, articles, notes.
//   --url https://...   where visitors can read the original (shown as a link)
//   --lang ko|en        language of the document (default: detected automatically)
//   --author "..."      --date YYYY-MM-DD
//   --id short-name     id for the document (default: made from the file name)
//   --no-build          don't rebuild yet (when adding several files, build once at the end)
//   --dry-run           show what would be indexed and the cost; changes nothing in the index
//
// One-time setup for PDFs:  cd rag && npm install
// Needs the OpenAI key in Jeju_4-3/.env (the folder ABOVE the website folder), like ingest.mjs.

import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const RAG = path.dirname(fileURLToPath(import.meta.url));
const SOURCES = path.resolve(process.env.SOURCES_DIR || path.join(RAG, 'sources'));
const MANIFEST = path.join(SOURCES, 'manifest.json');

// ───────── arguments ─────────
const argv = process.argv.slice(2);
const opts = { flags: new Set(), values: {}, files: [] };
const VALUE_OPTS = new Set(['--title', '--url', '--lang', '--author', '--date', '--id', '--remove']);
for (let i = 0; i < argv.length; i++) {
  const a = argv[i];
  if (VALUE_OPTS.has(a)) opts.values[a.slice(2)] = argv[++i];
  else if (a.startsWith('--')) opts.flags.add(a.slice(2));
  else opts.files.push(a);
}
const fail = (msg) => { console.error(`\n${msg}\n`); process.exit(1); };
const manifest = JSON.parse(fs.readFileSync(MANIFEST, 'utf8'));
const save = () => fs.writeFileSync(MANIFEST, JSON.stringify(manifest, null, 2) + '\n');

// ───────── --list ─────────
if (opts.flags.has('list')) {
  console.log(`\n${manifest.documents.length} documents in ${path.relative(process.cwd(), SOURCES) || '.'}:\n`);
  for (const d of manifest.documents) {
    const size = fs.existsSync(path.join(SOURCES, d.file)) ? fs.statSync(path.join(SOURCES, d.file)).size : 0;
    console.log(`  ${d.id.padEnd(22)} ${(d.trust === 'primary' ? 'OFFICIAL' : 'other').padEnd(9)} ${(d.language || '?').padEnd(3)} ${String(Math.round(size / 1000)).padStart(5)} KB  ${d.title}`);
  }
  console.log('');
  process.exit(0);
}

// ───────── --remove ─────────
if (opts.values.remove) {
  const id = opts.values.remove;
  const i = manifest.documents.findIndex((d) => d.id === id);
  if (i < 0) fail(`No document with id "${id}". See: node rag/add-source.mjs --list`);
  const [doc] = manifest.documents.splice(i, 1);
  save();
  console.log(`Removed "${doc.title}" from the manifest (its text file ${doc.file} was kept).`);
  build();
  process.exit(0);
}

// ───────── add ─────────
if (opts.files.length !== 1) fail('Usage: node rag/add-source.mjs <file.pdf|.txt|.md> [--title "..."] [--official]\n       node rag/add-source.mjs --list');
const input = path.resolve(opts.files[0]);
if (!fs.existsSync(input)) fail(`File not found: ${input}`);
const ext = path.extname(input).toLowerCase();

let text;
if (ext === '.pdf') text = await pdfToText(input);
else if (['.txt', '.md'].includes(ext)) text = fs.readFileSync(input, 'utf8');
else fail(`Can't read ${ext} files. Use a PDF, .txt or .md (for Word or Hangul files, save as PDF first).`);

const letters = text.replace(/\[\[page \d+\]\]/g, '').replace(/\s/g, '');
if (letters.length < 200) {
  fail(`Only ${letters.length} characters of text found in ${path.basename(input)}.\n` +
       `It is probably a scanned document (pictures of pages, no text layer), which the bot can't read.\n` +
       `It needs OCR first (e.g. open it in Adobe Acrobat or Naver/Google Drive and export as text).`);
}

const base = path.basename(input, ext);
const id = opts.values.id || slug(base);
const official = opts.flags.has('official');
const hangul = (text.match(/[가-힣]/g) || []).length;
const latin = (text.match(/[A-Za-z]/g) || []).length;
const lang = opts.values.lang || (hangul > latin * 0.3 ? 'ko' : 'en');
const folder = official ? 'official' : 'added';
const rel = `${folder}/${id}.txt`;

fs.mkdirSync(path.join(SOURCES, folder), { recursive: true });
fs.writeFileSync(path.join(SOURCES, rel), text);

const entry = {
  id,
  file: rel,
  type: official ? 'official_report' : 'document',
  trust: official ? 'primary' : 'secondary',
  language: lang,
  title: opts.values.title || base,
  author: opts.values.author || null,
  date: opts.values.date || null,
  source_url: opts.values.url || null,
  original_file: path.basename(input),
  added: new Date().toISOString().slice(0, 10),
};
const existing = manifest.documents.findIndex((d) => d.id === id);
if (existing >= 0) manifest.documents[existing] = entry; else manifest.documents.push(entry);
save();

const pages = (text.match(/\[\[page \d+\]\]/g) || []).length;
console.log(`\n${existing >= 0 ? 'Replaced' : 'Added'}: ${entry.title}`);
console.log(`  id ${id} · ${official ? 'OFFICIAL source' : 'regular source'} · ${lang === 'ko' ? 'Korean' : 'English'} · ` +
            `${pages ? pages + ' pages · ' : ''}${letters.length.toLocaleString()} characters → rag/sources/${rel}`);
if (!entry.source_url) console.log('  (no --url given, so visitors will see the title but no link)');

build();

// ───────── helpers ─────────
function build() {
  if (opts.flags.has('no-build')) {
    console.log('\nNot rebuilding yet (--no-build). When done adding files, run:  node rag/ingest.mjs\n');
    return;
  }
  console.log('\nRebuilding the search index…\n');
  const extra = ['dry-run', 'mock'].filter((f) => opts.flags.has(f)).map((f) => `--${f}`);
  const r = spawnSync(process.execPath, [path.join(RAG, 'ingest.mjs'), ...extra], { stdio: 'inherit' });
  if (r.status !== 0) process.exit(r.status || 1);
  if (!opts.flags.has('dry-run')) {
    console.log('\nDone. To put it live, upload these to GitHub:');
    console.log('  • the rag/sources folder');
    console.log('  • netlify/functions/ask/data/index.mjs\n');
  }
}

function slug(s) {
  const out = s.normalize('NFKD').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 40);
  return out || `doc-${Date.now()}`;
}

async function pdfToText(file) {
  let pdfjs, pkgDir;
  try {
    pdfjs = await import('pdfjs-dist/legacy/build/pdf.mjs');
    pkgDir = path.dirname(createRequire(import.meta.url).resolve('pdfjs-dist/package.json'));
  } catch {
    fail('Reading PDFs needs a one-time setup. Run:\n  cd rag && npm install\nthen try again.');
  }
  const doc = await pdfjs.getDocument({
    data: new Uint8Array(fs.readFileSync(file)),
    cMapUrl: path.join(pkgDir, 'cmaps') + path.sep,          // needed for Korean fonts
    cMapPacked: true,
    standardFontDataUrl: path.join(pkgDir, 'standard_fonts') + path.sep,
    isEvalSupported: false,
    useSystemFonts: false,
    verbosity: 0,
  }).promise;
  const out = [];
  for (let n = 1; n <= doc.numPages; n++) {
    const content = await (await doc.getPage(n)).getTextContent();
    let s = '';
    for (const it of content.items) s += (it.str || '') + (it.hasEOL ? '\n' : '');
    out.push(`[[page ${n}]]\n${s.trim()}`);
    if (n % 50 === 0) process.stdout.write(`  read ${n}/${doc.numPages} pages\r`);
  }
  return out.join('\n') + '\n';
}
