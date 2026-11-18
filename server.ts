import express from 'express';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer as createViteServer } from 'vite';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT || 3000);
const HOST = '0.0.0.0';

// ── Translation LRU Cache & Provider Backoff (ported from interpreter/translation.py) ──

const MAX_CACHE_SIZE = 512;
const translationCache = new Map<string, string>();

function getCachedTranslation(key: string): string | undefined {
  const val = translationCache.get(key);
  if (val !== undefined) {
    translationCache.delete(key);
    translationCache.set(key, val);
  }
  return val;
}

function setCachedTranslation(key: string, value: string): void {
  if (translationCache.has(key)) {
    translationCache.delete(key);
  } else if (translationCache.size >= MAX_CACHE_SIZE) {
    const oldest = translationCache.keys().next().value;
    if (oldest !== undefined) {
      translationCache.delete(oldest);
    }
  }
  translationCache.set(key, value);
}

class RateLimitedError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'RateLimitedError';
  }
}

const PROVIDER_COOLDOWN_BASE_MS = 30_000;
const PROVIDER_COOLDOWN_MAX_MS = 600_000;

const providerState: Record<'google' | 'mymemory', { until: number; cooldown: number }> = {
  google: { until: 0, cooldown: 0 },
  mymemory: { until: 0, cooldown: 0 },
};

const MYMEMORY_REGION_MAP: Record<string, string> = {
  en: 'en-GB',
  ru: 'ru-RU',
  ar: 'ar-SA',
  bg: 'bg-BG',
  'zh-CN': 'zh-CN',
  zh: 'zh-CN',
  cs: 'cs-CZ',
  nl: 'nl-NL',
  fr: 'fr-FR',
  de: 'de-DE',
  el: 'el-GR',
  hi: 'hi-IN',
  it: 'it-IT',
  ja: 'ja-JP',
  ko: 'ko-KR',
  pl: 'pl-PL',
  pt: 'pt-PT',
  es: 'es-ES',
  tr: 'tr-TR',
  uk: 'uk-UA',
};

function cleanTranslation(source: string, translated: string): string {
  const splitSentences = (t: string) =>
    t
      .trim()
      .split(/(?<=[.!?…])\s+/)
      .filter(Boolean);
  const sentences = splitSentences(translated);
  if (splitSentences(source).length > 1) {
    return translated;
  }
  const kept = sentences.filter(
    (x, i) => i === 0 || x.trim().toLowerCase() !== sentences[i - 1].trim().toLowerCase(),
  );
  let out = kept.join(' ');
  const srcWords = source.trim().split(/\s+/).filter(Boolean).length;
  const outWords = out.trim().split(/\s+/).filter(Boolean).length;
  if (srcWords <= 2 && outWords > 3 * srcWords + 3 && kept.length > 1) {
    out = kept[0];
  }
  return out;
}

async function googleTranslate(text: string, src: string, tgt: string): Promise<{ translated: string; detectedSrc?: string }> {
  const params = new URLSearchParams({
    client: 'gtx',
    sl: src || 'auto',
    tl: tgt,
    dt: 't',
    q: text,
  });
  const res = await fetch(`https://translate.googleapis.com/translate_a/single?${params.toString()}`, {
    headers: { 'User-Agent': 'Mozilla/5.0' },
    signal: AbortSignal.timeout(6000),
  });
  if (res.status === 429) {
    throw new RateLimitedError('Google returned 429 Too Many Requests');
  }
  if (!res.ok) {
    throw new Error(`Google HTTP ${res.status}`);
  }
  const data = (await res.json()) as [Array<[string | null, ...unknown[]]>, unknown, string?];
  const translated = (data[0] || [])
    .map((part) => part[0] ?? '')
    .join('')
    .trim();
  const detectedSrc = typeof data[2] === 'string' ? data[2] : undefined;
  return { translated, detectedSrc };
}

async function mymemoryTranslate(text: string, src: string, tgt: string): Promise<{ translated: string; detectedSrc?: string }> {
  if (src === 'auto') {
    throw new Error('MyMemory does not support auto-detect');
  }
  const srcCode = MYMEMORY_REGION_MAP[src] || src;
  const tgtCode = MYMEMORY_REGION_MAP[tgt] || tgt;
  const params = new URLSearchParams({
    q: text,
    langpair: `${srcCode}|${tgtCode}`,
  });
  const res = await fetch(`https://api.mymemory.translated.net/get?${params.toString()}`, {
    headers: { 'User-Agent': 'Mozilla/5.0' },
    signal: AbortSignal.timeout(6000),
  });
  if (res.status === 429) {
    throw new RateLimitedError('MyMemory returned 429 Too Many Requests');
  }
  if (!res.ok) {
    throw new Error(`MyMemory HTTP ${res.status}`);
  }
  const data = (await res.json()) as {
    responseStatus?: number | string;
    responseDetails?: string;
    responseData?: { translatedText?: string };
  };
  if (
    data.responseStatus === 429 ||
    String(data.responseDetails || '')
      .toUpperCase()
      .includes('QUOTA')
  ) {
    throw new RateLimitedError(String(data.responseDetails || 'MyMemory quota exceeded'));
  }
  if (data.responseStatus !== 200 && data.responseStatus !== '200') {
    throw new Error(`MyMemory error: ${data.responseDetails || 'unknown'}`);
  }
  return {
    translated: (data.responseData?.translatedText || '').trim(),
  };
}

// ── In-memory structured logs & mistake feedback (ephemeral per container) ──

interface LogRecord {
  ts: string;
  level: string;
  event: string;
  channel?: string;
  fields?: Record<string, unknown>;
}

const MAX_LOGS = 500;
const logsBuffer: LogRecord[] = [];
const feedbackReports: Record<string, unknown>[] = [];

function pushLog(entry: LogRecord): void {
  logsBuffer.push(entry);
  if (logsBuffer.length > MAX_LOGS) {
    logsBuffer.shift();
  }
}

async function startServer(): Promise<void> {
  const app = express();
  app.use(express.json({ limit: '1mb' }));

  app.get('/api/health', (_req, res) => {
    res.json({
      ok: true,
      runtime: `Node.js ${process.version}`,
      uptimeSec: Math.round(process.uptime()),
    });
  });

  app.post('/api/translate', async (req, res) => {
    const { text, src = 'auto', tgt = 'en' } = req.body as {
      text?: string;
      src?: string;
      tgt?: string;
    };

    const cleanText = (text || '').trim();
    if (!cleanText) {
      res.json({ translated: '', provider: 'none', elapsedMs: 0, cached: false });
      return;
    }
    if (src === tgt) {
      res.json({ translated: cleanText, provider: 'identity', elapsedMs: 0, cached: false, detectedSrc: src });
      return;
    }

    const cacheKey = `${src}|${tgt}|${cleanText}`;
    const cached = getCachedTranslation(cacheKey);
    if (cached !== undefined) {
      res.json({ translated: cached, provider: 'cache', elapsedMs: 0.2, cached: true });
      return;
    }

    const providers: Array<{
      name: 'google' | 'mymemory';
      fn: (t: string, s: string, tg: string) => Promise<{ translated: string; detectedSrc?: string }>;
    }> = [
      { name: 'google', fn: googleTranslate },
      { name: 'mymemory', fn: mymemoryTranslate },
    ];

    const errors: string[] = [];
    for (const provider of providers) {
      const state = providerState[provider.name];
      const waitMs = state.until - Date.now();
      if (waitMs > 0) {
        errors.push(`${provider.name}: cooling down (${Math.ceil(waitMs / 1000)}s)`);
        continue;
      }

      const t0 = performance.now();
      try {
        const result = await provider.fn(cleanText, src, tgt);
        state.cooldown = 0;
        const cleaned = cleanTranslation(cleanText, result.translated || cleanText);
        const elapsedMs = Math.round((performance.now() - t0) * 100) / 100;
        setCachedTranslation(cacheKey, cleaned);
        pushLog({
          ts: new Date().toISOString(),
          level: 'INFO',
          event: 'translation_completed',
          fields: {
            provider: provider.name,
            src,
            tgt,
            input_chars: cleanText.length,
            output_chars: cleaned.length,
            elapsed_ms: elapsedMs,
          },
        });
        res.json({
          translated: cleaned,
          provider: provider.name,
          elapsedMs,
          cached: false,
          detectedSrc: result.detectedSrc,
        });
        return;
      } catch (err) {
        if (err instanceof RateLimitedError) {
          state.cooldown = Math.min(
            PROVIDER_COOLDOWN_MAX_MS,
            Math.max(PROVIDER_COOLDOWN_BASE_MS, state.cooldown * 2),
          );
          state.until = Date.now() + state.cooldown;
          pushLog({
            ts: new Date().toISOString(),
            level: 'WARNING',
            event: 'translation_rate_limited',
            fields: { provider: provider.name, cooldown_s: state.cooldown / 1000, error: err.message },
          });
          errors.push(`${provider.name}: rate limited`);
        } else {
          const msg = err instanceof Error ? err.message : String(err);
          pushLog({
            ts: new Date().toISOString(),
            level: 'WARNING',
            event: 'translation_provider_failed',
            fields: { provider: provider.name, src, tgt, error: msg },
          });
          errors.push(`${provider.name}: ${msg}`);
        }
      }
    }

    pushLog({
      ts: new Date().toISOString(),
      level: 'ERROR',
      event: 'translation_failed',
      fields: { src, tgt, errors },
    });
    res.status(502).json({
      error: errors.join('; ') || 'All translation providers failed',
      translated: `[translation failed: ${cleanText}]`,
    });
  });

  app.get('/api/stats', (_req, res) => {
    const mem = process.memoryUsage();
    const now = Date.now();
    res.json({
      nodeVersion: process.version,
      platform: `${os.platform()} ${os.release()}`,
      arch: os.arch(),
      cpus: os.cpus().length,
      rssMb: Math.round(mem.rss / (1024 * 1024)),
      heapUsedMb: Math.round(mem.heapUsed / (1024 * 1024)),
      uptimeSec: Math.round(process.uptime()),
      loadAvg1m: Math.round((os.loadavg()[0] || 0) * 100) / 100,
      translationCacheSize: translationCache.size,
      providers: {
        google: {
          coolingDownSec: Math.max(0, Math.ceil((providerState.google.until - now) / 1000)),
          cooldownSec: Math.round(providerState.google.cooldown / 1000),
        },
        mymemory: {
          coolingDownSec: Math.max(0, Math.ceil((providerState.mymemory.until - now) / 1000)),
          cooldownSec: Math.round(providerState.mymemory.cooldown / 1000),
        },
      },
    });
  });

  app.get('/api/logs', (_req, res) => {
    res.json({ logs: logsBuffer });
  });

  app.post('/api/logs', (req, res) => {
    const body = req.body as Partial<LogRecord>;
    pushLog({
      ts: body.ts || new Date().toISOString(),
      level: body.level || 'INFO',
      event: body.event || 'client_event',
      channel: body.channel,
      fields: body.fields,
    });
    res.json({ ok: true });
  });

  app.get('/api/feedback', (_req, res) => {
    res.json({ reports: feedbackReports });
  });

  app.post('/api/feedback', (req, res) => {
    const body = (req.body || {}) as Record<string, unknown>;
    const report: Record<string, unknown> = {
      id: `fb-${Date.now()}`,
      timestamp: new Date().toISOString(),
      ...body,
    };
    feedbackReports.unshift(report);
    if (feedbackReports.length > 100) {
      feedbackReports.pop();
    }
    pushLog({
      ts: new Date().toISOString(),
      level: 'INFO',
      event: 'feedback_saved',
      channel: String(report.channel || ''),
      fields: { problems: report.problems },
    });
    res.json({ ok: true, report });
  });

  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(__dirname, 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  app.listen(PORT, HOST, () => {
    console.log(`Voice Interpreter server listening on http://${HOST}:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
