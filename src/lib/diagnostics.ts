import { OK, WARN, BAD, MUTED, vadThresholdFor } from './config';
import type {
  AppSettings,
  ChannelRuntimeState,
  IssueEntry,
  IssueGroup,
  PhraseMetric,
  ServerStats,
} from './types';

export function pct(values: Array<number | null | undefined>, q: number): number | null {
  const valid = values
    .filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
    .sort((a, b) => a - b);
  if (valid.length === 0) return null;
  const idx = Math.min(valid.length - 1, Math.floor(valid.length * q));
  return valid[idx];
}

export function judge(value: number | null, okBelow: number, warnBelow: number): string {
  if (value === null) return MUTED;
  return value < okBelow ? OK : value < warnBelow ? WARN : BAD;
}

export function formatDuration(sec: number): string {
  const s = Math.max(0, Math.floor(sec));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  return [h, m, r].map((v) => String(v).padStart(2, '0')).join(':');
}

export function explainIssue(event: string, technical: string, channel?: string): Omit<IssueEntry, 'id' | 'time'> {
  const prefix = channel ? `[${channel}] ` : '';
  switch (event) {
    case 'mic_permission_denied':
      return {
        level: 'ERROR',
        event,
        title: `${prefix}Microphone permission denied`,
        meaning: 'The browser blocked access to the microphone or audio input device.',
        action: 'Allow microphone access in your browser address bar, then click Re-scan devices or Unmute.',
        technical,
        where: 'audioPipeline.ts -> startChannel()',
        details: technical,
      };
    case 'speech_api_unsupported':
      return {
        level: 'WARNING',
        event,
        title: `${prefix}Browser SpeechRecognition unavailable`,
        meaning: 'This browser does not expose Web Speech API (or blocked it inside an embedded frame). Audio meters still run live.',
        action: 'Use Chrome/Edge for live mic speech recognition, or use the Phrase Simulator / Replay bar to test the full translation pipeline.',
        technical,
        where: 'audioPipeline.ts -> startRecognition()',
        details: technical,
      };
    case 'translation_rate_limited':
      return {
        level: 'WARNING',
        event,
        title: 'Online translation rate-limited',
        meaning: 'Google Translate or MyMemory returned HTTP 429 and entered exponential cooldown; the fallback provider is used automatically.',
        action: 'Wait for the cooldown to expire (30 s – 10 min) or continue using cached/fallback translations.',
        technical,
        where: 'server.ts -> /api/translate',
        details: technical,
      };
    case 'translation_failed':
      return {
        level: 'ERROR',
        event,
        title: `${prefix}Online translation failed`,
        meaning: 'All online translation providers failed for a phrase.',
        action: 'Check internet connectivity or wait for provider cooldowns to clear.',
        technical,
        where: 'server.ts -> /api/translate',
        details: technical,
      };
    case 'device_not_found':
      return {
        level: 'WARNING',
        event,
        title: `${prefix}Audio device disconnected`,
        meaning: 'The selected audio device is no longer listed by the browser.',
        action: 'Reconnect the device or choose another input in Settings -> Listen to.',
        technical,
        where: 'audioPipeline.ts -> startChannel()',
        details: technical,
      };
    default:
      return {
        level: 'WARNING',
        event,
        title: `${prefix}${event.replace(/_/g, ' ')}`,
        meaning: technical || 'A pipeline event occurred.',
        action: 'Review technical details below.',
        technical,
        where: 'Web Interpreter runtime',
        details: technical,
      };
  }
}

export function groupIssues(issues: IssueEntry[]): IssueGroup[] {
  const map = new Map<string, IssueGroup>();
  for (const item of issues) {
    if (item.level === 'INFO') continue;
    const existing = map.get(item.event);
    if (!existing) {
      map.set(item.event, {
        event: item.event,
        first: item.time,
        last: item.time,
        count: 1,
        level: item.level === 'ERROR' ? 'ERROR' : 'WARNING',
        issue: item,
      });
    } else {
      existing.count += 1;
      existing.last = item.time;
      existing.issue = item;
      if (item.level === 'ERROR') existing.level = 'ERROR';
    }
  }
  return Array.from(map.values()).sort((a, b) => b.last.localeCompare(a.last));
}

export function buildSpeedReportText(args: {
  startedAtIso: string;
  settings: AppSettings;
  phrases: PhraseMetric[];
  channels: ChannelRuntimeState[];
  issues: IssueEntry[];
  serverStats: ServerStats | null;
  totalPhrases: number;
}): string {
  const { startedAtIso, settings, phrases, channels, issues, serverStats, totalPhrases } = args;
  const p2t = phrases.map((p) => p.pauseToTextMs);
  const fp = phrases.map((p) => p.firstPreviewMs);
  const rec = phrases.map((p) => p.processMs);
  const mt = phrases.map((p) => p.mtMs);
  const fmt = (v: number | null) => (v === null ? '-' : `${Math.round(v)} ms`);

  const lines: string[] = [
    `Voice Interpreter — Speed & Accuracy Report (Web Runtime)`,
    `Session started : ${startedAtIso}`,
    `Report built    : ${new Date().toISOString()}`,
    `Engine / Model  : ${settings.model} (Previews: ${settings.previewModel})`,
    `Phrases tracked : ${phrases.length} recent (${totalPhrases} total this run)`,
    '',
    '── Latency Breakdown (p50 / p95) ──────────────────────────────',
    `  Pause -> final text   : ${fmt(pct(p2t, 0.5))} / ${fmt(pct(p2t, 0.95))}`,
    `  Speech -> 1st preview : ${fmt(pct(fp, 0.5))} / ${fmt(pct(fp, 0.95))}`,
    `  Recognition time      : ${fmt(pct(rec, 0.5))} / ${fmt(pct(rec, 0.95))}`,
    `  Translation time      : ${fmt(pct(mt, 0.5))} / ${fmt(pct(mt, 0.95))}`,
    `  Early final rate      : ${
      phrases.length ? `${Math.round((100 * phrases.filter((p) => p.early).length) / phrases.length)} %` : '-'
    }`,
    '',
    '── Channels ───────────────────────────────────────────────────',
  ];

  settings.channels.forEach((ch, idx) => {
    const rt = channels[idx];
    const med = pct(rt.levels, 0.5);
    const share = rt.speechFlags.length
      ? `${Math.round((100 * rt.speechFlags.filter(Boolean).length) / rt.speechFlags.length)} %`
      : '-';
    lines.push(
      `  [${idx + 1}] ${ch.name} (${ch.src} -> ${ch.tgt}) | ${
        ch.muted ? 'MUTED' : rt.listening ? 'LISTENING' : 'IDLE'
      }`,
      `      Device      : ${ch.device}`,
      `      Sensitivity : ${Math.round(ch.sensitivity * 100)}% (VAD threshold ${vadThresholdFor(ch.sensitivity).toFixed(2)})`,
      `      Level (10s) : ${med === null ? '-' : `${med.toFixed(1)} dBFS`}  |  Speech share: ${share}`,
    );
  });

  if (serverStats) {
    lines.push(
      '',
      '── Server Runtime ─────────────────────────────────────────────',
      `  Node.js         : ${serverStats.nodeVersion} (${serverStats.platform})`,
      `  Memory RSS/Heap : ${serverStats.rssMb} MB / ${serverStats.heapUsedMb} MB`,
      `  Translation cache: ${serverStats.translationCacheSize} / 512 entries`,
    );
  }

  const grouped = groupIssues(issues);
  lines.push('', '── Issues ─────────────────────────────────────────────────────');
  if (grouped.length === 0) {
    lines.push('  No warnings or errors recorded this session.');
  } else {
    for (const g of grouped) {
      lines.push(`  [${g.level}] ${g.issue.title} (${g.count}x, last ${g.last})`);
    }
  }

  return lines.join('\n');
}
