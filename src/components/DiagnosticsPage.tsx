import { Copy, Gauge, RefreshCw } from 'lucide-react';
import {
  BAD,
  CHANNEL_COLOURS,
  LANGUAGES,
  MODELS,
  MUTED,
  OK,
  WARN,
  vadThresholdFor,
} from '../lib/config';
import { formatDuration, groupIssues, judge, pct } from '../lib/diagnostics';
import type {
  AppSettings,
  ChannelRuntimeState,
  IssueEntry,
  PhraseMetric,
  ServerStats,
} from '../lib/types';

interface DiagnosticsPageProps {
  settings: AppSettings;
  channels: ChannelRuntimeState[];
  phrases: PhraseMetric[];
  issues: IssueEntry[];
  serverStats: ServerStats | null;
  listeningSinceMs: number | null;
  watchdogRestarts: number;
  waitingPhrases: number;
  waitingTranslations: number;
  totalPhrasesRun: number;
  onCopyDiagnostics: () => void;
  onOpenSpeedReport: () => void;
  onRefreshStats: () => void;
}

interface CardRow {
  name: string;
  tip: string;
  value: string;
  colour?: string;
}

function MetricCard({ title, rows }: { title: string; rows: CardRow[] }) {
  return (
    <div className="rounded-lg border border-white/10 bg-[#262626] overflow-hidden">
      <div className="border-b border-white/10 bg-[#2d2d30] px-4 py-2.5 text-sm font-semibold text-gray-100">
        {title}
      </div>
      <div className="grid grid-cols-[minmax(150px,auto)_1fr] gap-x-5 gap-y-1.5 px-4 py-3 text-xs">
        {rows.map((r) => (
          <div key={r.name} className="contents">
            <div className="text-gray-400" title={r.tip}>
              {r.name}
            </div>
            <div
              className="font-medium break-words"
              title={r.tip}
              style={{ color: r.colour ?? '#f3f3f3' }}
            >
              {r.value}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function DiagnosticsPage({
  settings,
  channels,
  phrases,
  issues,
  serverStats,
  listeningSinceMs,
  watchdogRestarts,
  waitingPhrases,
  waitingTranslations,
  totalPhrasesRun,
  onCopyDiagnostics,
  onOpenSpeedReport,
  onRefreshStats,
}: DiagnosticsPageProps) {
  const listeningChannels = channels.filter((c) => c.listening);
  const stuckChannels = settings.channels
    .map((s, idx) => ({ s, rt: channels[idx] }))
    .filter((item) => !item.s.muted && !item.rt.listening);

  let overallText = 'Idle';
  let overallColour = WARN;
  if (settings.channels.every((c) => c.muted)) {
    overallText = 'Muted';
    overallColour = WARN;
  } else if (waitingPhrases >= 5) {
    overallText = 'Falling behind: phrases are piling up';
    overallColour = BAD;
  } else if (listeningChannels.length > 0 && stuckChannels.length > 0) {
    overallText = `Listening on ${listeningChannels.length} channel; ${stuckChannels[0].s.name} is not: ${
      stuckChannels[0].rt.problem || 'stopped'
    }`;
    overallColour = WARN;
  } else if (listeningChannels.length > 0) {
    overallText = 'Healthy - listening';
    overallColour = OK;
  }

  const groupedIssues = groupIssues(issues);
  const errorGroups = groupedIssues.filter((g) => g.level === 'ERROR');
  if (errorGroups.length > 0 && overallColour === OK) {
    overallText =
      errorGroups.length === 1
        ? `Listening - problem: ${errorGroups[0].issue.title}`
        : `Listening - ${errorGroups.length} problems, see Recent issues below`;
    overallColour = WARN;
  }

  const translationSummary = settings.channels
    .map((c) => {
      const src = LANGUAGES[c.src]?.translateCode ?? 'auto';
      const tgt = LANGUAGES[c.tgt]?.translateCode ?? 'en';
      return `${c.name}: Online (${src}→${tgt})`;
    })
    .join('   ');

  const healthRows: CardRow[] = [
    { name: 'Overall', tip: 'Summary of both pipelines', value: overallText, colour: overallColour },
    {
      name: 'Listening for',
      tip: 'Time since listening started',
      value:
        listeningSinceMs && listeningChannels.length > 0
          ? formatDuration((Date.now() - listeningSinceMs) / 1000)
          : '-',
    },
    {
      name: 'Main model',
      tip: 'Active transcription engine / model profile',
      value: `${MODELS[settings.model] ?? settings.model} (Browser Web Speech + Simulator)`,
    },
    {
      name: 'Preview model',
      tip: 'Live interim preview strategy',
      value: settings.previewModel,
      colour: OK,
    },
    {
      name: 'Translation',
      tip: 'Online translation backend (Google + MyMemory fallback)',
      value: translationSummary,
      colour: OK,
    },
  ];

  const p2t = phrases.map((p) => p.pauseToTextMs);
  const fp = phrases.map((p) => p.firstPreviewMs);
  const rec = phrases.map((p) => p.processMs);
  const mt = phrases.map((p) => p.mtMs);
  const fmtMs = (a: number | null, b: number | null) =>
    a === null ? '-' : `${Math.round(a)} ms / ${Math.round(b ?? a)} ms`;

  const p2t50 = pct(p2t, 0.5);
  const fp50 = pct(fp, 0.5);
  const rec50 = pct(rec, 0.5);
  const mt50 = pct(mt, 0.5);
  const earlyRate = phrases.length
    ? (100 * phrases.filter((p) => p.early).length) / phrases.length
    : null;

  const speedRows: CardRow[] = [
    {
      name: 'Pause → final text',
      tip: 'From end of speech to final text on screen. Good: < 400 ms',
      value: fmtMs(p2t50, pct(p2t, 0.95)),
      colour: judge(p2t50, 400, 800),
    },
    {
      name: 'Speech → first preview',
      tip: 'Start of speech to first live preview. Good: < 900 ms',
      value: fmtMs(fp50, pct(fp, 0.95)),
      colour: judge(fp50, 900, 1500),
    },
    {
      name: 'Recognition time',
      tip: 'Speech engine processing time per phrase',
      value: fmtMs(rec50, pct(rec, 0.95)),
      colour: judge(rec50, 300, 600),
    },
    {
      name: 'Translation time',
      tip: 'Translation API / cache latency per phrase',
      value: fmtMs(mt50, pct(mt, 0.95)),
      colour: judge(mt50, 150, 400),
    },
    {
      name: 'Early transcription used',
      tip: 'Phrases finalized immediately with zero extra wait',
      value: earlyRate === null ? '-' : `${Math.round(earlyRate)} %`,
      colour: earlyRate === null ? MUTED : earlyRate >= 50 ? OK : WARN,
    },
  ];

  const rank: Record<string, number> = { [OK]: 0, [MUTED]: 0, [WARN]: 1, [BAD]: 2 };
  const perChannelRow = (
    name: string,
    tip: string,
    fn: (idx: number) => { text: string; colour?: string },
  ): CardRow => {
    const items = settings.channels.map((s, idx) => ({ name: s.name, ...fn(idx) }));
    const worst = items
      .map((i) => i.colour)
      .filter((c): c is string => Boolean(c))
      .sort((a, b) => (rank[b] ?? 0) - (rank[a] ?? 0))[0];
    return {
      name,
      tip,
      value: items.map((i) => `${i.name} ${i.text}`).join('  ·  '),
      colour: worst,
    };
  };

  const audioRows: CardRow[] = [
    perChannelRow('Level now', 'Current input level in dBFS', (idx) => {
      const lv = channels[idx].levels;
      return { text: lv.length ? `${lv[lv.length - 1].toFixed(1)} dBFS` : '-' };
    }),
    perChannelRow('Level (last 10 s)', 'Median level; speech is best around -35 to -12 dBFS', (idx) => {
      const med = pct(channels[idx].levels, 0.5);
      return med === null
        ? { text: '-' }
        : { text: `${med.toFixed(1)} dBFS`, colour: med >= -40 && med <= -10 ? OK : WARN };
    }),
    perChannelRow('Speech share (10 s)', 'Percentage of recent audio blocks classified as speech', (idx) => {
      const sf = channels[idx].speechFlags;
      return {
        text: sf.length ? `${Math.round((100 * sf.filter(Boolean).length) / sf.length)} %` : '-',
      };
    }),
    perChannelRow('Onset delay', 'Median time from sound onset to speech trigger', (idx) => {
      const med = pct(
        phrases.filter((p) => p.channel === idx).map((p) => p.onsetMs),
        0.5,
      );
      return { text: med === null ? '-' : `${Math.round(med)} ms`, colour: judge(med, 150, 400) };
    }),
    perChannelRow('Sensitivity', 'Slider value and VAD probability threshold', (idx) => {
      const sens = settings.channels[idx].sensitivity;
      return {
        text: `${Math.round(sens * 100)} % (threshold ${vadThresholdFor(sens).toFixed(2)})`,
      };
    }),
    perChannelRow('Loudest peak', 'Close to 0 dBFS means input clipping', (idx) => {
      const pk = channels[idx].peakRecent;
      return pk > -99
        ? { text: `${pk.toFixed(1)} dBFS`, colour: pk > -1 ? BAD : OK }
        : { text: '-' };
    }),
  ];

  const doubtfulPct = phrases.length
    ? (100 * phrases.filter((p) => p.doubtful).length) / phrases.length
    : null;
  const emptyPct = phrases.length
    ? (100 * phrases.filter((p) => !p.text).length) / phrases.length
    : null;

  const qualityRows: CardRow[] = [
    {
      name: 'Doubtful results',
      tip: 'Results flagged by text_filters suspiciousReasons(). Good: < 10 %',
      value: doubtfulPct === null ? '-' : `${Math.round(doubtfulPct)} %`,
      colour: judge(doubtfulPct, 10, 25),
    },
    {
      name: 'Retries used',
      tip: 'Re-checks that improved a doubtful transcription',
      value: String(phrases.filter((p) => p.retryUsed).length),
    },
    {
      name: 'Empty phrases',
      tip: 'Hallucinations or noise phrases discarded by filters',
      value: emptyPct === null ? '-' : `${Math.round(emptyPct)} %`,
      colour: judge(emptyPct, 10, 30),
    },
    {
      name: 'Context resets',
      tip: 'Times phrase history context reset after silence',
      value: String(phrases.filter((p) => p.contextReset).length),
    },
  ];

  const computerRows: CardRow[] = [
    {
      name: 'Server runtime',
      tip: 'Backend Node.js runtime environment',
      value: serverStats ? `${serverStats.nodeVersion} (${serverStats.platform})` : 'Connecting…',
    },
    {
      name: 'Server memory',
      tip: 'Express server RSS and JS heap usage',
      value: serverStats ? `${serverStats.rssMb} MB RSS / ${serverStats.heapUsedMb} MB Heap` : '-',
    },
    {
      name: 'CPU cores / load',
      tip: 'Container logical CPU count and 1m load average',
      value: serverStats ? `${serverStats.cpus} cores (load ${serverStats.loadAvg1m})` : '-',
    },
    {
      name: 'Translation cache',
      tip: 'Server LRU cache entries (max 512)',
      value: serverStats ? `${serverStats.translationCacheSize} / 512 phrases` : '-',
    },
    {
      name: 'Provider cooldowns',
      tip: 'Online translator HTTP 429 backoff status',
      value: serverStats
        ? `Google: ${
            serverStats.providers.google.coolingDownSec > 0
              ? `${serverStats.providers.google.coolingDownSec}s wait`
              : 'Ready'
          } · MyMemory: ${
            serverStats.providers.mymemory.coolingDownSec > 0
              ? `${serverStats.providers.mymemory.coolingDownSec}s wait`
              : 'Ready'
          }`
        : '-',
      colour:
        serverStats &&
        (serverStats.providers.google.coolingDownSec > 0 ||
          serverStats.providers.mymemory.coolingDownSec > 0)
          ? WARN
          : OK,
    },
  ];

  const pipelineRows: CardRow[] = [
    {
      name: 'Waiting phrases',
      tip: 'Finished phrases queued for transcription',
      value: String(waitingPhrases),
      colour: waitingPhrases < 2 ? OK : waitingPhrases < 5 ? WARN : BAD,
    },
    {
      name: 'Waiting translations',
      tip: 'Phrases currently awaiting online translation',
      value: String(waitingTranslations),
      colour: waitingTranslations < 3 ? OK : WARN,
    },
    {
      name: 'Channels active',
      tip: 'Live capture & recognition status per channel',
      value: settings.channels
        .map((s, idx) => `${s.name}: ${channels[idx].listening ? '✔' : s.muted ? 'muted' : '✖'}`)
        .join('   '),
      colour: listeningChannels.length > 0 ? OK : WARN,
    },
    {
      name: 'Watchdog restarts',
      tip: 'Times a channel pipeline restarted automatically',
      value: String(watchdogRestarts),
      colour: watchdogRestarts === 0 ? OK : WARN,
    },
    {
      name: 'Phrases this run',
      tip: 'Total completed phrases since opening the app',
      value: String(totalPhrasesRun),
    },
  ];

  const recentRows = [...phrases].reverse().slice(0, 20);

  return (
    <div className="flex h-full flex-col gap-3 overflow-y-auto px-6 py-4">
      <div className="flex flex-wrap items-center gap-2">
        <h1 className="text-2xl font-semibold text-white">Diagnostics</h1>
        <div className="flex-1" />
        <button
          type="button"
          onClick={onRefreshStats}
          className="inline-flex items-center gap-1.5 rounded-md border border-white/15 bg-[#2d2d30] px-3 py-1.5 text-xs font-medium text-gray-200 hover:bg-[#38383c] cursor-pointer"
        >
          <RefreshCw className="h-3.5 w-3.5" />
          <span>Refresh stats</span>
        </button>
        <button
          type="button"
          onClick={onCopyDiagnostics}
          className="inline-flex items-center gap-1.5 rounded-md border border-white/15 bg-[#2d2d30] px-3 py-1.5 text-xs font-medium text-gray-200 hover:bg-[#38383c] cursor-pointer"
        >
          <Copy className="h-3.5 w-3.5" />
          <span>Copy</span>
        </button>
        <button
          type="button"
          onClick={onOpenSpeedReport}
          className="inline-flex items-center gap-1.5 rounded-md bg-[#0078D4] px-3 py-1.5 text-xs font-medium text-white hover:bg-[#106ebe] cursor-pointer"
        >
          <Gauge className="h-3.5 w-3.5" />
          <span>Speed report</span>
        </button>
      </div>

      <MetricCard title="Health" rows={healthRows} />

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        <MetricCard title="Speed (last 50 phrases: median / slowest 5 %)" rows={speedRows} />
        <MetricCard title="Audio" rows={audioRows} />
        <MetricCard title="Recognition quality (last 50 phrases)" rows={qualityRows} />
        <MetricCard title="Computer & Translation Backend" rows={computerRows} />
        <MetricCard title="Pipeline" rows={pipelineRows} />

        <div className="rounded-lg border border-white/10 bg-[#262626] overflow-hidden flex flex-col">
          <div className="border-b border-white/10 bg-[#2d2d30] px-4 py-2.5 text-sm font-semibold text-gray-100">
            Recent issues (warnings and errors)
          </div>
          <div className="max-h-[210px] flex-1 overflow-y-auto p-3 font-mono text-xs text-gray-300 whitespace-pre-wrap">
            {groupedIssues.length === 0
              ? 'No warnings or errors since the app opened.'
              : groupedIssues
                  .map((g) => {
                    const times =
                      g.count > 1 ? `${g.count} times, ${g.first}-${g.last}` : `at ${g.last}`;
                    return [
                      `${g.level}: ${g.issue.title} (${times})`,
                      `  What it means: ${g.issue.meaning}`,
                      `  What to do:    ${g.issue.action}`,
                      g.issue.technical ? `  Technical:     ${g.issue.technical}` : '',
                    ]
                      .filter(Boolean)
                      .join('\n');
                  })
                  .join('\n\n')}
          </div>
        </div>
      </div>

      <div className="rounded-lg border border-white/10 bg-[#262626] overflow-hidden">
        <div className="border-b border-white/10 bg-[#2d2d30] px-4 py-2.5 text-sm font-semibold text-gray-100">
          Recent phrases
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="border-b border-white/10 bg-[#222222] text-gray-400">
              <tr>
                <th className="px-3 py-2 font-medium">Time</th>
                <th className="px-3 py-2 font-medium">Who</th>
                <th className="px-3 py-2 font-medium">Model</th>
                <th className="px-3 py-2 font-medium">Pause→text</th>
                <th className="px-3 py-2 font-medium">1st preview</th>
                <th className="px-3 py-2 font-medium">Recognition</th>
                <th className="px-3 py-2 font-medium">Early</th>
                <th className="px-3 py-2 font-medium">Text</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {recentRows.length === 0 ? (
                <tr>
                  <td colSpan={8} className="px-3 py-6 text-center text-gray-500">
                    No phrases processed yet in this session.
                  </td>
                </tr>
              ) : (
                recentRows.map((p) => (
                  <tr key={p.key} className="hover:bg-white/[0.03]">
                    <td className="px-3 py-2 font-mono text-gray-400 whitespace-nowrap">{p.time}</td>
                    <td
                      className="px-3 py-2 font-semibold whitespace-nowrap"
                      style={{ color: CHANNEL_COLOURS[p.channel % CHANNEL_COLOURS.length] }}
                    >
                      {p.who}
                    </td>
                    <td className="px-3 py-2 text-gray-300 whitespace-nowrap">{p.model}</td>
                    <td
                      className="px-3 py-2 font-mono whitespace-nowrap"
                      style={{ color: judge(p.pauseToTextMs, 400, 800) }}
                    >
                      {Math.round(p.pauseToTextMs)} ms
                    </td>
                    <td className="px-3 py-2 font-mono text-gray-300 whitespace-nowrap">
                      {p.firstPreviewMs !== null ? `${Math.round(p.firstPreviewMs)} ms` : '-'}
                    </td>
                    <td className="px-3 py-2 font-mono text-gray-300 whitespace-nowrap">
                      {p.processMs !== null ? `${Math.round(p.processMs)} ms` : '-'}
                    </td>
                    <td className="px-3 py-2 text-gray-300 whitespace-nowrap">
                      {p.early ? 'yes' : 'no'}
                    </td>
                    <td className="px-3 py-2 text-gray-200 max-w-md truncate" title={p.text}>
                      {p.text}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
