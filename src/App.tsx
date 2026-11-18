import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Eraser,
  MessageSquare,
  MessageSquareWarning,
  Settings as SettingsIcon,
  Wrench,
} from 'lucide-react';
import { ChannelStrip } from './components/ChannelStrip';
import { ConversationView } from './components/ConversationView';
import { DiagnosticsPage } from './components/DiagnosticsPage';
import { ReportMistakeModal, SpeedReportModal } from './components/Modals';
import { SettingsPage } from './components/SettingsPage';
import { SimulatorBar } from './components/SimulatorBar';
import {
  BAD,
  CHANNEL_COLOURS,
  LANGUAGES,
  LANGUAGE_NAMES,
  MAX_ENTRIES,
  MODELS,
  OK,
  SAME_TURN_SEC,
  WARN,
  vadThresholdFor,
} from './lib/config';
import { enumerateWebAudioDevices, WebChannelPipeline } from './lib/audioPipeline';
import { buildSpeedReportText, explainIssue } from './lib/diagnostics';
import { loadSettings, saveSettings } from './lib/settings';
import {
  collapseRepetitions,
  continuationJoin,
  hallucinationReason,
  suspiciousReasons,
  trailingMarksLength,
} from './lib/textFilters';
import type {
  AppSettings,
  AudioDeviceInfo,
  ChannelRuntimeState,
  ChannelSettings,
  ConversationEntry,
  IssueEntry,
  PhraseMetric,
  ServerStats,
} from './lib/types';

type PageKey = 'interpreter' | 'diagnostics' | 'settings';

interface ToastMessage {
  id: number;
  title: string;
  body: string;
  tone: 'info' | 'success' | 'error';
}

function makeInitialRuntime(settings: ChannelSettings): ChannelRuntimeState {
  return {
    listening: false,
    problem: null,
    statusText: settings.muted ? 'Muted - not listening.' : 'Starting…',
    statusColour: settings.muted ? BAD : null,
    rmsDb: -100,
    peakDb: -100,
    speaking: false,
    speechProb: 0,
    threshold: vadThresholdFor(settings.sensitivity),
    levels: [],
    speechFlags: [],
    peakRecent: -100,
    musicDetected: false,
    failures: 0,
  };
}

function formatClockTime(): string {
  const d = new Date();
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
}

function formatHms(): string {
  const d = new Date();
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}:${String(
    d.getSeconds(),
  ).padStart(2, '0')}`;
}

export function App() {
  const [page, setPage] = useState<PageKey>('interpreter');
  const [settings, setSettings] = useState<AppSettings>(() => loadSettings());
  const settingsRef = useRef(settings);
  settingsRef.current = settings;

  const [inputDevices, setInputDevices] = useState<AudioDeviceInfo[]>([]);
  const [outputDevices, setOutputDevices] = useState<AudioDeviceInfo[]>([]);
  const devicesRef = useRef<AudioDeviceInfo[]>([]);

  const [runtime, setRuntime] = useState<[ChannelRuntimeState, ChannelRuntimeState]>(() => [
    makeInitialRuntime(settings.channels[0]),
    makeInitialRuntime(settings.channels[1]),
  ]);

  const [entries, setEntries] = useState<ConversationEntry[]>([]);
  const [phrases, setPhrases] = useState<PhraseMetric[]>([]);
  const [totalPhrasesRun, setTotalPhrasesRun] = useState(0);
  const [waitingTranslations, setWaitingTranslations] = useState(0);
  const [issues, setIssues] = useState<IssueEntry[]>([]);
  const [serverStats, setServerStats] = useState<ServerStats | null>(null);
  const [statusLine, setStatusLine] = useState('Ready.');
  const [listeningSinceMs, setListeningSinceMs] = useState<number | null>(null);
  const [watchdogRestarts] = useState(0);

  const [showReportModal, setShowReportModal] = useState(false);
  const [speedReportText, setSpeedReportText] = useState<string | null>(null);
  const [toasts, setToasts] = useState<ToastMessage[]>([]);

  const startedAtIsoRef = useRef(new Date().toISOString());
  const pipelinesRef = useRef<[WebChannelPipeline | null, WebChannelPipeline | null]>([null, null]);
  const runIdsRef = useRef<[number, number]>([0, 0]);
  const firstPreviewRef = useRef<Map<string, number>>(new Map());
  const simSeqRef = useRef(10_000);

  const pushToast = useCallback(
    (title: string, body: string, tone: ToastMessage['tone'] = 'info') => {
      const id = Date.now() + Math.floor(Math.random() * 1000);
      setToasts((prev) => [...prev.slice(-3), { id, title, body, tone }]);
      window.setTimeout(() => {
        setToasts((prev) => prev.filter((t) => t.id !== id));
      }, 4000);
    },
    [],
  );

  const recordIssue = useCallback((event: string, technical: string, channel?: string) => {
    const explained = explainIssue(event, technical, channel);
    const entry: IssueEntry = {
      id: `${Date.now()}-${Math.random()}`,
      time: formatHms(),
      ...explained,
    };
    setIssues((prev) => [...prev.slice(-39), entry]);
    fetch('/api/logs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        level: explained.level,
        event,
        channel,
        fields: { technical },
      }),
    }).catch(() => {});
  }, []);

  const speakTranslation = useCallback((channelIndex: number, text: string) => {
    const ch = settingsRef.current.channels[channelIndex];
    if (!ch.speak || !text || !('speechSynthesis' in window)) return;

    const utter = new SpeechSynthesisUtterance(text);
    const tgtEntry = LANGUAGES[ch.tgt];
    if (tgtEntry) {
      utter.lang = tgtEntry.speechCode;
    }

    // Mute loopback pipelines while TTS speaks to avoid self-translation feedback
    utter.onstart = () => {
      settingsRef.current.channels.forEach((c, idx) => {
        if (c.device.toLowerCase().includes('[loopback]')) {
          pipelinesRef.current[idx]?.setTtsMuted(true);
        }
      });
    };
    utter.onend = () => {
      pipelinesRef.current.forEach((p) => p?.setTtsMuted(false));
    };
    utter.onerror = () => {
      pipelinesRef.current.forEach((p) => p?.setTtsMuted(false));
    };

    window.speechSynthesis.speak(utter);
  }, []);

  const translateAndDisplay = useCallback(
    async (
      key: string,
      channelIndex: number,
      heardText: string,
      opts: {
        tStartMs: number;
        tEndMs: number;
        onsetMs: number | null;
        firstPreviewMs: number | null;
        early: boolean;
      },
    ) => {
      const s = settingsRef.current;
      const ch = s.channels[channelIndex];
      const cleanedHeard = collapseRepetitions(heardText.trim());
      const halluReason = hallucinationReason(cleanedHeard);

      if (halluReason) {
        // Drop hallucinated phrases from conversation and record empty phrase metric
        setEntries((prev) => prev.filter((e) => e.key !== key));
        setStatusLine(`Filtered hallucination (${halluReason}): "${cleanedHeard}"`);
        return;
      }

      const speechSec = Math.max(0.2, (opts.tEndMs - opts.tStartMs) / 1000);
      const reasons = suspiciousReasons(cleanedHeard, speechSec);

      // Commit heard text immediately (translation pending)
      setEntries((prev) => {
        const idx = prev.findIndex((e) => e.key === key);
        const nowSec = performance.now() / 1000;

        if (idx >= 0) {
          const updated = [...prev];
          updated[idx] = {
            ...updated[idx],
            heard: cleanedHeard,
            final: true,
          };
          return updated;
        }

        // Check continuation join with previous final entry from same channel
        const lastSameIdx = [...prev]
          .reverse()
          .findIndex((e) => e.channel === channelIndex && e.final);
        if (lastSameIdx >= 0) {
          const realIdx = prev.length - 1 - lastSameIdx;
          const prevEntry = prev[realIdx];
          const joinRes = continuationJoin(prevEntry.heard, cleanedHeard);
          if (joinRes.joined) {
            const cut = trailingMarksLength(prevEntry.heard);
            const mergedHeard =
              prevEntry.heard.slice(0, prevEntry.heard.length - cut) + ' ' + joinRes.text;
            const updated = [...prev];
            updated[realIdx] = {
              ...prevEntry,
              heard: mergedHeard,
            };
            return updated;
          }
        }

        const prevLast = prev[prev.length - 1];
        const showLabel =
          !prevLast || prevLast.channel !== channelIndex || nowSec - prevLast.t > SAME_TURN_SEC;

        const nextEntry: ConversationEntry = {
          key,
          channel: channelIndex,
          time: formatClockTime(),
          t: nowSec,
          heard: cleanedHeard,
          stable: cleanedHeard.split(/\s+/).length,
          final: true,
          translation: null,
          translationFinal: false,
          showLabel,
        };
        const merged = [...prev, nextEntry];
        return merged.length > MAX_ENTRIES ? merged.slice(merged.length - MAX_ENTRIES) : merged;
      });

      const srcCode = LANGUAGES[ch.src]?.translateCode ?? 'auto';
      const tgtCode = LANGUAGES[ch.tgt]?.translateCode ?? 'en';

      setWaitingTranslations((n) => n + 1);
      const tTranslateStart = performance.now();

      try {
        const res = await fetch('/api/translate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text: cleanedHeard,
            src: srcCode,
            tgt: tgtCode,
          }),
        });
        const data = (await res.json()) as {
          translated?: string;
          provider?: string;
          elapsedMs?: number;
          detectedSrc?: string;
          error?: string;
        };

        const translatedText = data.translated || `[translation failed: ${cleanedHeard}]`;
        const mtMs = Math.round((performance.now() - tTranslateStart) * 10) / 10;
        const pauseToTextMs = Math.max(
          25,
          Math.round(performance.now() - opts.tEndMs),
        );
        const processMs = Math.max(20, Math.round(opts.tEndMs - opts.tStartMs));

        const detectedName =
          data.detectedSrc && srcCode === 'auto'
            ? LANGUAGE_NAMES[data.detectedSrc.split('-')[0]] || data.detectedSrc
            : undefined;

        setEntries((prev) =>
          prev.map((e) =>
            e.key === key
              ? {
                  ...e,
                  translation: translatedText,
                  translationFinal: true,
                  detectedLang: detectedName,
                }
              : e,
          ),
        );

        if (res.ok && data.translated) {
          speakTranslation(channelIndex, data.translated);
        } else if (data.error) {
          recordIssue('translation_failed', data.error, ch.name);
        }

        const metric: PhraseMetric = {
          key,
          time: formatHms(),
          who: ch.name,
          channel: channelIndex,
          model: MODELS[s.model] ?? s.model,
          pauseToTextMs,
          firstPreviewMs: opts.firstPreviewMs,
          processMs,
          mtMs,
          early: opts.early,
          doubtful: reasons.length > 0,
          retryUsed: false,
          contextReset: false,
          onsetMs: opts.onsetMs,
          text: cleanedHeard,
          translation: translatedText,
          srcLang: ch.src,
          tgtLang: ch.tgt,
          provider: data.provider,
        };
        setPhrases((prev) => [...prev.slice(-49), metric]);
        setTotalPhrasesRun((c) => c + 1);
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        recordIssue('translation_failed', msg, ch.name);
        setEntries((prev) =>
          prev.map((e) =>
            e.key === key
              ? {
                  ...e,
                  translation: `[translation failed: ${cleanedHeard}]`,
                  translationFinal: true,
                }
              : e,
          ),
        );
      } finally {
        setWaitingTranslations((n) => Math.max(0, n - 1));
      }
    },
    [recordIssue, speakTranslation],
  );

  const stopChannel = useCallback((idx: number) => {
    const existing = pipelinesRef.current[idx];
    if (existing) {
      existing.stop();
      pipelinesRef.current[idx] = null;
    }
    setEntries((prev) => prev.filter((e) => e.channel !== idx || e.final));
    setRuntime((prev) => {
      const next: [ChannelRuntimeState, ChannelRuntimeState] = [...prev];
      const muted = settingsRef.current.channels[idx].muted;
      next[idx] = {
        ...next[idx],
        listening: false,
        rmsDb: -100,
        peakDb: -100,
        speaking: false,
        speechProb: 0,
        statusText: muted ? 'Muted - not listening.' : 'Stopped.',
        statusColour: muted ? BAD : null,
      };
      return next;
    });
  }, []);

  const startChannel = useCallback(
    async (idx: number, overrideSettings?: ChannelSettings) => {
      const ch = overrideSettings ?? settingsRef.current.channels[idx];
      if (ch.muted) {
        stopChannel(idx);
        return;
      }

      // Stop previous pipeline on this channel
      pipelinesRef.current[idx]?.stop();
      runIdsRef.current[idx] += 1;
      const runId = runIdsRef.current[idx];

      const pipeline = new WebChannelPipeline(
        ch.device,
        devicesRef.current,
        ch.src,
        ch.sensitivity,
        ch.replay,
        {
          onLevel: (rmsDb, peakDb, speaking, prob, threshold) => {
            if (runIdsRef.current[idx] !== runId) return;
            setRuntime((prev) => {
              const next: [ChannelRuntimeState, ChannelRuntimeState] = [...prev];
              const cur = next[idx];
              const levels = rmsDb > -99 ? [...cur.levels.slice(-199), rmsDb] : cur.levels;
              const speechFlags = [...cur.speechFlags.slice(-199), speaking];
              const peakRecent =
                rmsDb > -99 ? Math.max(peakDb, cur.peakRecent - 0.05) : cur.peakRecent;
              next[idx] = {
                ...cur,
                rmsDb,
                peakDb,
                speaking,
                speechProb: prob,
                threshold,
                levels,
                speechFlags,
                peakRecent,
              };
              return next;
            });
          },
          onStatus: (text, colour) => {
            if (runIdsRef.current[idx] !== runId) return;
            setRuntime((prev) => {
              const next: [ChannelRuntimeState, ChannelRuntimeState] = [...prev];
              next[idx] = {
                ...next[idx],
                statusText: text,
                statusColour: colour ?? (text.startsWith('⚠') ? WARN : null),
              };
              return next;
            });
          },
          onPartial: (seq, heard, tStartMs) => {
            if (runIdsRef.current[idx] !== runId) return;
            if (settingsRef.current.previewModel === 'Off (final text only)') return;
            const key = `${idx}-${runId}-${seq}`;
            const nowMs = performance.now();
            if (!firstPreviewRef.current.has(key)) {
              firstPreviewRef.current.set(key, Math.round(nowMs - tStartMs));
            }
            setEntries((prev) => {
              const existingIdx = prev.findIndex((e) => e.key === key);
              if (existingIdx >= 0) {
                const existing = prev[existingIdx];
                if (existing.final) return prev;
                const oldWords = existing.heard.split(/\s+/).filter(Boolean);
                const newWords = heard.split(/\s+/).filter(Boolean);
                let stable = 0;
                while (
                  stable < Math.min(oldWords.length, newWords.length) &&
                  oldWords[stable] === newWords[stable]
                ) {
                  stable += 1;
                }
                const updated = [...prev];
                updated[existingIdx] = {
                  ...existing,
                  heard,
                  stable,
                };
                return updated;
              }
              const nowSec = nowMs / 1000;
              const prevLast = prev[prev.length - 1];
              const showLabel =
                !prevLast || prevLast.channel !== idx || nowSec - prevLast.t > SAME_TURN_SEC;
              const entry: ConversationEntry = {
                key,
                channel: idx,
                time: formatClockTime(),
                t: nowSec,
                heard,
                stable: 0,
                final: false,
                translation: null,
                translationFinal: false,
                showLabel,
              };
              return [...prev, entry];
            });
          },
          onFinal: (seq, text, tStartMs, tEndMs, onsetMs) => {
            if (runIdsRef.current[idx] !== runId) return;
            const key = `${idx}-${runId}-${seq}`;
            const fp = firstPreviewRef.current.get(key) ?? null;
            firstPreviewRef.current.delete(key);
            void translateAndDisplay(key, idx, text, {
              tStartMs,
              tEndMs,
              onsetMs,
              firstPreviewMs: fp,
              early: true,
            });
          },
          onIssue: (event, technical) => {
            recordIssue(event, technical, ch.name);
          },
        },
      );

      pipelinesRef.current[idx] = pipeline;
      const ok = await pipeline.start();
      if (runIdsRef.current[idx] !== runId) {
        pipeline.stop();
        return;
      }

      setRuntime((prev) => {
        const next: [ChannelRuntimeState, ChannelRuntimeState] = [...prev];
        next[idx] = {
          ...next[idx],
          listening: ok,
          problem: ok ? null : 'Could not open audio input',
        };
        return next;
      });
      if (ok) {
        setListeningSinceMs((prev) => prev ?? Date.now());
        setStatusLine('Listening…');
      }
    },
    [recordIssue, stopChannel, translateAndDisplay],
  );

  // Initial device scan + auto-start non-muted channels
  useEffect(() => {
    let mounted = true;
    const init = async () => {
      const { inputs, outputs } = await enumerateWebAudioDevices();
      if (!mounted) return;
      setInputDevices(inputs);
      setOutputDevices(outputs);
      devicesRef.current = inputs;

      for (let i = 0; i < 2; i++) {
        if (!settingsRef.current.channels[i].muted) {
          void startChannel(i, settingsRef.current.channels[i]);
        }
      }
    };
    void init();

    return () => {
      mounted = false;
      pipelinesRef.current.forEach((p) => p?.stop());
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Poll server stats every 4s while Diagnostics page is active (or once on boot)
  const fetchServerStats = useCallback(async () => {
    try {
      const res = await fetch('/api/stats');
      if (res.ok) {
        const data = (await res.json()) as ServerStats;
        setServerStats(data);
      }
    } catch {
      // Ignore transient network errors
    }
  }, []);

  useEffect(() => {
    void fetchServerStats();
    const timer = window.setInterval(() => {
      void fetchServerStats();
    }, 4000);
    return () => window.clearInterval(timer);
  }, [fetchServerStats]);

  // Keep listeningSinceMs accurate
  const anyListening = runtime.some((r) => r.listening);
  useEffect(() => {
    if (!anyListening) {
      setListeningSinceMs(null);
    } else {
      setListeningSinceMs((prev) => prev ?? Date.now());
    }
  }, [anyListening]);

  const handleRescanDevices = async () => {
    try {
      if (navigator.mediaDevices?.getUserMedia) {
        const tmp = await navigator.mediaDevices.getUserMedia({ audio: true });
        tmp.getTracks().forEach((t) => t.stop());
      }
    } catch {
      // Permission might already be denied or unavailable
    }
    const { inputs, outputs } = await enumerateWebAudioDevices();
    setInputDevices(inputs);
    setOutputDevices(outputs);
    devicesRef.current = inputs;
    setStatusLine(`Refreshed - ${inputs.length} in / ${outputs.length} out device(s).`);
    pushToast('Devices refreshed', `Found ${inputs.length} input(s) and ${outputs.length} output(s).`, 'info');
    settingsRef.current.channels.forEach((ch, idx) => {
      if (!ch.muted) void startChannel(idx, ch);
    });
  };

  const updateChannelSettings = (idx: number, patch: Partial<ChannelSettings>) => {
    const nextChannels: [ChannelSettings, ChannelSettings] = [
      { ...settings.channels[0] },
      { ...settings.channels[1] },
    ];
    const prevCh = nextChannels[idx];
    const updatedCh = { ...prevCh, ...patch };
    nextChannels[idx] = updatedCh;

    const nextSettings: AppSettings = { ...settings, channels: nextChannels };
    setSettings(nextSettings);
    saveSettings(nextSettings);

    if ('sensitivity' in patch && patch.sensitivity !== undefined) {
      pipelinesRef.current[idx]?.setSensitivity(patch.sensitivity);
      const thr = vadThresholdFor(patch.sensitivity);
      setRuntime((prev) => {
        const next: [ChannelRuntimeState, ChannelRuntimeState] = [...prev];
        next[idx] = { ...next[idx], threshold: thr };
        return next;
      });
    }

    if ('speak' in patch && !patch.speak && 'speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }

    const needsRestart =
      'device' in patch ||
      'src' in patch ||
      'tgt' in patch ||
      'muted' in patch ||
      'replay' in patch ||
      'output' in patch;

    if (needsRestart) {
      if (updatedCh.muted) {
        stopChannel(idx);
      } else {
        void startChannel(idx, updatedCh);
      }
    }
  };

  const updateSharedSettings = (
    patch: Partial<Pick<AppSettings, 'model' | 'previewModel' | 'autoLanguage'>>,
  ) => {
    const next: AppSettings = { ...settings, ...patch };
    setSettings(next);
    saveSettings(next);
    setStatusLine('Settings updated.');
  };

  const changeTextSize = (step: number) => {
    const nextSize = Math.max(9, Math.min(28, settings.textSize + step));
    const next = { ...settings, textSize: nextSize };
    setSettings(next);
    saveSettings(next);
  };

  const handleSimulatePhrase = async (channelIdx: number, rawText: string) => {
    simSeqRef.current += 1;
    const seq = simSeqRef.current;
    const runId = runIdsRef.current[channelIdx];
    const key = `sim-${channelIdx}-${runId}-${seq}`;
    const words = rawText.trim().split(/\s+/).filter(Boolean);
    const tStartMs = performance.now();

    // Briefly stream a live grey preview before finalizing if previews are enabled
    if (settings.previewModel !== 'Off (final text only)' && words.length > 2) {
      const halfText = words.slice(0, Math.ceil(words.length / 2)).join(' ');
      setEntries((prev) => {
        const nowSec = performance.now() / 1000;
        const prevLast = prev[prev.length - 1];
        const showLabel =
          !prevLast || prevLast.channel !== channelIdx || nowSec - prevLast.t > SAME_TURN_SEC;
        return [
          ...prev,
          {
            key,
            channel: channelIdx,
            time: formatClockTime(),
            t: nowSec,
            heard: halfText,
            stable: Math.max(0, Math.floor(words.length / 3)),
            final: false,
            translation: null,
            translationFinal: false,
            showLabel,
          },
        ];
      });
      await new Promise((r) => setTimeout(r, 160));
    }

    const tEndMs = performance.now();
    await translateAndDisplay(key, channelIdx, rawText, {
      tStartMs,
      tEndMs,
      onsetMs: 38,
      firstPreviewMs: settings.previewModel === 'Off (final text only)' ? null : 160,
      early: true,
    });
  };

  const handleSubmitMistake = async (payload: {
    phraseKey: string;
    problems: string[];
    correctText: string;
    actualLanguage: string | null;
    note: string;
  }) => {
    const phrase = phrases.find((p) => p.key === payload.phraseKey);
    if (!phrase) return;
    try {
      await fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          channel: phrase.who,
          text: phrase.text,
          translation: phrase.translation,
          srcLang: phrase.srcLang,
          tgtLang: phrase.tgtLang,
          problems: payload.problems,
          correctText: payload.correctText,
          actualLanguage: payload.actualLanguage,
          note: payload.note,
        }),
      });
      setShowReportModal(false);
      pushToast(
        'Thank you, saved',
        'Mistake report saved for diagnostics and simulator replay.',
        'success',
      );
    } catch {
      pushToast('Save failed', 'Could not submit mistake report.', 'error');
    }
  };

  const handleCopyDiagnostics = async () => {
    const text = buildSpeedReportText({
      startedAtIso: startedAtIsoRef.current,
      settings,
      phrases,
      channels: runtime,
      issues,
      serverStats,
      totalPhrases: totalPhrasesRun,
    });
    try {
      await navigator.clipboard.writeText(text);
      pushToast('Copied', 'Diagnostics copied to the clipboard', 'success');
    } catch {
      pushToast('Clipboard unavailable', 'Use Speed report to view and copy manually.', 'info');
    }
  };

  const handleOpenSpeedReport = () => {
    const text = buildSpeedReportText({
      startedAtIso: startedAtIsoRef.current,
      settings,
      phrases,
      channels: runtime,
      issues,
      serverStats,
      totalPhrases: totalPhrasesRun,
    });
    setSpeedReportText(text);
  };

  const allMuted = settings.channels.every((c) => c.muted);
  const badgeLabel = anyListening ? 'LISTENING' : allMuted ? 'MUTED' : 'IDLE';
  const badgeColour = anyListening ? OK : allMuted ? BAD : WARN;

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#1c1c1c] text-gray-100">
      {/* Fluent Left Navigation Sidebar */}
      <aside className="flex w-52 shrink-0 flex-col justify-between border-r border-white/10 bg-[#202020] p-2">
        <div className="flex flex-col gap-1">
          <div className="px-3 py-2.5 text-xs font-semibold uppercase tracking-wider text-gray-400">
            Voice Interpreter
          </div>

          <button
            type="button"
            onClick={() => setPage('interpreter')}
            className={`flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors cursor-pointer ${
              page === 'interpreter'
                ? 'bg-[#2d2d30] text-white border-l-2 border-[#0078D4]'
                : 'text-gray-300 hover:bg-white/5'
            }`}
          >
            <MessageSquare className="h-4 w-4 shrink-0" />
            <span>Interpreter</span>
          </button>

          <button
            type="button"
            onClick={() => setPage('diagnostics')}
            className={`flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors cursor-pointer ${
              page === 'diagnostics'
                ? 'bg-[#2d2d30] text-white border-l-2 border-[#0078D4]'
                : 'text-gray-300 hover:bg-white/5'
            }`}
          >
            <Wrench className="h-4 w-4 shrink-0" />
            <span>Diagnostics</span>
          </button>
        </div>

        <button
          type="button"
          onClick={() => setPage('settings')}
          className={`flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors cursor-pointer ${
            page === 'settings'
              ? 'bg-[#2d2d30] text-white border-l-2 border-[#0078D4]'
              : 'text-gray-300 hover:bg-white/5'
          }`}
        >
          <SettingsIcon className="h-4 w-4 shrink-0" />
          <span>Settings</span>
        </button>
      </aside>

      {/* Main Content Area */}
      <main className="relative flex min-w-0 flex-1 flex-col overflow-hidden">
        {page === 'interpreter' && (
          <div className="flex h-full flex-col gap-2.5 px-6 py-4">
            {/* Header */}
            <header className="flex flex-wrap items-center gap-2.5">
              <h1 className="text-2xl font-semibold text-white">Voice Interpreter</h1>
              <span className="rounded-md bg-[#2980B9] px-2 py-0.5 text-[11px] font-bold tracking-wide text-white">
                WEB
              </span>
              <span
                className="rounded-md px-2 py-0.5 text-[11px] font-bold tracking-wide text-white"
                style={{ backgroundColor: badgeColour }}
              >
                {badgeLabel}
              </span>
              <div className="flex-1" />
              <button
                type="button"
                onClick={() => {
                  setEntries([]);
                  setStatusLine('Cleared.');
                }}
                className="inline-flex items-center gap-1.5 rounded-md border border-white/15 bg-[#2d2d30] px-3 py-1.5 text-xs font-medium text-gray-200 hover:bg-[#38383c] cursor-pointer"
              >
                <Eraser className="h-3.5 w-3.5" />
                <span>Clear</span>
              </button>
              <button
                type="button"
                onClick={() => {
                  if (phrases.length === 0) {
                    pushToast('Nothing to report yet', 'No phrases have been shown yet.', 'info');
                  } else {
                    setShowReportModal(true);
                  }
                }}
                title="Mark a recent phrase as wrong for diagnostics and replay"
                className="inline-flex items-center gap-1.5 rounded-md border border-white/15 bg-[#2d2d30] px-3 py-1.5 text-xs font-medium text-gray-200 hover:bg-[#38383c] cursor-pointer"
              >
                <MessageSquareWarning className="h-3.5 w-3.5" />
                <span>Report a mistake</span>
              </button>
            </header>

            {/* Two Channel Strips */}
            <section className="flex flex-col gap-2.5 lg:flex-row">
              {settings.channels.map((ch, idx) => (
                <ChannelStrip
                  key={idx}
                  colour={CHANNEL_COLOURS[idx % CHANNEL_COLOURS.length]}
                  settings={ch}
                  runtime={runtime[idx]}
                  onToggleSpeak={() => updateChannelSettings(idx, { speak: !ch.speak })}
                  onToggleMute={() => updateChannelSettings(idx, { muted: !ch.muted })}
                  onOpenSettings={() => setPage('settings')}
                />
              ))}
            </section>

            {/* Conversation Header + Text Size Controls */}
            <div className="flex items-center gap-2 pt-0.5">
              <span className="text-sm font-semibold text-gray-200">Conversation</span>
              <div className="flex-1" />
              <span className="text-xs text-gray-400">Text size</span>
              <button
                type="button"
                onClick={() => changeTextSize(-1)}
                title="Smaller conversation text"
                className="w-11 rounded border border-white/15 bg-[#2d2d30] py-1 text-xs font-medium text-gray-200 hover:bg-[#38383c] cursor-pointer"
              >
                A−
              </button>
              <button
                type="button"
                onClick={() => changeTextSize(1)}
                title="Larger conversation text"
                className="w-11 rounded border border-white/15 bg-[#2d2d30] py-1 text-xs font-medium text-gray-200 hover:bg-[#38383c] cursor-pointer"
              >
                A+
              </button>
            </div>

            {/* Unified Two-Channel Conversation */}
            <ConversationView
              entries={entries}
              channelNames={settings.channels.map((c) => c.name)}
              textSizePt={settings.textSize}
            />

            {/* Simulator / Test Phrase Bar */}
            <SimulatorBar
              channels={settings.channels}
              onSimulatePhrase={(chIdx, text) => void handleSimulatePhrase(chIdx, text)}
            />

            {/* Status Line */}
            <footer className="text-xs text-gray-400">{statusLine}</footer>
          </div>
        )}

        {page === 'diagnostics' && (
          <DiagnosticsPage
            settings={settings}
            channels={runtime}
            phrases={phrases}
            issues={issues}
            serverStats={serverStats}
            listeningSinceMs={listeningSinceMs}
            watchdogRestarts={watchdogRestarts}
            waitingPhrases={0}
            waitingTranslations={waitingTranslations}
            totalPhrasesRun={totalPhrasesRun}
            onCopyDiagnostics={() => void handleCopyDiagnostics()}
            onOpenSpeedReport={handleOpenSpeedReport}
            onRefreshStats={() => void fetchServerStats()}
          />
        )}

        {page === 'settings' && (
          <SettingsPage
            settings={settings}
            inputDevices={inputDevices}
            outputDevices={outputDevices}
            onRescanDevices={() => void handleRescanDevices()}
            onUpdateChannel={updateChannelSettings}
            onUpdateShared={updateSharedSettings}
          />
        )}

        {/* Toast InfoBars */}
        {toasts.length > 0 && (
          <div className="pointer-events-none absolute right-5 top-4 z-40 flex flex-col gap-2">
            {toasts.map((t) => (
              <div
                key={t.id}
                className={`pointer-events-auto min-w-[260px] max-w-sm rounded-lg border px-3.5 py-2.5 text-xs shadow-xl ${
                  t.tone === 'success'
                    ? 'border-[#2ECC71]/40 bg-[#1f3326] text-green-100'
                    : t.tone === 'error'
                      ? 'border-[#E74C3C]/40 bg-[#361e1e] text-red-100'
                      : 'border-white/15 bg-[#2d2d30] text-gray-100'
                }`}
              >
                <div className="font-semibold">{t.title}</div>
                <div className="mt-0.5 text-gray-300">{t.body}</div>
              </div>
            ))}
          </div>
        )}
      </main>

      {showReportModal && (
        <ReportMistakeModal
          recentPhrases={phrases}
          onClose={() => setShowReportModal(false)}
          onSubmit={(payload) => void handleSubmitMistake(payload)}
        />
      )}

      {speedReportText !== null && (
        <SpeedReportModal
          reportText={speedReportText}
          onClose={() => setSpeedReportText(null)}
        />
      )}
    </div>
  );
}
export default App;
