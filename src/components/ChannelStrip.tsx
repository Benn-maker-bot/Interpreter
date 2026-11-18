import { Mic, MicOff, Settings, Volume2 } from 'lucide-react';
import { OK, plainDeviceName } from '../lib/config';
import type { ChannelRuntimeState, ChannelSettings } from '../lib/types';
import { LevelMeter } from './LevelMeter';
import { SpeechMeter } from './SpeechMeter';

interface ChannelStripProps {
  colour: string;
  settings: ChannelSettings;
  runtime: ChannelRuntimeState;
  onToggleSpeak: () => void;
  onToggleMute: () => void;
  onOpenSettings: () => void;
}

export function ChannelStrip({
  colour,
  settings,
  runtime,
  onToggleSpeak,
  onToggleMute,
  onOpenSettings,
}: ChannelStripProps) {
  const isLoopback = settings.device.toLowerCase().includes('[loopback]');
  const devIcon = isLoopback ? '🔊 ' : '🎙 ';
  const rawName = devIcon + plainDeviceName(settings.device);
  const shortDevice = rawName.length <= 54 ? rawName : `${rawName.slice(0, 52)}…`;

  const levelText = !runtime.listening
    ? '-'
    : runtime.rmsDb <= -99
      ? '-∞ dBFS'
      : `${runtime.rmsDb.toFixed(1)} dBFS`;

  const speechText = !runtime.listening
    ? '-'
    : `${Math.round(runtime.speechProb * 100)}% / ${Math.round(runtime.threshold * 100)}%${
        runtime.speaking ? ' SPEECH' : ''
      }`;

  return (
    <div className="flex flex-1 min-w-[280px] rounded-lg border border-white/10 bg-[#262626] p-2.5 shadow-sm">
      <div
        className="mr-2.5 w-1 shrink-0 rounded-full"
        style={{ backgroundColor: colour }}
      />
      <div className="flex min-w-0 flex-1 flex-col gap-1">
        <div className="flex items-center gap-1.5">
          <span className="text-sm font-semibold truncate" style={{ color: colour }}>
            {settings.name}
          </span>
          <div className="flex-1" />
          <button
            type="button"
            onClick={onToggleSpeak}
            title="Read this channel's translations aloud (Settings -> Speak and replay to)"
            className={`inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs font-medium transition-colors cursor-pointer ${
              settings.speak
                ? 'border-[#0078D4] bg-[#0078D4] text-white'
                : 'border-white/15 bg-[#323232] text-gray-200 hover:bg-[#3c3c3c]'
            }`}
          >
            <Volume2 className="h-3.5 w-3.5" />
            <span>{settings.speak ? 'Speak: On' : 'Speak'}</span>
          </button>
          <button
            type="button"
            onClick={onToggleMute}
            title="Stop listening to this channel (releases the device)"
            className={`inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs font-medium transition-colors cursor-pointer ${
              settings.muted
                ? 'border-[#0078D4] bg-[#0078D4] text-white'
                : 'border-white/15 bg-[#323232] text-gray-200 hover:bg-[#3c3c3c]'
            }`}
          >
            {settings.muted ? <Mic className="h-3.5 w-3.5" /> : <MicOff className="h-3.5 w-3.5" />}
            <span>{settings.muted ? 'Unmute' : 'Mute'}</span>
          </button>
          <button
            type="button"
            onClick={onOpenSettings}
            title="Device, languages and sensitivity (Settings page)"
            className="inline-flex items-center justify-center rounded-md p-1.5 text-gray-300 hover:bg-white/10 hover:text-white transition-colors cursor-pointer"
          >
            <Settings className="h-3.5 w-3.5" />
          </button>
        </div>

        <div
          className="truncate text-xs text-gray-400"
          title={`${settings.device}\n${settings.src} -> ${settings.tgt}`}
        >
          {shortDevice} · {settings.src} → {settings.tgt}
        </div>

        <div className="grid grid-cols-[1fr_118px] items-center gap-x-2">
          <LevelMeter
            rmsDb={runtime.rmsDb}
            peakDb={runtime.peakDb}
            active={runtime.listening}
          />
          <span className="font-mono text-[11px] text-gray-300 text-right whitespace-nowrap">
            {levelText}
          </span>

          <SpeechMeter
            prob={runtime.speechProb}
            threshold={runtime.threshold}
            speaking={runtime.speaking}
            active={runtime.listening}
          />
          <span
            className="font-mono text-[11px] text-right whitespace-nowrap"
            style={{ color: runtime.listening && runtime.speaking ? OK : '#d1d5db' }}
          >
            {speechText}
          </span>
        </div>

        <div
          className="text-xs leading-snug"
          style={{ color: runtime.statusColour ?? '#9AA0A6' }}
        >
          {runtime.statusText}
        </div>
      </div>
    </div>
  );
}
