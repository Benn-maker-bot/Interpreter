import { RefreshCw } from 'lucide-react';
import {
  CHANNEL_COLOURS,
  LANGUAGES,
  MODELS,
  PREVIEW_CHOICES,
  TARGET_LANGUAGES,
} from '../lib/config';
import type { AppSettings, AudioDeviceInfo, ChannelSettings } from '../lib/types';

interface SettingsPageProps {
  settings: AppSettings;
  inputDevices: AudioDeviceInfo[];
  outputDevices: AudioDeviceInfo[];
  onRescanDevices: () => void;
  onUpdateChannel: (index: number, patch: Partial<ChannelSettings>) => void;
  onUpdateShared: (patch: Partial<Pick<AppSettings, 'model' | 'previewModel' | 'autoLanguage'>>) => void;
}

export function SettingsPage({
  settings,
  inputDevices,
  outputDevices,
  onRescanDevices,
  onUpdateChannel,
  onUpdateShared,
}: SettingsPageProps) {
  const srcLanguages = ['Auto-detect', ...TARGET_LANGUAGES.slice().sort()];
  const tgtLanguages = TARGET_LANGUAGES.slice().sort();

  return (
    <div className="flex h-full flex-col gap-3 overflow-y-auto px-6 py-4">
      <div className="flex items-center gap-3">
        <h1 className="text-2xl font-semibold text-white">Settings</h1>
        <div className="flex-1" />
        <button
          type="button"
          onClick={onRescanDevices}
          title="Find audio devices again (e.g. after plugging one in or granting mic permission)"
          className="inline-flex items-center gap-1.5 rounded-md border border-white/15 bg-[#2d2d30] px-3 py-1.5 text-xs font-medium text-gray-200 hover:bg-[#38383c] cursor-pointer"
        >
          <RefreshCw className="h-3.5 w-3.5" />
          <span>Re-scan devices</span>
        </button>
      </div>

      <p className="text-xs text-gray-400">
        Changes apply at once. A channel restarts when its device, languages or output change.
      </p>

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        {settings.channels.map((ch, idx) => {
          const colour = CHANNEL_COLOURS[idx % CHANNEL_COLOURS.length];
          const inputLabels = inputDevices.map((d) => d.label);
          const outputLabels = outputDevices.map((d) => d.label);
          const hasInput = inputLabels.includes(ch.device);
          const hasOutput = outputLabels.includes(ch.output);

          return (
            <div
              key={idx}
              className="rounded-lg border border-white/10 bg-[#262626] overflow-hidden"
            >
              <div
                className="border-b border-white/10 bg-[#2d2d30] px-4 py-2.5 text-sm font-semibold"
                style={{ color: colour }}
              >
                Channel {idx + 1} · {ch.name}
              </div>

              <div className="grid grid-cols-[140px_1fr] items-center gap-x-3 gap-y-3 p-4 text-xs">
                <label className="text-gray-300" title="Shown in the conversation">
                  Name
                </label>
                <input
                  type="text"
                  maxLength={16}
                  value={ch.name}
                  onChange={(e) => onUpdateChannel(idx, { name: e.target.value })}
                  onBlur={(e) => {
                    if (!e.target.value.trim()) {
                      onUpdateChannel(idx, { name: `Channel ${idx + 1}` });
                    }
                  }}
                  className="rounded-md border border-white/15 bg-[#1f1f1f] px-2.5 py-1.5 text-xs text-white focus:border-[#0078D4] focus:outline-none"
                />

                <label
                  className="text-gray-300"
                  title="Device to listen to. [Loopback] captures browser tab or system audio."
                >
                  Listen to
                </label>
                <select
                  value={ch.device}
                  onChange={(e) => onUpdateChannel(idx, { device: e.target.value })}
                  className="rounded-md border border-white/15 bg-[#1f1f1f] px-2.5 py-1.5 text-xs text-white focus:border-[#0078D4] focus:outline-none"
                >
                  {inputLabels.map((lbl) => (
                    <option key={lbl} value={lbl}>
                      {lbl}
                    </option>
                  ))}
                  {!hasInput && ch.device ? (
                    <option value={ch.device}>{ch.device} (not connected)</option>
                  ) : null}
                </select>

                <label className="text-gray-300" title="Language spoken -> language shown">
                  Translate
                </label>
                <div className="flex items-center gap-2">
                  <select
                    value={ch.src}
                    onChange={(e) => onUpdateChannel(idx, { src: e.target.value })}
                    className="flex-1 min-w-0 rounded-md border border-white/15 bg-[#1f1f1f] px-2 py-1.5 text-xs text-white focus:border-[#0078D4] focus:outline-none"
                  >
                    {srcLanguages.map((lang) => (
                      <option key={lang} value={lang} disabled={!(lang in LANGUAGES)}>
                        {lang}
                      </option>
                    ))}
                  </select>
                  <span className="text-gray-400">→</span>
                  <select
                    value={ch.tgt}
                    onChange={(e) => onUpdateChannel(idx, { tgt: e.target.value })}
                    className="flex-1 min-w-0 rounded-md border border-white/15 bg-[#1f1f1f] px-2 py-1.5 text-xs text-white focus:border-[#0078D4] focus:outline-none"
                  >
                    {tgtLanguages.map((lang) => (
                      <option key={lang} value={lang}>
                        {lang}
                      </option>
                    ))}
                  </select>
                </div>

                <label
                  className="text-gray-300"
                  title="Higher = also catch quieter/less clear speech. Changes what counts as speech, not volume."
                >
                  Speech sensitivity
                </label>
                <div className="flex items-center gap-3">
                  <input
                    type="range"
                    min={0}
                    max={100}
                    value={Math.round(ch.sensitivity * 100)}
                    onChange={(e) =>
                      onUpdateChannel(idx, { sensitivity: Number(e.target.value) / 100 })
                    }
                    className="flex-1 accent-[#0078D4] cursor-pointer"
                  />
                  <span className="w-10 text-right font-mono text-gray-200">
                    {Math.round(ch.sensitivity * 100)}%
                  </span>
                </div>

                <label
                  className="text-gray-300"
                  title="Output device for Speak (this channel's translations read aloud) and for replay"
                >
                  Speak and replay to
                </label>
                <select
                  value={ch.output}
                  onChange={(e) => onUpdateChannel(idx, { output: e.target.value })}
                  className="rounded-md border border-white/15 bg-[#1f1f1f] px-2.5 py-1.5 text-xs text-white focus:border-[#0078D4] focus:outline-none"
                >
                  {outputLabels.map((lbl) => (
                    <option key={lbl} value={lbl}>
                      {lbl}
                    </option>
                  ))}
                  {!hasOutput && ch.output ? (
                    <option value={ch.output}>{ch.output} (not connected)</option>
                  ) : null}
                </select>

                <div />
                <label className="inline-flex items-center gap-2 text-gray-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={ch.replay}
                    onChange={(e) => onUpdateChannel(idx, { replay: e.target.checked })}
                    className="h-3.5 w-3.5 rounded accent-[#0078D4]"
                  />
                  <span>Hear input through output (audio monitor)</span>
                </label>
              </div>
            </div>
          );
        })}
      </div>

      <div className="rounded-lg border border-white/10 bg-[#262626] overflow-hidden">
        <div className="border-b border-white/10 bg-[#2d2d30] px-4 py-2.5 text-sm font-semibold text-gray-100">
          Both channels
        </div>
        <div className="grid grid-cols-[140px_minmax(240px,360px)] items-center gap-x-3 gap-y-3 p-4 text-xs">
          <label className="text-gray-300" title="Speech recognition model profile">
            Model
          </label>
          <select
            value={settings.model}
            onChange={(e) => onUpdateShared({ model: e.target.value })}
            className="rounded-md border border-white/15 bg-[#1f1f1f] px-2.5 py-1.5 text-xs text-white focus:border-[#0078D4] focus:outline-none"
          >
            {Object.keys(MODELS).map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>

          <label className="text-gray-300" title="Model for the live grey text">
            Previews
          </label>
          <select
            value={settings.previewModel}
            onChange={(e) => onUpdateShared({ previewModel: e.target.value })}
            className="rounded-md border border-white/15 bg-[#1f1f1f] px-2.5 py-1.5 text-xs text-white focus:border-[#0078D4] focus:outline-none"
          >
            {Object.keys(PREVIEW_CHOICES).map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>

          <div />
          <label
            className="inline-flex items-center gap-2 text-gray-300 cursor-pointer"
            title="If a phrase is clearly in another language, recognise it, label it and translate it."
          >
            <input
              type="checkbox"
              checked={settings.autoLanguage}
              onChange={(e) => onUpdateShared({ autoLanguage: e.target.checked })}
              className="h-3.5 w-3.5 rounded accent-[#0078D4]"
            />
            <span>Detect other languages</span>
          </label>
        </div>
      </div>
    </div>
  );
}
