import { useState } from 'react';
import { Play } from 'lucide-react';
import { CHANNEL_COLOURS } from '../lib/config';
import type { ChannelSettings } from '../lib/types';

interface SimulatorBarProps {
  channels: [ChannelSettings, ChannelSettings];
  onSimulatePhrase: (channelIndex: number, text: string) => void;
}

const SAMPLE_CLIPS: Array<{ label: string; channel: number; text: string }> = [
  {
    label: 'Me (EN): Audio check',
    channel: 0,
    text: 'Can you hear me clearly on the call today?',
  },
  {
    label: 'Friend (RU): Greeting',
    channel: 1,
    text: 'Да, слышу тебя очень хорошо, давай начнём.',
  },
  {
    label: 'Friend (RU): Continuation ...',
    channel: 1,
    text: 'Мы можем обсудить новый проект...',
  },
  {
    label: 'Friend (RU): ...continued',
    channel: 1,
    text: '...и сразу проверить перевод.',
  },
  {
    label: 'Whisper Hallucination Test',
    channel: 1,
    text: 'Thanks for watching',
  },
];

export function SimulatorBar({ channels, onSimulatePhrase }: SimulatorBarProps) {
  const [channelIndex, setChannelIndex] = useState(1);
  const [inputText, setInputText] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = inputText.trim();
    if (!trimmed) return;
    onSimulatePhrase(channelIndex, trimmed);
    setInputText('');
  };

  return (
    <div className="rounded-lg border border-white/10 bg-[#242424] px-3 py-2">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[11px] font-semibold uppercase tracking-wide text-gray-400">
          Phrase Simulator
        </span>

        <div className="flex flex-wrap items-center gap-1.5">
          {SAMPLE_CLIPS.map((clip) => (
            <button
              key={clip.label}
              type="button"
              onClick={() => onSimulatePhrase(clip.channel, clip.text)}
              className="inline-flex items-center gap-1 rounded border border-white/10 bg-[#2f2f2f] px-2 py-0.5 text-[11px] text-gray-200 hover:bg-[#3a3a3a] cursor-pointer"
            >
              <Play className="h-2.5 w-2.5 text-[#4FC3F7]" />
              <span>{clip.label}</span>
            </button>
          ))}
        </div>
      </div>

      <form onSubmit={handleSubmit} className="mt-2 flex items-center gap-2">
        <select
          value={channelIndex}
          onChange={(e) => setChannelIndex(Number(e.target.value))}
          className="rounded border border-white/15 bg-[#1b1b1b] px-2 py-1 text-xs font-medium focus:border-[#0078D4] focus:outline-none"
          style={{ color: CHANNEL_COLOURS[channelIndex % CHANNEL_COLOURS.length] }}
        >
          {channels.map((ch, idx) => (
            <option key={idx} value={idx} className="text-white">
              {ch.name} ({ch.src} → {ch.tgt})
            </option>
          ))}
        </select>

        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Type or paste a phrase to simulate live preview → transcription → translation..."
          className="flex-1 rounded border border-white/15 bg-[#1b1b1b] px-2.5 py-1 text-xs text-white placeholder:text-gray-500 focus:border-[#0078D4] focus:outline-none"
        />

        <button
          type="submit"
          className="inline-flex items-center gap-1 rounded bg-[#0078D4] px-3 py-1 text-xs font-medium text-white hover:bg-[#106ebe] cursor-pointer"
        >
          <Play className="h-3 w-3" />
          <span>Run phrase</span>
        </button>
      </form>
    </div>
  );
}
