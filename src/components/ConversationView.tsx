import { useEffect, useRef } from 'react';
import { CHANNEL_COLOURS } from '../lib/config';
import type { ConversationEntry } from '../lib/types';

interface ConversationViewProps {
  entries: ConversationEntry[];
  channelNames: string[];
  textSizePt: number;
}

export function ConversationView({
  entries,
  channelNames,
  textSizePt,
}: ConversationViewProps) {
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const wasAtBottomRef = useRef(true);

  const handleScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    wasAtBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight <= 16;
  };

  useEffect(() => {
    const el = scrollRef.current;
    if (el && wasAtBottomRef.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [entries]);

  return (
    <div
      ref={scrollRef}
      onScroll={handleScroll}
      className="flex-1 overflow-y-auto rounded-lg border border-white/10 bg-[#222222] px-4 py-3"
      style={{ fontSize: `${textSizePt}pt`, lineHeight: 1.45 }}
    >
      {entries.length === 0 ? (
        <div className="flex h-full items-center justify-center text-sm text-gray-500 select-none">
          Speak into your microphone or use the Simulator bar below to see live transcription and translation.
        </div>
      ) : (
        <div className="flex flex-col">
          {entries.map((entry) => {
            const colour = CHANNEL_COLOURS[entry.channel % CHANNEL_COLOURS.length];
            const name = channelNames[entry.channel] ?? '?';
            const words = entry.heard.split(/\s+/).filter(Boolean);
            const stableWords = words.slice(0, entry.stable).join(' ');
            const unstableWords = words.slice(entry.stable).join(' ');

            return (
              <div
                key={entry.key}
                className={`grid grid-cols-[126px_1fr] gap-x-3 ${
                  entry.showLabel ? 'mt-2.5 first:mt-0' : 'mt-1'
                }`}
              >
                {/* Left speaker & timestamp column */}
                <div className="truncate whitespace-nowrap pt-0.5 text-[0.88em]">
                  {entry.showLabel ? (
                    <>
                      <span className="font-mono text-[#8a8f94]">{entry.time}  </span>
                      <span className="font-semibold" style={{ color: colour }}>
                        {name}
                      </span>
                    </>
                  ) : null}
                </div>

                {/* Right transcript & translation column */}
                <div className="min-w-0 break-words">
                  <div>
                    {entry.detectedLang ? (
                      <span className="mr-1.5 text-[0.85em] font-medium text-gray-400">
                        [{entry.detectedLang}]
                      </span>
                    ) : null}
                    {entry.final ? (
                      <span className="text-gray-100">{entry.heard}</span>
                    ) : (
                      <span>
                        <span className="text-gray-200">
                          … {stableWords}
                          {stableWords && unstableWords ? ' ' : ''}
                        </span>
                        <span className="text-[#8a8f94]">{unstableWords}</span>
                      </span>
                    )}
                  </div>
                  <div className="mt-0.5">
                    {entry.translation === null ? (
                      <span className="text-[#8a8f94]">→ …</span>
                    ) : (
                      <>
                        <span style={{ color: colour }}>→ </span>
                        <span
                          className={
                            entry.translationFinal ? 'text-gray-200' : 'text-[#8a8f94]'
                          }
                        >
                          {entry.translation || '(no translation)'}
                        </span>
                      </>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
