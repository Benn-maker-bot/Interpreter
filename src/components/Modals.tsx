import { useState } from 'react';
import { X } from 'lucide-react';
import { FEEDBACK_PROBLEMS, LANGUAGE_NAMES } from '../lib/config';
import type { PhraseMetric } from '../lib/types';

interface ReportMistakeModalProps {
  recentPhrases: PhraseMetric[];
  onClose: () => void;
  onSubmit: (payload: {
    phraseKey: string;
    problems: string[];
    correctText: string;
    actualLanguage: string | null;
    note: string;
  }) => void;
}

export function ReportMistakeModal({
  recentPhrases,
  onClose,
  onSubmit,
}: ReportMistakeModalProps) {
  const reversed = [...recentPhrases].reverse();
  const [selectedKey, setSelectedKey] = useState(reversed[0]?.key ?? '');
  const [checked, setChecked] = useState<Record<string, boolean>>({});
  const [correctText, setCorrectText] = useState('');
  const [actualLang, setActualLang] = useState<string>('');
  const [note, setNote] = useState('');

  const langs = Object.entries(LANGUAGE_NAMES).sort((a, b) => a[1].localeCompare(b[1]));
  const hasAnyChecked = Object.values(checked).some(Boolean);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/65 p-4">
      <div className="w-full max-w-2xl rounded-xl border border-white/15 bg-[#2b2b2b] shadow-2xl">
        <div className="flex items-center justify-between border-b border-white/10 px-5 py-3.5">
          <h2 className="text-base font-semibold text-white">Report a mistake</h2>
          <button
            type="button"
            onClick={onClose}
            className="rounded p-1 text-gray-400 hover:bg-white/10 hover:text-white cursor-pointer"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="flex flex-col gap-3 p-5 text-xs">
          <div>
            <label className="mb-1 block text-gray-300">Which phrase?</label>
            <select
              value={selectedKey}
              onChange={(e) => setSelectedKey(e.target.value)}
              className="w-full rounded-md border border-white/15 bg-[#1e1e1e] px-2.5 py-2 text-xs text-white focus:border-[#0078D4] focus:outline-none"
            >
              {reversed.map((p) => (
                <option key={p.key} value={p.key}>
                  {p.time} · {p.who}: {p.text.slice(0, 85) || '(empty)'}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="mb-1.5 block text-gray-300">
              What was wrong? (tick all that apply)
            </label>
            <div className="grid grid-cols-1 gap-1.5 sm:grid-cols-2">
              {Object.entries(FEEDBACK_PROBLEMS).map(([key, label]) => (
                <label
                  key={key}
                  className="flex items-center gap-2 rounded border border-white/10 bg-[#222222] px-2.5 py-1.5 text-gray-200 cursor-pointer hover:bg-[#292929]"
                >
                  <input
                    type="checkbox"
                    checked={Boolean(checked[key])}
                    onChange={(e) =>
                      setChecked((prev) => ({ ...prev, [key]: e.target.checked }))
                    }
                    className="h-3.5 w-3.5 accent-[#0078D4]"
                  />
                  <span>{label}</span>
                </label>
              ))}
            </div>
          </div>

          <div>
            <label className="mb-1 block text-gray-300">
              What was really said (optional, makes the report most useful)
            </label>
            <input
              type="text"
              value={correctText}
              onChange={(e) => setCorrectText(e.target.value)}
              placeholder="The correct words, in the language spoken"
              className="w-full rounded-md border border-white/15 bg-[#1e1e1e] px-2.5 py-1.5 text-xs text-white focus:border-[#0078D4] focus:outline-none"
            />
          </div>

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <div>
              <label className="mb-1 block text-gray-300">Language actually spoken</label>
              <select
                value={actualLang}
                onChange={(e) => setActualLang(e.target.value)}
                className="w-full rounded-md border border-white/15 bg-[#1e1e1e] px-2.5 py-1.5 text-xs text-white focus:border-[#0078D4] focus:outline-none"
              >
                <option value="">(as detected)</option>
                {langs.map(([code, name]) => (
                  <option key={code} value={code}>
                    {name}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="mb-1 block text-gray-300">Anything else (optional)</label>
              <input
                type="text"
                value={note}
                onChange={(e) => setNote(e.target.value)}
                placeholder="Additional context"
                className="w-full rounded-md border border-white/15 bg-[#1e1e1e] px-2.5 py-1.5 text-xs text-white focus:border-[#0078D4] focus:outline-none"
              />
            </div>
          </div>
        </div>

        <div className="flex justify-end gap-2 border-t border-white/10 px-5 py-3">
          <button
            type="button"
            onClick={onClose}
            className="rounded-md border border-white/15 bg-[#323232] px-4 py-1.5 text-xs font-medium text-gray-200 hover:bg-[#3d3d3d] cursor-pointer"
          >
            Cancel
          </button>
          <button
            type="button"
            disabled={!hasAnyChecked}
            onClick={() =>
              onSubmit({
                phraseKey: selectedKey,
                problems: Object.entries(checked)
                  .filter(([, v]) => v)
                  .map(([k]) => k),
                correctText,
                actualLanguage: actualLang || null,
                note,
              })
            }
            className="rounded-md bg-[#0078D4] px-4 py-1.5 text-xs font-medium text-white hover:bg-[#106ebe] disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
          >
            Save report
          </button>
        </div>
      </div>
    </div>
  );
}

export function SpeedReportModal({
  reportText,
  onClose,
}: {
  reportText: string;
  onClose: () => void;
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/65 p-4">
      <div className="w-full max-w-3xl rounded-xl border border-white/15 bg-[#2b2b2b] shadow-2xl">
        <div className="flex items-center justify-between border-b border-white/10 px-5 py-3.5">
          <h2 className="text-base font-semibold text-white">Speed report (this run)</h2>
          <button
            type="button"
            onClick={onClose}
            className="rounded p-1 text-gray-400 hover:bg-white/10 hover:text-white cursor-pointer"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="p-5">
          <pre className="max-h-[60vh] overflow-auto rounded-lg border border-white/10 bg-[#1a1a1a] p-4 font-mono text-xs leading-relaxed text-gray-200">
            {reportText}
          </pre>
        </div>

        <div className="flex justify-end border-t border-white/10 px-5 py-3">
          <button
            type="button"
            onClick={onClose}
            className="rounded-md bg-[#0078D4] px-5 py-1.5 text-xs font-medium text-white hover:bg-[#106ebe] cursor-pointer"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
