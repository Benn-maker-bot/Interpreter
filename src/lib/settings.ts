import { DEFAULT_SENS, LANGUAGES, TARGET_LANGUAGES } from './config';
import type { AppSettings } from './types';

const STORAGE_KEY = 'voice_interpreter_settings_v1';

export const DEFAULT_SETTINGS: AppSettings = {
  channels: [
    {
      name: 'Me',
      device: 'Default Microphone',
      src: 'English',
      tgt: 'Russian',
      sensitivity: DEFAULT_SENS,
      muted: false,
      speak: false,
      output: 'Default Audio Output',
      replay: false,
    },
    {
      name: 'Friend',
      device: 'Browser Tab / System Audio [Loopback]',
      src: 'Russian',
      tgt: 'English',
      sensitivity: DEFAULT_SENS,
      muted: true,
      speak: false,
      output: 'Default Audio Output',
      replay: false,
    },
  ],
  model: 'Web Speech (Live Browser)',
  previewModel: 'Small (balanced)',
  autoLanguage: true,
  textSize: 13,
};

export function loadSettings(): AppSettings {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return structuredClone(DEFAULT_SETTINGS);
    const parsed = JSON.parse(raw) as Partial<AppSettings>;
    const merged: AppSettings = {
      ...structuredClone(DEFAULT_SETTINGS),
      ...parsed,
      channels: [
        { ...DEFAULT_SETTINGS.channels[0], ...(parsed.channels?.[0] ?? {}) },
        { ...DEFAULT_SETTINGS.channels[1], ...(parsed.channels?.[1] ?? {}) },
      ],
    };
    for (const ch of merged.channels) {
      if (!(ch.src in LANGUAGES)) ch.src = 'Auto-detect';
      if (!TARGET_LANGUAGES.includes(ch.tgt)) ch.tgt = 'English';
      ch.sensitivity = Math.max(0, Math.min(1, Number(ch.sensitivity) || DEFAULT_SENS));
    }
    merged.textSize = Math.max(9, Math.min(28, Number(merged.textSize) || 13));
    return merged;
  } catch {
    return structuredClone(DEFAULT_SETTINGS);
  }
}

export function saveSettings(next: AppSettings): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  } catch {
    // Ignore storage quota errors in ephemeral/private tabs
  }
}
