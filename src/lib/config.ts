export const OK = '#2ECC71';
export const WARN = '#F1C40F';
export const BAD = '#E74C3C';
export const MUTED = '#9AA0A6';
export const ACCENT = '#0078D4';

export const CHANNEL_COLOURS = ['#4FC3F7', '#FFB74D'] as const;

export const DEFAULT_SENS = 0.5;
export const MAX_TEXT_HISTORY_CHARS = 12_000;
export const MAX_ENTRIES = 300;
export const SAME_TURN_SEC = 60;

export const SILENCE_SEC = 0.3;
export const MAX_PHRASE_SEC = 10;
export const MUSIC_SILENCE_SEC = 0.9;
export const MUSIC_MAX_PHRASE_SEC = 15;
export const PARTIAL_INTERVAL_MS = 200;
export const EARLY_FINAL_SEC = 0.1;
export const SILENCE_FLOOR_DBFS = -65;

export const MIN_CHARS_PER_SEC = 5.0;
export const LOGPROB_FLOOR = -1.0;
export const MAX_COMPRESSION = 2.4;

export interface LanguageEntry {
  speechCode: string;      // BCP-47 / Whisper language code
  translateCode: string;   // Google/MyMemory code
}

export const LANGUAGES: Record<string, LanguageEntry> = {
  'Auto-detect': { speechCode: 'auto', translateCode: 'auto' },
  English: { speechCode: 'en-US', translateCode: 'en' },
  Russian: { speechCode: 'ru-RU', translateCode: 'ru' },
  Arabic: { speechCode: 'ar-SA', translateCode: 'ar' },
  Bulgarian: { speechCode: 'bg-BG', translateCode: 'bg' },
  'Chinese (Mandarin)': { speechCode: 'zh-CN', translateCode: 'zh-CN' },
  Czech: { speechCode: 'cs-CZ', translateCode: 'cs' },
  Dutch: { speechCode: 'nl-NL', translateCode: 'nl' },
  French: { speechCode: 'fr-FR', translateCode: 'fr' },
  German: { speechCode: 'de-DE', translateCode: 'de' },
  Greek: { speechCode: 'el-GR', translateCode: 'el' },
  Hindi: { speechCode: 'hi-IN', translateCode: 'hi' },
  Italian: { speechCode: 'it-IT', translateCode: 'it' },
  Japanese: { speechCode: 'ja-JP', translateCode: 'ja' },
  Korean: { speechCode: 'ko-KR', translateCode: 'ko' },
  Polish: { speechCode: 'pl-PL', translateCode: 'pl' },
  Portuguese: { speechCode: 'pt-PT', translateCode: 'pt' },
  Spanish: { speechCode: 'es-ES', translateCode: 'es' },
  Turkish: { speechCode: 'tr-TR', translateCode: 'tr' },
  Ukrainian: { speechCode: 'uk-UA', translateCode: 'uk' },
};

export const TARGET_LANGUAGES = Object.keys(LANGUAGES).filter(
  (name) => LANGUAGES[name].speechCode !== 'auto',
);

export const LANGUAGE_NAMES: Record<string, string> = {
  ...Object.fromEntries(
    Object.entries(LANGUAGES)
      .filter(([, v]) => v.translateCode !== 'auto')
      .map(([name, v]) => [v.translateCode.split('-')[0], name]),
  ),
  be: 'Belarusian',
  sr: 'Serbian',
  sv: 'Swedish',
  fi: 'Finnish',
  kk: 'Kazakh',
  he: 'Hebrew',
  hu: 'Hungarian',
  ro: 'Romanian',
  da: 'Danish',
  no: 'Norwegian',
  lv: 'Latvian',
  lt: 'Lithuanian',
  et: 'Estonian',
};

export const MODELS: Record<string, string> = {
  'Web Speech (Live Browser)': 'web-speech-live',
  'Tiny  - fastest': 'tiny',
  'Small - recommended ✓': 'small',
  'Medium - most accurate': 'medium',
  'Turbo - best accuracy ★': 'large-v3-turbo',
};

export const PREVIEW_CHOICES: Record<string, string | null> = {
  'Off (final text only)': 'off',
  'Tiny (instant)': 'tiny',
  'Small (balanced)': 'small',
  'Main model': null,
};

export const FEEDBACK_PROBLEMS: Record<string, string> = {
  words: 'Wrong or missing words',
  language: 'Wrong language',
  translation: 'Bad translation',
  cut: 'Cut in the middle of a sentence',
  invented: 'Text that was never said',
  missed: 'Speech that was not shown at all',
};

export const HALLUCINATIONS = new Set([
  'thank you',
  'thanks',
  'thank you very much',
  'thanks for watching',
  'thank you for watching',
  'please subscribe',
  'subscribe',
  'you',
  'bye',
  'hmm',
  'um',
  'uh',
  'oh',
  'so',
  'продолжение следует',
  'спасибо за просмотр',
  'подписывайтесь на канал',
]);

export const SUSPECT_PHRASES = new Set([
  'спасибо',
  'спасибо за внимание',
  'продолжение следует',
  'субтитры сделал dimatorzok',
  'редактор субтитров а синецкая',
  'thank you',
  'thanks for watching',
]);

export const HALLUCINATION_FRAGMENTS = [
  'субтитр',
  'dimatorzok',
  'редактор субтитров',
  'корректор',
  'amara.org',
  'subtitles by',
  'ご視聴ありがとうございました',
];

export const CONTINUATION_MARKS = ['...', '…'];

export function vadThresholdFor(sensitivity: number): number {
  const clamped = Math.max(0, Math.min(1, sensitivity));
  return 0.85 - 0.7 * clamped;
}

export function plainDeviceName(label: string): string {
  return label.replace(/\s*\[Loopback\]$/i, '').replace(/^🔊\s*/, '').trim();
}
