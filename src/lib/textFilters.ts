import {
  CONTINUATION_MARKS,
  HALLUCINATION_FRAGMENTS,
  HALLUCINATIONS,
  LOGPROB_FLOOR,
  MIN_CHARS_PER_SEC,
  SUSPECT_PHRASES,
} from './config';

export function suspiciousReasons(
  text: string,
  speechSec: number,
  avgLogprob = -0.25,
): string[] {
  const reasons: string[] = [];
  const normalized = text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, '')
    .replace(/\s+/g, ' ')
    .trim();

  if (speechSec >= 1.5 && text.length / speechSec < MIN_CHARS_PER_SEC) {
    reasons.push('too_short');
  }
  if (SUSPECT_PHRASES.has(normalized) && speechSec >= 1.5) {
    reasons.push('stock_phrase');
  }
  if (avgLogprob < LOGPROB_FLOOR) {
    reasons.push('low_confidence');
  }
  if (text.length > 20) {
    const words = text.toLowerCase().split(/\s+/).filter(Boolean);
    const uniqueRatio = words.length > 0 ? new Set(words).size / words.length : 1;
    if (words.length >= 6 && uniqueRatio < 0.35) {
      reasons.push('repetitive');
    }
  }
  return reasons;
}

export function continuationJoin(
  textBefore: string,
  newText: string,
): { joined: boolean; text: string } {
  const before = textBefore.trimEnd();
  const next = newText.trimStart();
  const endsWithMark = CONTINUATION_MARKS.some((m) => before.endsWith(m));
  const startsWithMark = CONTINUATION_MARKS.some((m) => next.startsWith(m));
  if (endsWithMark && startsWithMark) {
    return {
      joined: true,
      text: next.replace(/^[.…\s]+/, ''),
    };
  }
  return { joined: false, text: newText };
}

export function trailingMarksLength(textBefore: string): number {
  const stripped = textBefore.trimEnd().replace(/[.…]+$/, '').trimEnd();
  return textBefore.length - stripped.length;
}

export function cleanTranslation(source: string, translated: string): string {
  const splitSentences = (t: string) =>
    t
      .trim()
      .split(/(?<=[.!?…])\s+/)
      .filter(Boolean);
  const sentences = splitSentences(translated);
  if (splitSentences(source).length > 1) {
    return translated;
  }
  const kept = sentences.filter(
    (x, i) => i === 0 || x.trim().toLowerCase() !== sentences[i - 1].trim().toLowerCase(),
  );
  let out = kept.join(' ');
  const srcWords = source.trim().split(/\s+/).filter(Boolean).length;
  const outWords = out.trim().split(/\s+/).filter(Boolean).length;
  if (srcWords <= 2 && outWords > 3 * srcWords + 3 && kept.length > 1) {
    out = kept[0];
  }
  return out;
}

export function collapseRepetitions(text: string): string {
  let out = text.replace(
    /(\b(?:\p{L}+[\s,]+){1,11}?\p{L}+)(?:[\s,.!?…-]+\1\b){2,}/giu,
    (_, g1: string) => `${g1}, ${g1}…`,
  );
  out = out.replace(
    /\b(\p{L}+)([-\s,.!?]+)(?:\1\2?){3,}\b/giu,
    (_, word: string, sep: string) => [word, word.toLowerCase(), word.toLowerCase()].join(sep),
  );
  return out;
}

export function isMusicLoop(text: string): boolean {
  const letters = Array.from(text.toLowerCase()).filter((c) => /\p{L}/u.test(c));
  if (letters.length >= 8) {
    const counts = new Map<string, number>();
    for (const c of letters) {
      counts.set(c, (counts.get(c) ?? 0) + 1);
    }
    const maxCount = Math.max(...counts.values());
    if (maxCount / letters.length > 0.8) {
      return true;
    }
  }
  const words = text.toLowerCase().match(/[\p{L}\p{N}_]+/gu) ?? [];
  return words.length >= 4 && new Set(words).size === 1;
}

export function hallucinationReason(text: string): string | null {
  const lower = text.toLowerCase();
  if (HALLUCINATION_FRAGMENTS.some((f) => lower.includes(f))) {
    return 'subtitle credit';
  }
  const normalized = lower
    .replace(/[^\p{L}\p{N}\s]/gu, '')
    .replace(/\s+/g, ' ')
    .trim();
  if (!normalized) {
    return 'empty';
  }
  if (HALLUCINATIONS.has(normalized)) {
    return 'stock phrase';
  }
  return isMusicLoop(text) ? 'music loop' : null;
}

export function isHallucination(text: string): boolean {
  return hallucinationReason(text) !== null;
}
