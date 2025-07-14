
/*
 * Sanskrit IAST → Russian diacritic transliteration
 * -------------------------------------------------
 * The mapping below follows conventions used in Russian indological
 * literature.  Cyrillic base letters are augmented with combining diacritics
 * (macron, dot below, dot above, etc.) using Unicode combining characters.
 */

// Unicode combining marks that we will need
const MACRON = "\u0304";        // 04  (combining macron)
const DOT_BELOW = "\u0323";     // ̣
const DOT_ABOVE = "\u0307";     // ̇
const ACUTE = "\u0301";         // ́
const TILDE = "\u0303";         // ̃

/**
 * Rules are ordered from longest (multi-letter) to shortest so that digraphs
 * are processed before their components (e.g. «kh» before «h»).
 */
// Mapping dictionary (lower-case keys)
const REPLACEMENTS: Record<string, string> = {
  // Diphthongs
  'ai': 'аи',
  'au': 'ау',

  // Aspirated / compounds
  'kh': 'кх',
  'gh': 'гх',
  'ch': 'чх',
  'jh': 'джх',
  'ṭh': `т${DOT_BELOW}х`,
  'ḍh': `д${DOT_BELOW}х`,
  'th': 'тх',
  'dh': 'дх',
  'ph': 'пх',
  'bh': 'бх',

  // Single consonants w/ diacritics
  'ṅ': `н${DOT_ABOVE}`,
  'ñ': `н${TILDE}`,
  'ṭ': `т${DOT_BELOW}`,
  'ḍ': `д${DOT_BELOW}`,
  'ṇ': `н${DOT_BELOW}`,
  'ś': `ш${ACUTE}`,
  'ṣ': `ш${DOT_BELOW}`,
  'ḥ': `х${DOT_BELOW}`,
  'ṃ': `м${DOT_BELOW}`,
  'ṁ': `м${DOT_ABOVE}`,

  // Plain consonants
  'k': 'к',
  'g': 'г',
  'c': 'ч',
  'j': 'дж',
  't': 'т',
  'd': 'д',
  'n': 'н',
  'p': 'п',
  'b': 'б',
  'm': 'м',
  'y': 'й',
  'r': 'р',
  'l': 'л',
  'v': 'в',
  's': 'с',
  'h': 'х',

  // Vowels with diacritics
  'ā': `а${MACRON}`,
  'ī': `и${MACRON}`,
  'ū': `у${MACRON}`,
  'ṝ': `р${DOT_BELOW}${MACRON}`,
  'ṝ': `р${DOT_BELOW}${MACRON}`,
  'ṛ': `р${DOT_BELOW}`,
  'ḹ': `л${DOT_BELOW}${MACRON}`,
  'ḹ': `л${DOT_BELOW}${MACRON}`,
  'ḷ': `л${DOT_BELOW}`,

  // Plain vowels
  'a': 'а',
  'i': 'и',
  'u': 'у',
  'e': 'е',
  'o': 'о',
};

// Sort keys by length desc to ensure digraphs handled first
const REPLACEMENT_REGEX = new RegExp(
  Object.keys(REPLACEMENTS)
    .sort((a, b) => b.length - a.length)
    .map((k) => k.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&'))
    .join('|'),
  'gi'
);

function applyCase(sample: string, replacement: string): string {
  // if sample is all caps
  if (sample.toUpperCase() === sample) return replacement.toUpperCase();
  // if only first letter capitalised
  if (sample[0].toUpperCase() === sample[0]) {
    return replacement[0].toUpperCase() + replacement.slice(1);
  }
  return replacement;
}

/**
 * Convert IAST Sanskrit string to Russian diacritic notation.
 * @param input – raw IAST string
 */
export function convertIastToRus(input: string): string {
  return input.normalize('NFC').replace(REPLACEMENT_REGEX, (match) => {
    const lower = match.toLowerCase();
    const replacement = REPLACEMENTS[lower] ?? match;
    return applyCase(match, replacement);
  });
} 