import { Phrase } from "./api.ts";

export interface ParsedTranscriptBlock {
  speaker?: "Me" | "Them";
  text: string;
  startTime?: number;
}

/**
 * Normalizes any speaker string to "Me" or "Them".
 */
export function normalizeSpeakerLabel(speaker?: string): "Me" | "Them" {
  const s = (speaker || "").trim().toLowerCase();
  return ["system audio", "system / remote", "system", "remote", "them"].includes(s) ? "Them" : "Me";
}

/**
 * Parses a time string (e.g. "01:23", "01:23.50", "00:01:23") into seconds.
 */
export function parseTimeString(timeStr?: string): number | undefined {
  if (!timeStr) return undefined;
  const parts = timeStr.trim().split(":");
  if (parts.length === 2) {
    const mins = parseFloat(parts[0]);
    const secs = parseFloat(parts[1]);
    if (!isNaN(mins) && !isNaN(secs)) {
      return mins * 60 + secs;
    }
  } else if (parts.length === 3) {
    const hrs = parseFloat(parts[0]);
    const mins = parseFloat(parts[1]);
    const secs = parseFloat(parts[2]);
    if (!isNaN(hrs) && !isNaN(mins) && !isNaN(secs)) {
      return hrs * 3600 + mins * 60 + secs;
    }
  }
  return undefined;
}

/**
 * Checks if a string looks like an explicit speaker prefix.
 */
function isSpeakerPrefix(prefix: string): boolean {
  const p = prefix.trim().toLowerCase();
  return ["me", "them", "you", "system", "remote", "system audio"].includes(p) || /^speaker\s*\d+$/i.test(p);
}

/**
 * Parses full transcript raw text into discrete phrase blocks.
 * Supports:
 * - Double newline paragraph breaks
 * - Single newline lines starting with a speaker prefix ("Me:", "Them:")
 * - Optional timestamps (e.g. "[00:15] Me: Hello")
 */
export function parseTranscriptBlocks(rawText: string): ParsedTranscriptBlock[] {
  const trimmed = rawText.trim();
  if (!trimmed) return [];

  const rawParagraphs = trimmed.split(/\r?\n\s*\r?\n+/);
  const blocks: ParsedTranscriptBlock[] = [];

  const LINE_REGEX = /^(?:\[(\d{1,2}:\d{2}(?:\.\d+)?)\]\s*)?(?:([A-Za-z0-9_ -]+):\s+)?(.*)$/s;

  for (const para of rawParagraphs) {
    const lines = para.split(/\r?\n/).map((l) => l.trim()).filter((l) => l.length > 0);
    if (lines.length === 0) continue;

    let currentBlock: ParsedTranscriptBlock | null = null;

    for (let lineIdx = 0; lineIdx < lines.length; lineIdx++) {
      const line = lines[lineIdx];
      const match = line.match(LINE_REGEX);

      const timeStr = match ? match[1] : undefined;
      const prefixStr = match ? match[2] : undefined;
      const textPart = match ? match[3] : line;

      const hasValidSpeaker = Boolean(prefixStr && isSpeakerPrefix(prefixStr));
      const hasValidTime = Boolean(timeStr);

      if (hasValidSpeaker || hasValidTime) {
        if (currentBlock) {
          blocks.push(currentBlock);
        }
        currentBlock = {
          speaker: hasValidSpeaker ? normalizeSpeakerLabel(prefixStr) : undefined,
          startTime: parseTimeString(timeStr),
          text: textPart || "",
        };
      } else {
        if (!currentBlock) {
          // First line of paragraph without explicit known speaker
          currentBlock = {
            text: line,
          };
        } else {
          // Subsequent line in the same paragraph continues the current block
          currentBlock.text += (currentBlock.text ? " " : "") + line;
        }
      }
    }

    if (currentBlock) {
      blocks.push(currentBlock);
    }
  }

  return blocks;
}

/**
 * Builds formatted full transcript text from a list of phrases.
 * Preserves speaker labels ("Me:" / "Them:") when multiple speakers exist
 * or when the previous transcript used them.
 */
export function buildFinalTranscriptFromPhrases(
  phrases: Phrase[],
  existingTranscript?: string
): string {
  if (!phrases || phrases.length === 0) return "";

  const hasMultipleSpeakers = phrases.some((p) => normalizeSpeakerLabel(p.speaker) === "Them");
  const existingHasPrefixes = existingTranscript
    ? /^(?:Me|Them|You|System|Remote|System Audio|Speaker\s*\d+):\s*/im.test(existingTranscript)
    : true;
  const usePrefixes = hasMultipleSpeakers || existingHasPrefixes || !existingTranscript;

  return phrases
    .map((p) => {
      const txt = p.text || "";
      if (!usePrefixes) {
        return txt;
      }
      const spk = normalizeSpeakerLabel(p.speaker);
      return `${spk}: ${txt}`;
    })
    .join("\n\n");
}

/**
 * Reconciles parsed transcript blocks with existing phrases using
 * sequence alignment to preserve timestamps, audio jump targets, and phrase IDs.
 */
export function reconcilePhrases(
  oldPhrases: Phrase[],
  blocks: ParsedTranscriptBlock[],
  totalDuration: number,
  sessionId?: string
): Phrase[] {
  if (blocks.length === 0) return [];

  // If no existing phrases, create new phrases with distributed timestamps
  if (oldPhrases.length === 0) {
    const M = blocks.length;
    return blocks.map((b, idx) => {
      const st = b.startTime !== undefined
        ? b.startTime
        : (totalDuration > 0 ? (idx / M) * totalDuration : idx * 3.0);
      const et = totalDuration > 0
        ? ((idx + 1) / M) * totalDuration
        : st + 3.0;
      return {
        session_id: sessionId,
        phrase_id: `p_edit_${Date.now()}_${idx}`,
        speaker: b.speaker || "Me",
        start_time: Math.round(st * 100) / 100,
        end_time: Math.round(et * 100) / 100,
        duration: Math.max(0.5, Math.round((et - st) * 100) / 100),
        text: b.text,
      };
    });
  }

  // Exact count match: in-place 1-to-1 preservation of timestamps and IDs
  if (blocks.length === oldPhrases.length) {
    return blocks.map((b, idx) => {
      const old = oldPhrases[idx];
      const speaker = b.speaker || old.speaker || "Me";
      const startTime = b.startTime !== undefined ? b.startTime : old.start_time;
      return {
        ...old,
        speaker,
        start_time: startTime,
        text: b.text,
      };
    });
  }

  // Count differs: Compute optimal monotonic alignment using dynamic programming
  const M = blocks.length;
  const N = oldPhrases.length;

  const scoreMatrix: number[][] = Array.from({ length: M }, () => new Array(N).fill(0));
  for (let i = 0; i < M; i++) {
    const b = blocks[i];
    const bText = b.text.trim().toLowerCase();
    const bWords = bText.match(/\w+/g) || [];

    for (let j = 0; j < N; j++) {
      const p = oldPhrases[j];
      let s = 0;

      // Timestamp proximity
      if (b.startTime !== undefined) {
        const diff = Math.abs(b.startTime - p.start_time);
        if (diff < 0.5) s += 1000;
        else if (diff < 2.0) s += 500;
      }

      // Text similarity
      const pText = (p.text || "").trim().toLowerCase();
      if (bText && pText && bText === pText) {
        s += 300;
      } else if (bWords.length > 0) {
        const pWords = pText.match(/\w+/g) || [];
        if (pWords.length > 0) {
          const pSet = new Set(pWords);
          let common = 0;
          for (const w of bWords) {
            if (pSet.has(w)) common++;
          }
          if (common > 0) {
            s += (2 * common / (bWords.length + pWords.length)) * 200;
          }
        }
      }

      // Position relative proximity
      const relPosB = i / Math.max(1, M - 1);
      const relPosP = j / Math.max(1, N - 1);
      s += Math.max(0, 50 * (1 - Math.abs(relPosB - relPosP)));

      scoreMatrix[i][j] = s;
    }
  }

  // Standard alignment DP
  const F: number[][] = Array.from({ length: M + 1 }, () => new Array(N + 1).fill(-1e9));
  const parent: { prevI: number; prevJ: number; matched: boolean }[][] = Array.from(
    { length: M + 1 },
    () => Array.from({ length: N + 1 }, () => ({ prevI: -1, prevJ: -1, matched: false }))
  );

  F[0][0] = 0;
  for (let j = 1; j <= N; j++) {
    F[0][j] = 0;
    parent[0][j] = { prevI: 0, prevJ: j - 1, matched: false };
  }

  for (let i = 1; i <= M; i++) {
    for (let j = 1; j <= N; j++) {
      let best = F[i][j - 1];
      let bestP = { prevI: i, prevJ: j - 1, matched: false };

      if (F[i - 1][j] > best) {
        best = F[i - 1][j];
        bestP = { prevI: i - 1, prevJ: j, matched: false };
      }

      const matchS = F[i - 1][j - 1] + scoreMatrix[i - 1][j - 1];
      if (matchS > best) {
        best = matchS;
        bestP = { prevI: i - 1, prevJ: j - 1, matched: true };
      }

      F[i][j] = best;
      parent[i][j] = bestP;
    }
  }

  const blockToOld: (number | null)[] = new Array(M).fill(null);
  let curI = M;
  let curJ = N;
  while (curI > 0 && curJ >= 0) {
    const p = parent[curI][curJ];
    if (p.matched) {
      blockToOld[curI - 1] = curJ - 1;
      curI = p.prevI;
      curJ = p.prevJ;
    } else {
      curI = p.prevI;
      curJ = p.prevJ;
    }
  }

  const result: Phrase[] = [];
  for (let i = 0; i < M; i++) {
    const b = blocks[i];
    const oldIdx = blockToOld[i];
    if (oldIdx !== null && oldPhrases[oldIdx]) {
      const old = oldPhrases[oldIdx];
      result.push({
        ...old,
        speaker: b.speaker || old.speaker || "Me",
        start_time: b.startTime !== undefined ? b.startTime : old.start_time,
        text: b.text,
      });
    } else {
      // Unmatched block -> interpolate timestamps
      let prevEt = 0.0;
      for (let k = i - 1; k >= 0; k--) {
        if (result[k]) {
          prevEt = result[k].end_time;
          break;
        }
      }

      let nextSt: number | null = null;
      for (let k = i + 1; k < M; k++) {
        const oIdx = blockToOld[k];
        if (oIdx !== null && oldPhrases[oIdx]) {
          nextSt = oldPhrases[oIdx].start_time;
          break;
        }
      }

      let st: number;
      let et: number;
      if (b.startTime !== undefined) {
        st = b.startTime;
        et = nextSt !== null ? Math.max(st + 0.5, nextSt) : st + 3.0;
      } else if (nextSt !== null && nextSt > prevEt) {
        const gap = nextSt - prevEt;
        st = prevEt + gap * 0.1;
        et = prevEt + gap * 0.9;
      } else {
        st = prevEt;
        et = prevEt + 3.0;
      }

      result.push({
        session_id: sessionId,
        phrase_id: `p_edit_${Date.now()}_${i}`,
        speaker: b.speaker || "Me",
        start_time: Math.round(st * 100) / 100,
        end_time: Math.round(et * 100) / 100,
        duration: Math.max(0.5, Math.round((et - st) * 100) / 100),
        text: b.text,
      });
    }
  }

  return result;
}
