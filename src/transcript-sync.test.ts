import { describe, expect, it } from "vitest";
import { Phrase } from "./api.ts";
import {
  buildFinalTranscriptFromPhrases,
  normalizeSpeakerLabel,
  parseTimeString,
  parseTranscriptBlocks,
  reconcilePhrases,
} from "./transcript-sync.ts";

describe("transcript-sync", () => {
  describe("normalizeSpeakerLabel", () => {
    it("normalizes system loopback speakers to Them", () => {
      expect(normalizeSpeakerLabel("Them")).toBe("Them");
      expect(normalizeSpeakerLabel("them")).toBe("Them");
      expect(normalizeSpeakerLabel("system")).toBe("Them");
      expect(normalizeSpeakerLabel("System")).toBe("Them");
      expect(normalizeSpeakerLabel("remote")).toBe("Them");
      expect(normalizeSpeakerLabel("system audio")).toBe("Them");
      expect(normalizeSpeakerLabel("system / remote")).toBe("Them");
    });

    it("normalizes microphone speakers to Me", () => {
      expect(normalizeSpeakerLabel("Me")).toBe("Me");
      expect(normalizeSpeakerLabel("me")).toBe("Me");
      expect(normalizeSpeakerLabel("You")).toBe("Me");
      expect(normalizeSpeakerLabel("")).toBe("Me");
      expect(normalizeSpeakerLabel(undefined)).toBe("Me");
    });
  });

  describe("parseTimeString", () => {
    it("parses MM:SS and MM:SS.ms format", () => {
      expect(parseTimeString("01:23")).toBe(83);
      expect(parseTimeString("00:05")).toBe(5);
      expect(parseTimeString("01:23.5")).toBe(83.5);
    });

    it("parses HH:MM:SS format", () => {
      expect(parseTimeString("01:00:00")).toBe(3600);
      expect(parseTimeString("01:02:03")).toBe(3723);
    });

    it("returns undefined for invalid strings", () => {
      expect(parseTimeString(undefined)).toBeUndefined();
      expect(parseTimeString("invalid")).toBeUndefined();
    });
  });

  describe("parseTranscriptBlocks", () => {
    it("returns empty array for empty or whitespace text", () => {
      expect(parseTranscriptBlocks("")).toEqual([]);
      expect(parseTranscriptBlocks("   \n\n  ")).toEqual([]);
    });

    it("parses standard Me/Them paragraphs", () => {
      const text = "Me: Hello there\n\nThem: General Kenobi\n\nMe: You are a bold one";
      const blocks = parseTranscriptBlocks(text);
      expect(blocks).toHaveLength(3);
      expect(blocks[0]).toEqual({ speaker: "Me", startTime: undefined, text: "Hello there" });
      expect(blocks[1]).toEqual({ speaker: "Them", startTime: undefined, text: "General Kenobi" });
      expect(blocks[2]).toEqual({ speaker: "Me", startTime: undefined, text: "You are a bold one" });
    });

    it("parses single-newline separated speaker lines", () => {
      const text = "Me: Hello\nThem: Hi";
      const blocks = parseTranscriptBlocks(text);
      expect(blocks).toHaveLength(2);
      expect(blocks[0]).toEqual({ speaker: "Me", startTime: undefined, text: "Hello" });
      expect(blocks[1]).toEqual({ speaker: "Them", startTime: undefined, text: "Hi" });
    });

    it("parses optional timestamp tags", () => {
      const text = "[00:15] Me: Welcome everyone\n\n[01:00.50] Them: Thanks for having us";
      const blocks = parseTranscriptBlocks(text);
      expect(blocks).toHaveLength(2);
      expect(blocks[0]).toEqual({ speaker: "Me", startTime: 15, text: "Welcome everyone" });
      expect(blocks[1]).toEqual({ speaker: "Them", startTime: 60.5, text: "Thanks for having us" });
    });

    it("does not mistake ordinary colons for speakers", () => {
      const text = "Note: this is important\n\nNotice: please read";
      const blocks = parseTranscriptBlocks(text);
      expect(blocks).toHaveLength(2);
      expect(blocks[0].speaker).toBeUndefined();
      expect(blocks[0].text).toBe("Note: this is important");
      expect(blocks[1].speaker).toBeUndefined();
      expect(blocks[1].text).toBe("Notice: please read");
    });
  });

  describe("buildFinalTranscriptFromPhrases", () => {
    it("returns empty string when no phrases", () => {
      expect(buildFinalTranscriptFromPhrases([])).toBe("");
    });

    it("formats multi-speaker phrases with speaker prefixes", () => {
      const phrases: Phrase[] = [
        { phrase_id: "1", start_time: 0, end_time: 2, duration: 2, speaker: "Me", text: "Hello" },
        { phrase_id: "2", start_time: 2.5, end_time: 4, duration: 1.5, speaker: "Them", text: "Hi there" },
      ];
      const result = buildFinalTranscriptFromPhrases(phrases);
      expect(result).toBe("Me: Hello\n\nThem: Hi there");
    });

    it("formats single-speaker with prefixes if existing transcript used prefixes", () => {
      const phrases: Phrase[] = [
        { phrase_id: "1", start_time: 0, end_time: 2, duration: 2, speaker: "Me", text: "Line 1" },
        { phrase_id: "2", start_time: 2.5, end_time: 4, duration: 1.5, speaker: "Me", text: "Line 2" },
      ];
      const result = buildFinalTranscriptFromPhrases(phrases, "Me: Initial");
      expect(result).toBe("Me: Line 1\n\nMe: Line 2");
    });

    it("formats single-speaker without prefixes if existing transcript did not use prefixes", () => {
      const phrases: Phrase[] = [
        { phrase_id: "1", start_time: 0, end_time: 2, duration: 2, speaker: "Me", text: "Line 1" },
        { phrase_id: "2", start_time: 2.5, end_time: 4, duration: 1.5, speaker: "Me", text: "Line 2" },
      ];
      const result = buildFinalTranscriptFromPhrases(phrases, "Plain initial line");
      expect(result).toBe("Line 1\n\nLine 2");
    });
  });

  describe("reconcilePhrases", () => {
    it("updates text in-place when block count matches phrase count", () => {
      const oldPhrases: Phrase[] = [
        { phrase_id: "p1", start_time: 1.0, end_time: 3.0, duration: 2.0, speaker: "Me", text: "Original one" },
        { phrase_id: "p2", start_time: 3.5, end_time: 6.0, duration: 2.5, speaker: "Them", text: "Original two" },
      ];
      const blocks = [
        { speaker: "Me" as const, text: "Edited one" },
        { speaker: "Them" as const, text: "Edited two" },
      ];

      const reconciled = reconcilePhrases(oldPhrases, blocks, 10.0);
      expect(reconciled).toHaveLength(2);
      expect(reconciled[0].phrase_id).toBe("p1");
      expect(reconciled[0].start_time).toBe(1.0);
      expect(reconciled[0].end_time).toBe(3.0);
      expect(reconciled[0].text).toBe("Edited one");

      expect(reconciled[1].phrase_id).toBe("p2");
      expect(reconciled[1].start_time).toBe(3.5);
      expect(reconciled[1].end_time).toBe(6.0);
      expect(reconciled[1].text).toBe("Edited two");
    });

    it("deletes deleted phrases while preserving surviving phrases' metadata", () => {
      const oldPhrases: Phrase[] = [
        { phrase_id: "p1", start_time: 1.0, end_time: 3.0, duration: 2.0, speaker: "Me", text: "Alpha" },
        { phrase_id: "p2", start_time: 3.5, end_time: 6.0, duration: 2.5, speaker: "Them", text: "Beta to delete" },
        { phrase_id: "p3", start_time: 7.0, end_time: 9.0, duration: 2.0, speaker: "Me", text: "Gamma" },
      ];
      // Beta is removed in blocks
      const blocks = [
        { speaker: "Me" as const, text: "Alpha" },
        { speaker: "Me" as const, text: "Gamma" },
      ];

      const reconciled = reconcilePhrases(oldPhrases, blocks, 10.0);
      expect(reconciled).toHaveLength(2);
      expect(reconciled[0].phrase_id).toBe("p1");
      expect(reconciled[0].text).toBe("Alpha");
      expect(reconciled[0].start_time).toBe(1.0);

      expect(reconciled[1].phrase_id).toBe("p3");
      expect(reconciled[1].text).toBe("Gamma");
      expect(reconciled[1].start_time).toBe(7.0);
    });

    it("inserts new phrases with valid interpolated timestamps", () => {
      const oldPhrases: Phrase[] = [
        { phrase_id: "p1", start_time: 1.0, end_time: 3.0, duration: 2.0, speaker: "Me", text: "Alpha" },
        { phrase_id: "p2", start_time: 7.0, end_time: 9.0, duration: 2.0, speaker: "Me", text: "Gamma" },
      ];
      // A new phrase is inserted between Alpha and Gamma
      const blocks = [
        { speaker: "Me" as const, text: "Alpha" },
        { speaker: "Them" as const, text: "Inserted middle" },
        { speaker: "Me" as const, text: "Gamma" },
      ];

      const reconciled = reconcilePhrases(oldPhrases, blocks, 10.0);
      expect(reconciled).toHaveLength(3);
      expect(reconciled[0].phrase_id).toBe("p1");
      expect(reconciled[0].text).toBe("Alpha");

      expect(reconciled[1].text).toBe("Inserted middle");
      expect(reconciled[1].speaker).toBe("Them");
      expect(reconciled[1].start_time).toBeGreaterThanOrEqual(3.0);
      expect(reconciled[1].end_time).toBeLessThanOrEqual(7.0);

      expect(reconciled[2].phrase_id).toBe("p2");
      expect(reconciled[2].text).toBe("Gamma");
    });

    it("handles empty blocks", () => {
      const oldPhrases: Phrase[] = [
        { phrase_id: "p1", start_time: 1.0, end_time: 3.0, duration: 2.0, speaker: "Me", text: "Alpha" },
      ];
      expect(reconcilePhrases(oldPhrases, [], 10.0)).toEqual([]);
    });

    it("creates phrases when old phrases was empty", () => {
      const blocks = [
        { speaker: "Me" as const, text: "First" },
        { speaker: "Them" as const, text: "Second" },
      ];
      const reconciled = reconcilePhrases([], blocks, 10.0);
      expect(reconciled).toHaveLength(2);
      expect(reconciled[0].text).toBe("First");
      expect(reconciled[1].text).toBe("Second");
      expect(reconciled[0].start_time).toBe(0);
      expect(reconciled[1].start_time).toBe(5);
    });
  });
});
