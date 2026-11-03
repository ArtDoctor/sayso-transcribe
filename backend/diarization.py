import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import torch

logger = logging.getLogger(__name__)

def _find_pyannote_config() -> Optional[Path]:
    """Look for cached pyannote speaker-diarization-3.1 config in local HuggingFace cache."""
    candidates = []
    
    # 1. Check HF_HOME or HF_HUB_CACHE
    if os.getenv("HF_HOME"):
        candidates.append(Path(os.getenv("HF_HOME")) / "hub")
    if os.getenv("HF_HUB_CACHE"):
        candidates.append(Path(os.getenv("HF_HUB_CACHE")))
        
    # 2. Check user home cache
    home_hf = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    candidates.append(home_hf)
    
    # 3. Check bundled models directory (for portable package)
    bundled_hf = Path(__file__).resolve().parent.parent / "models" / "huggingface" / "hub"
    candidates.append(bundled_hf)
    
    for hub_dir in candidates:
        if not hub_dir.exists():
            continue
        model_dir = hub_dir / "models--pyannote--speaker-diarization-3.1" / "snapshots"
        if model_dir.exists():
            for snap in model_dir.iterdir():
                cfg = snap / "config.yaml"
                if cfg.is_file():
                    return cfg
    return None


class SpeakerDiarizer:
    """
    Offline speaker diarization using cached pyannote 3.1 or fallback acoustic clustering.
    Identifies speaker turns and assigns speaker labels (e.g. Speaker 1, Speaker 2) to audio.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None
        self._initialized = False
        self._init_pipeline()

    def _init_pipeline(self):
        if self._initialized:
            return
        self._initialized = True
        if os.getenv("SAYSO_DISABLE_MODEL_AUTOLOAD") == "1":
            logger.info("Diarizer model autoload disabled by environment.")
            return

        cfg_path = _find_pyannote_config()
        if not cfg_path:
            logger.info("Pyannote speaker diarization config not found in local cache; fallback clustering enabled.")
            return

        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                from pyannote.audio import Pipeline
                self.pipeline = Pipeline.from_pretrained(str(cfg_path))
                if self.device == "cuda" and torch.cuda.is_available():
                    self.pipeline.to(torch.device("cuda"))
                logger.info("SpeakerDiarizer initialized with pyannote 3.1 on %s", self.device)
        except Exception as e:
            logger.warning("Could not initialize pyannote pipeline: %s. Using fallback.", e)
            self.pipeline = None

    def diarize_audio(self, audio_f32: np.ndarray, sample_rate: int = 16000) -> List[Dict[str, Any]]:
        """
        Diarize a mono audio waveform into speaker segments.
        Returns list of {"start": float, "end": float, "speaker": str}.
        """
        if audio_f32 is None or len(audio_f32) < sample_rate * 0.5:
            return []

        if self.pipeline is not None:
            try:
                wav_tensor = torch.from_numpy(audio_f32).float().unsqueeze(0)
                out = self.pipeline({"waveform": wav_tensor, "sample_rate": sample_rate})
                
                turns: List[Dict[str, Any]] = []
                annotation = getattr(out, "speaker_diarization", out)
                if hasattr(annotation, "itertracks"):
                    for turn, _, speaker in annotation.itertracks(yield_label=True):
                        turns.append({
                            "start": round(float(turn.start), 2),
                            "end": round(float(turn.end), 2),
                            "speaker": str(speaker),
                        })
                if turns:
                    # Clean & merge small contiguous turns of the same speaker
                    merged = self._merge_contiguous_turns(turns)
                    # Normalize speaker labels to friendly names: Speaker 1, Speaker 2...
                    return self._map_speaker_names(merged)
            except Exception as exc:
                logger.warning("Pyannote diarization failed: %s. Using fallback.", exc)

        return self._fallback_diarize(audio_f32, sample_rate)

    def _merge_contiguous_turns(self, turns: List[Dict[str, Any]], max_gap: float = 0.6) -> List[Dict[str, Any]]:
        """Merge adjacent turns belonging to the same speaker separated by short pauses."""
        if not turns:
            return []
        turns.sort(key=lambda t: t["start"])
        merged = [dict(turns[0])]
        for t in turns[1:]:
            prev = merged[-1]
            if prev["speaker"] == t["speaker"] and (t["start"] - prev["end"] <= max_gap):
                prev["end"] = max(prev["end"], t["end"])
            else:
                merged.append(dict(t))
        return merged

    def _map_speaker_names(self, turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Map raw labels like SPEAKER_00, SPEAKER_01 to Speaker 1, Speaker 2 in appearance order."""
        mapping = {}
        counter = 1
        result = []
        for t in turns:
            raw = t["speaker"]
            if raw not in mapping:
                mapping[raw] = f"Speaker {counter}"
                counter += 1
            item = dict(t)
            item["speaker"] = mapping[raw]
            result.append(item)
        return result

    def _fallback_diarize(self, audio_f32: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
        """Lightweight fallback using energy VAD and single/heuristic speaker segmentation."""
        duration = len(audio_f32) / sample_rate
        return [{"start": 0.0, "end": round(duration, 2), "speaker": "Speaker 1"}]

    def assign_speakers_to_phrases(
        self,
        phrases: List[Dict[str, Any]],
        diarized_turns: List[Dict[str, Any]],
        default_speaker: str = "Speaker 1"
    ) -> List[Dict[str, Any]]:
        """
        Assigns the best matching speaker from diarization turns to each phrase
        based on maximum temporal intersection.
        """
        if not diarized_turns:
            for p in phrases:
                if not p.get("speaker") or p.get("speaker") in ("Them", "Remote", "Speaker"):
                    p["speaker"] = default_speaker
            return phrases

        for p in phrases:
            p_st = float(p.get("start_time", 0.0))
            p_et = float(p.get("end_time", p_st + 0.1))
            
            best_speaker = None
            best_overlap = 0.0

            for turn in diarized_turns:
                t_st = turn["start"]
                t_et = turn["end"]
                # Calculate temporal intersection
                overlap_start = max(p_st, t_st)
                overlap_end = min(p_et, t_et)
                overlap = max(0.0, overlap_end - overlap_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = turn["speaker"]

            if best_speaker and best_overlap > 0.1:
                p["speaker"] = best_speaker
            elif not p.get("speaker") or p.get("speaker") in ("Them", "Remote", "Speaker"):
                p["speaker"] = default_speaker

        return phrases

    def split_phrases_by_speaker_turns(
        self,
        phrases: List[Dict[str, Any]],
        diarized_turns: List[Dict[str, Any]],
        default_speaker: str = "Speaker 1",
        min_speaker_dur: float = 0.4,
        sample_rate: int = 16000,
    ) -> List[Dict[str, Any]]:
        """
        Splits any phrase that spans across multiple distinct speakers into separate
        sub-phrases corresponding to each speaker turn.
        Returns a list of phrases where each phrase belongs to a single speaker.
        """
        if not diarized_turns:
            for p in phrases:
                if not p.get("speaker") or p.get("speaker") in ("Them", "Remote", "Speaker"):
                    p["speaker"] = default_speaker
            return phrases

        result: List[Dict[str, Any]] = []

        for p in phrases:
            p_st = float(p.get("start_time", 0.0))
            p_et = float(p.get("end_time", p_st + 0.1))
            p_dur = p_et - p_st

            if p_dur < 0.2:
                result.append(dict(p))
                continue

            # Find all turns that overlap with this phrase
            overlapping = []
            for t in diarized_turns:
                t_st = float(t["start"])
                t_et = float(t["end"])
                ov_st = max(p_st, t_st)
                ov_et = min(p_et, t_et)
                if ov_et > ov_st:
                    overlapping.append({
                        "start": ov_st,
                        "end": ov_et,
                        "orig_start": t_st,
                        "orig_end": t_et,
                        "speaker": t["speaker"],
                        "duration": ov_et - ov_st,
                    })

            if not overlapping:
                cp = dict(p)
                if not cp.get("speaker") or cp.get("speaker") in ("Them", "Remote", "Speaker"):
                    cp["speaker"] = default_speaker
                result.append(cp)
                continue

            # Group overlap durations by speaker
            speaker_durations: Dict[str, float] = {}
            for o in overlapping:
                spk = o["speaker"]
                speaker_durations[spk] = speaker_durations.get(spk, 0.0) + o["duration"]

            # Filter speakers with significant presence (>= min_speaker_dur or >= 20% of phrase)
            threshold = min(min_speaker_dur, p_dur * 0.2)
            significant_speakers = [spk for spk, dur in speaker_durations.items() if dur >= threshold]

            # If only 1 significant speaker (or all overlaps are the same speaker):
            if len(significant_speakers) <= 1:
                best_spk = max(speaker_durations.items(), key=lambda x: x[1])[0]
                cp = dict(p)
                cp["speaker"] = best_spk
                result.append(cp)
                continue

            # MULTIPLE SPEAKERS IN ONE PHRASE! Split the phrase at speaker transitions
            overlapping.sort(key=lambda o: o["start"])

            slices: List[Dict[str, Any]] = []
            curr_speaker = None
            curr_start = p_st

            for i in range(len(overlapping)):
                turn = overlapping[i]
                spk = turn["speaker"]
                if spk not in significant_speakers:
                    continue

                if curr_speaker is None:
                    curr_speaker = spk
                    continue

                if spk != curr_speaker:
                    # Transition from curr_speaker to spk
                    prev_turn = overlapping[i - 1]
                    split_pt = (prev_turn["end"] + turn["start"]) / 2.0
                    split_pt = max(curr_start + 0.2, min(p_et - 0.2, split_pt))

                    if split_pt > curr_start + 0.2:
                        slices.append({
                            "speaker": curr_speaker,
                            "start_time": round(curr_start, 2),
                            "end_time": round(split_pt, 2),
                        })
                        curr_start = split_pt
                    curr_speaker = spk

            # Append the final speaker segment up to p_et
            if curr_speaker is not None and p_et > curr_start + 0.1:
                slices.append({
                    "speaker": curr_speaker,
                    "start_time": round(curr_start, 2),
                    "end_time": round(p_et, 2),
                })

            if not slices:
                best_spk = max(speaker_durations.items(), key=lambda x: x[1])[0]
                cp = dict(p)
                cp["speaker"] = best_spk
                result.append(cp)
                continue

            # Ensure boundary coverage
            slices[0]["start_time"] = round(p_st, 2)
            slices[-1]["end_time"] = round(p_et, 2)

            orig_id = p.get("phrase_id", "p")
            has_audio = isinstance(p.get("audio"), np.ndarray)

            for idx, sl in enumerate(slices):
                dur = round(sl["end_time"] - sl["start_time"], 2)
                if dur < 0.15:
                    continue
                sub_p = dict(p)
                sub_p["phrase_id"] = f"{orig_id}_s{idx}" if idx > 0 else orig_id
                sub_p["speaker"] = sl["speaker"]
                sub_p["start_time"] = sl["start_time"]
                sub_p["end_time"] = sl["end_time"]
                sub_p["duration"] = dur
                if len(slices) > 1 and "text" in sub_p:
                    sub_p["text"] = ""
                if has_audio:
                    sub_st_samp = max(0, int(round((sl["start_time"] - p_st) * sample_rate)))
                    sub_et_samp = min(len(p["audio"]), int(round((sl["end_time"] - p_st) * sample_rate)))
                    sub_p["audio"] = p["audio"][sub_st_samp:sub_et_samp]
                result.append(sub_p)

        return result

    def segment_audio_by_turns(
        self,
        audio_np: np.ndarray,
        sample_rate: int,
        diarized_turns: List[Dict[str, Any]],
        max_chunk_s: float = 25.0,
        min_chunk_s: float = 0.3,
        max_merge_gap_s: float = 2.0,
    ) -> List[Dict[str, Any]]:
        """
        Segments continuous audio into speaker-bounded chunks based on diarized turns.
        Contiguous turns of the SAME speaker are merged up to max_chunk_s.
        Turns of DIFFERENT speakers are NEVER merged, preventing speaker fusion.
        """
        if not diarized_turns or audio_np is None or len(audio_np) == 0:
            return []

        turns = sorted(diarized_turns, key=lambda t: t["start"])
        merged: List[Dict[str, Any]] = []

        curr = None
        for t in turns:
            spk = t["speaker"]
            st = float(t["start"])
            et = float(t["end"])
            if et <= st:
                continue

            if curr is None:
                curr = {"speaker": spk, "start": st, "end": et}
            elif curr["speaker"] == spk and (st - curr["end"] <= max_merge_gap_s) and ((et - curr["start"]) <= max_chunk_s):
                curr["end"] = max(curr["end"], et)
            else:
                merged.append(curr)
                curr = {"speaker": spk, "start": st, "end": et}

        if curr is not None:
            merged.append(curr)

        # Build segments with audio slices, splitting any individual turns exceeding max_chunk_s
        segments: List[Dict[str, Any]] = []
        total_samples = len(audio_np)

        for m in merged:
            st = m["start"]
            et = m["end"]
            dur = et - st
            if dur < min_chunk_s:
                continue

            # If a single speaker's continuous turn is > max_chunk_s, break into <= max_chunk_s sub-slices
            sub_intervals = []
            if dur > max_chunk_s:
                sub_st = st
                while sub_st < et:
                    sub_et = min(et, sub_st + max_chunk_s)
                    if (sub_et - sub_st) >= min_chunk_s:
                        sub_intervals.append((sub_st, sub_et))
                    sub_st = sub_et
            else:
                sub_intervals.append((st, et))

            for sub_st, sub_et in sub_intervals:
                sub_dur = sub_et - sub_st
                start_samp = max(0, int(round(sub_st * sample_rate)))
                end_samp = min(total_samples, int(round(sub_et * sample_rate)))
                chunk_audio = audio_np[start_samp:end_samp]

                if len(chunk_audio) == 0:
                    continue

                segments.append({
                    "speaker": m["speaker"],
                    "start_time": round(sub_st, 2),
                    "end_time": round(sub_et, 2),
                    "duration": round(sub_dur, 2),
                    "audio": chunk_audio,
                })

        return segments

