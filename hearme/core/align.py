def align_words_to_speakers(stt_result, diar):
    items = []
    for seg in stt_result["segments"]:
        for w in seg["words"]:
            items.append({"word": w["word"], "start": w["start"], "end": w["end"]})

    speaker_turns = []
    for turn, _, speaker in diar.itertracks(yield_label=True):
        start = float(turn.start)
        end = float(turn.end)
        words = [w for w in items if not (w["end"] <= start or w["start"] >= end)]
        speaker_turns.append({"speaker": speaker, "start": start, "end": end, "words": words})
    return items, speaker_turns
