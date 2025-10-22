import math
from collections import defaultdict

def cosine(a, b):
    denom = (math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(y*y for y in b)))
    if denom == 0: return 0.0
    return sum(x*y for x,y in zip(a,b)) / denom

def build_stable_user_map(speaker_turns, spk_embeds):
    # Fallback to chronological stable assignment if no embeddings
    uniq = []
    for t in speaker_turns:
        if t["speaker"] not in uniq:
            uniq.append(t["speaker"])
    return {spk: idx+1 for idx, spk in enumerate(uniq)}
