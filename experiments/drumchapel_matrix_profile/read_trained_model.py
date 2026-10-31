"""
read_trained_model.py  -  open trained_model_cache.pkl and export what is inside
==================================================================================
Put this file in the same folder as leak_localisation_net1.py and run it AFTER that script has run once.

The .pkl file is a Python "pickle": the trained encoder-decoder + the reference embedding database,
saved as Python objects. Excel cannot open it - this script reads it and writes:
    all_embeddings.csv   one row per simulated training leak: pipe, position, zone, leak size, time, z1 ... z16
and prints a short summary.
"""
import pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SCRIPT = "leak_localisation_net1"            # or "drumchapel_detection_localisation"
PKL = HERE / "net1_leak_localisation_outputs" / "trained_model_cache.pkl"

# The pickle refers to the classes (EncoderDecoder, EmbeddingLocaliser) of the script that created it,
# so they must be loaded first. Importing the script only defines them - it does NOT run the study.
sys.path.insert(0, str(HERE))
module = __import__(SCRIPT)
main = sys.modules["__main__"]
for name in ("EncoderDecoder", "EmbeddingLocaliser"):
    setattr(main, name, getattr(module, name))

with open(PKL, "rb") as f:
    cache = pickle.load(f)

loc = cache["loc"]          # the reference embedding database (+ trained model in loc.model)
meta = cache["meta"]        # what each simulated leak was (pipe, position, zone, size, time ...)
X = cache["X"]              # fingerprints = input of the encoder (one row per simulated leak)

print("What is inside the pickle:")
print(f"  X     fingerprints (encoder input)     : {X.shape[0]} leaks x {X.shape[1]} sensors")
print(f"  loc.Z embeddings (bottleneck output)   : {loc.Z.shape[0]} leaks x {loc.Z.shape[1]} numbers")
print(f"  loc.model                              : the trained encoder-decoder (weights {list(loc.model.W)})")
print(f"  meta                                   : description of every simulated leak {list(meta.columns)}")
print(f"  report / per_pipe / hist               : validation results and training history")

emb = pd.concat([meta.reset_index(drop=True),
                 pd.DataFrame(np.round(loc.Z, 5), columns=[f"z{i + 1}" for i in range(loc.Z.shape[1])])], axis=1)
out = PKL.parent / "all_embeddings.csv"
emb.to_csv(out, index=False)
print(f"\nAll embeddings written to: {out}")
print("\nFirst rows:")
print(emb.head().to_string(index=False))

# Example: how many stored embeddings per pipe
print("\nStored embeddings per pipe:")
print(emb.groupby("Pipe").size().to_string())
