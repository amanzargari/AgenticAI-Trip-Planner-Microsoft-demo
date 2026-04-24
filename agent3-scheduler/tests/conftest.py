import pathlib, sys

for mod in list(sys.modules):
    if mod in ("tools", "worker") or mod.startswith("tools."):
        del sys.modules[mod]

AGENT_DIR = str(pathlib.Path(__file__).parent.parent)
ROOT_DIR  = str(pathlib.Path(__file__).parent.parent.parent)
for p in (AGENT_DIR, ROOT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
