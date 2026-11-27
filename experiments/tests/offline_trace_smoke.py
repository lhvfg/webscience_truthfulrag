"""
Offline smoke test for TruthfulRAG.trace_sample() -- no GPU, no API key, no heavy deps.

What is real:   pipeline.trace_sample, modules.entropy_filter, modules.predict_answer(_wo_facts),
                LLMBackend, llm/openai.py (incl. to_thread + token counting), NetworkX graph
                storage, evaluate.py metrics, trace.py, trace_schema.validate.
What is faked:  heavy imports (torch, transformers, datasets, ...) are stubbed; the OpenAI
                client returns canned answers / logprobs; graph construction and graph
                retrieval are replaced by a fixed toy graph (they need a real LLM to be useful).

Run:  python experiments/tests/offline_trace_smoke.py
"""
import asyncio
import json
import os
import sys
import tempfile
import types

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))


# --------------------------------------------------------------------- stubs
class _StubModule(types.ModuleType):
    """Module whose unknown attributes resolve to a harmless placeholder class."""
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Any


def stub(name, **attrs):
    mod = _StubModule(name)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    return mod


class _Any:
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): return self
    def __getattr__(self, _): return _Any()
    @classmethod
    def from_pretrained(cls, *a, **k): return cls()


class _Enc:
    def encode(self, s): return s.split()
    def decode(self, toks): return " ".join(toks)


class _Inflect:
    def number_to_words(self, n): return str(n)


class _VDB:
    def __init__(self, *a, **k): self._data = {}
    def upsert(self, datas): return []
    def query(self, *a, **k): return []
    def save(self): pass
    def delete(self, ids): pass


# Any (sub)module under these roots that is not already stubbed below gets an
# auto-generated permissive stub, so we don't have to enumerate every import.
import importlib.abc  # noqa: E402
import importlib.machinery  # noqa: E402

STUB_ROOTS = {"torch", "transformers", "sentence_transformers", "datasets", "tiktoken",
              "json_repair", "inflect", "rouge", "pipmaster", "nano_vectordb", "ollama",
              "huggingface_hub", "accelerate", "bitsandbytes", "vllm", "openai"}


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname.split(".")[0] in STUB_ROOTS and fullname not in sys.modules:
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        return _StubModule(spec.name)

    def exec_module(self, module):
        module.__path__ = []


sys.meta_path.insert(0, _StubFinder())

torch = stub("torch", Tensor=object, no_grad=lambda: None, float32=None, bfloat16=None, float16=None,
             cuda=types.SimpleNamespace(is_available=lambda: False, OutOfMemoryError=MemoryError))
stub("torch.nn"); stub("torch.nn.functional")
torch.nn = sys.modules["torch.nn"]; torch.nn.functional = sys.modules["torch.nn.functional"]
stub("transformers", AutoModel=_Any, AutoTokenizer=_Any, AutoModelForCausalLM=_Any,
     GenerationConfig=_Any, BitsAndBytesConfig=_Any)
stub("sentence_transformers", SentenceTransformer=_Any)
stub("datasets", Dataset=list, load_dataset=lambda *a, **k: None)
stub("tiktoken", encoding_for_model=lambda m: _Enc(), get_encoding=lambda m: _Enc())
stub("json_repair", loads=json.loads)
stub("inflect", engine=_Inflect)
stub("rouge", Rouge=_Any)
stub("pipmaster", is_installed=lambda n: True, install=lambda n: None)
stub("nano_vectordb", NanoVectorDB=_VDB)
for m in ("ollama",):
    stub(m, AsyncClient=_Any)


# ------------------------------------------------------------- fake OpenAI
class _Err(Exception):
    pass


def _ns(**k):
    return types.SimpleNamespace(**k)


def _logprobs(spread):
    """Top-5 logprobs whose normalized entropy grows with `spread`."""
    import math
    raw = [1.0] + [spread] * 4
    z = sum(raw)
    return [_ns(token=f"t{i}", logprob=math.log(r / z), bytes=None) for i, r in enumerate(raw)]


class _Completions:
    def create(self, **p):
        prompt = p["messages"][-1]["content"]
        usage = _ns(total_tokens=100)
        if p.get("logprobs"):
            # Entropy probe: the path that mentions SINALOA (the counterfactual
            # answer) makes the model "less sure" -> higher entropy.
            spread = 0.6 if "SINALOA" in prompt else (0.05 if "Path:" in prompt else 0.1)
            content = [_ns(top_logprobs=_logprobs(spread)) for _ in range(2)]
            return _ns(choices=[_ns(logprobs=_ns(content=content), message=_ns(content=""))], usage=usage)
        if "SINALOA" in prompt:
            text = '{"reason": "facts say Sinaloa", "answer": "Sinaloa"}'
        else:
            text = "Tamaulipas"
        return _ns(choices=[_ns(logprobs=None, message=_ns(content=text))], usage=usage)


class _OpenAI:
    def __init__(self, *a, **k):
        self.chat = _ns(completions=_Completions())


stub("openai", OpenAI=_OpenAI, RateLimitError=_Err, APIConnectionError=_Err,
     APITimeoutError=_Err, APIError=_Err)

# ------------------------------------------------------------- real package
import importlib  # noqa: E402

# Locate the package (the folder that contains pipeline.py) and its exported
# pipeline class without hard-coding names.
PKG = next(d for d in os.listdir(ROOT) if os.path.isfile(os.path.join(ROOT, d, "pipeline.py")))
_pkg = importlib.import_module(PKG)
TruthfulRAG = next(v for k, v in vars(_pkg).items() if isinstance(v, type) and hasattr(v, "trace_sample"))
from trace_schema import validate  # noqa: E402

PATHS = [
    "Path:\nCIUDAD DEPORTIVA->TOROS DE NUEVO LAREDO\nNodes:\n...\nEdges:\n...",
    "Path:\nMUNICIPALITY OF NUEVO LAREDO->NUEVO LAREDO->SINALOA\nNodes:\nNUEVO LAREDO: a city in Sinaloa",
    "Path:\nESTADIO NUEVO LAREDO->CIUDAD DEPORTIVA\nNodes:\n...",
]


async def fake_make_kg(self, sample, **kw):
    g = self.chunk_entity_relation_graph
    await g.clear_graph() if hasattr(g, "clear_graph") else None
    g._graph.clear()
    await g.upsert_node('"NUEVO LAREDO"', {"description": "A city in the Mexican state of Sinaloa."})
    await g.upsert_node('"CIUDAD DEPORTIVA"', {"description": "A sports complex in Nuevo Laredo."})
    await g.upsert_edge('"CIUDAD DEPORTIVA"', '"NUEVO LAREDO"', {"description": "located in"})


async def fake_retrieve(self, sample, **kw):
    return list(PATHS)


TruthfulRAG.make_knowledge_graph = fake_make_kg
TruthfulRAG.knowledge_graph_retrieve = fake_retrieve


def load(name, idx=0):
    with open(os.path.join(ROOT, "datas", name), encoding="utf-8") as f:
        return json.load(f)[idx]


async def main():
    os.environ.setdefault("OPENAI_API_KEY", "offline-test")
    neg = load("musique_negative.json")
    gold = load("musique_golden.json")
    assert neg["id"] == gold["id"]

    tmp = tempfile.mkdtemp(prefix="cl_smoke_")

    def make(i):
        return TruthfulRAG(dataset=[], backend_type="openai", model_name="gpt-4o-mini",
                          similarity_model="all-MiniLM-L6-v2", threshold=1,
                          output_dir=os.path.join(tmp, f"out{i}"),
                          working_dir=os.path.join(tmp, f"kg{i}"))

    workers = [make(i) for i in range(3)]
    methods = ["wo_rag", "w_rag", "TruthfulRAG"]
    # Run the three methods CONCURRENTLY on three separate instances to check
    # that per-sample cost counters do not leak into each other.
    recs = await asyncio.gather(*[
        w.trace_sample(neg, m, "musique_neg", gold_true=gold["answer"])
        for w, m in zip(workers, methods)
    ])

    ok = True
    for r in recs:
        errs = validate(json.loads(json.dumps(r, ensure_ascii=False)))
        print(f"[{r['method']:>10}] pred={r['pred']!r:14} correct={r['correct']!s:5} "
              f"em={r['em']!s:5} calls={r['n_llm_calls']} tokens={r['n_tokens']} errs={errs}")
        ok &= not errs

    cl = recs[2]
    print("  candidates:", [(c["idx"], round(c["dH"], 3), c["has_answer"]) for c in cl["candidates"]])
    print("  kept:", cl["kept"], " graph:", cl["graph"], " gold_true:", cl["gold_true"])

    expect_calls = {"wo_rag": 1, "w_rag": 1, "TruthfulRAG": 1 + len(PATHS) + 1}
    for r in recs:
        if r["n_llm_calls"] != expect_calls[r["method"]]:
            ok = False
            print(f"  !! {r['method']}: expected {expect_calls[r['method']]} calls, got {r['n_llm_calls']}")
    if cl["candidates"][1]["has_answer"] is not True or cl["candidates"][0]["has_answer"] is not False:
        ok = False
        print("  !! has_answer labelling wrong")
    if cl["kept"][0] != 1:
        ok = False
        print("  !! highest-dH path should be kept first")

    print("SMOKE TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
