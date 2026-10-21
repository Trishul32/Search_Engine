"""Paper -> knowledge graph.

Turns a set of papers into an explorable graph of the research landscape:

    Paper A --introduces--> Method X --evaluated on--> Dataset Z
                               |                          ^
                           improves                       |
                               v                    Paper B uses it
                            Metric Y          Paper B --extends--> Method X

Entities are extracted from title + abstract (metadata only, so building a
graph never downloads PDFs):

* ``heuristic`` extractor (default, fast, deterministic): curated
  gazetteers of methods / datasets / metrics / tasks + linguistic patterns
  for what a paper *introduces* ("we propose X", "X: ..." titles,
  "called X", acronym definitions) and for claims ("improves", "extends").
* ``llm`` extractor: asks the configured LLM for the same schema as JSON
  and merges it with the heuristic result (falls back on any failure).

Cross-paper links are what make the graph useful: when paper B mentions a
method that paper A introduced, B --uses/extends--> A's method.
Extractions are cached per (paper, extractor) in the ``paper_entities``
table.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field

from sqlalchemy import select

from backend.app.core.config import settings
from backend.app.db.models import Paper, PaperEntities
from backend.app.db.session import db_session
from backend.app.services.relevance import mentions

logger = logging.getLogger(__name__)

HEURISTIC_VERSION = "heuristic-v4"
LLM_VERSION = "llm-v1"

# --------------------------------------------------------------------------
# Gazetteers: display name -> aliases (matched word-bounded; short all-caps
# aliases are case-sensitive, see relevance.mentions)
# --------------------------------------------------------------------------
_METHODS: dict[str, list[str]] = {
    "Transformer": ["transformer", "transformers", "self-attention"],
    "BERT": ["BERT"], "RoBERTa": ["RoBERTa"], "T5": ["T5"], "BART": ["BART"],
    "GPT-4": ["GPT-4"], "GPT-3": ["GPT-3"], "GPT-2": ["GPT-2"], "ChatGPT": ["ChatGPT"],
    "LLaMA": ["LLaMA", "Llama 2", "Llama-2", "Llama 3"], "CLIP": ["CLIP"],
    "Vision Transformer": ["vision transformer", "ViT"],
    "Large Language Models": ["large language model", "large language models", "LLM", "LLMs"],
    "Retrieval-Augmented Generation": ["retrieval-augmented generation", "retrieval augmented generation", "RAG"],
    "BM25": ["BM25"], "TF-IDF": ["TF-IDF", "tf-idf"],
    "Dense Retrieval": ["dense retrieval", "dense retriever", "dense retrievers"],
    "Sparse Retrieval": ["sparse retrieval", "lexical retrieval"],
    "Hybrid Retrieval": ["hybrid retrieval", "hybrid search"],
    "DPR": ["DPR", "dense passage retrieval", "dense passage retriever"], "ColBERT": ["ColBERT"], "SPLADE": ["SPLADE"], "Contriever": ["Contriever"],
    "Cross-Encoder": ["cross-encoder", "cross encoder"], "Bi-Encoder": ["bi-encoder", "dual-encoder", "dual encoder"],
    "Reranking": ["reranking", "re-ranking", "reranker", "rerankers"],
    "Sentence-BERT": ["Sentence-BERT", "SBERT"],
    "FAISS": ["FAISS"], "HNSW": ["HNSW"],
    "Approximate Nearest Neighbor Search": ["approximate nearest neighbor", "approximate nearest neighbour", "ANN search"],
    "Knowledge Distillation": ["knowledge distillation", "distillation"],
    "Contrastive Learning": ["contrastive learning", "contrastive loss"],
    "Attention Mechanism": ["attention mechanism"],
    "Graph Neural Network": ["graph neural network", "graph neural networks", "GNN", "GNNs"],
    "GCN": ["GCN", "graph convolutional network"], "GAT": ["graph attention network"],
    "CNN": ["convolutional neural network", "convolutional neural networks", "CNN", "CNNs"],
    "RNN": ["recurrent neural network", "RNN", "RNNs"], "LSTM": ["LSTM", "long short-term memory"],
    "GAN": ["generative adversarial network", "GAN", "GANs"],
    "VAE": ["variational autoencoder", "VAE"],
    "Diffusion Model": ["diffusion model", "diffusion models", "denoising diffusion"],
    "ResNet": ["ResNet"], "U-Net": ["U-Net"], "YOLO": ["YOLO"],
    "Word2Vec": ["word2vec"], "GloVe": ["GloVe"],
    "XGBoost": ["XGBoost"], "Random Forest": ["random forest", "random forests"],
    "Support Vector Machine": ["support vector machine", "SVM", "SVMs"],
    "Logistic Regression": ["logistic regression"],
    "k-Means": ["k-means"], "PCA": ["principal component analysis", "PCA"],
    "Reinforcement Learning": ["reinforcement learning"],
    "Deep Reinforcement Learning": ["deep reinforcement learning"],
    "PPO": ["PPO", "proximal policy optimization"], "DQN": ["DQN", "deep Q-network"],
    "RLHF": ["RLHF", "reinforcement learning from human feedback"], "DPO": ["DPO", "direct preference optimization"],
    "Monte Carlo Tree Search": ["Monte Carlo tree search", "MCTS"],
    "Chain-of-Thought": ["chain-of-thought", "chain of thought", "CoT"],
    "In-Context Learning": ["in-context learning"],
    "Instruction Tuning": ["instruction tuning", "instruction-tuned"],
    "Fine-Tuning": ["fine-tuning", "fine-tuned", "finetuning"],
    "LoRA": ["LoRA", "low-rank adaptation"], "Prompt Tuning": ["prompt tuning", "prompt learning"],
    "Mixture of Experts": ["mixture of experts", "mixture-of-experts", "MoE"],
    "Mamba": ["Mamba"], "State Space Model": ["state space model", "state-space model"],
    "Federated Learning": ["federated learning"], "Transfer Learning": ["transfer learning"],
    "Meta-Learning": ["meta-learning", "meta learning"], "Few-Shot Learning": ["few-shot learning", "few-shot"],
    "Zero-Shot Learning": ["zero-shot"], "Self-Supervised Learning": ["self-supervised"],
    "Semi-Supervised Learning": ["semi-supervised"], "Active Learning": ["active learning"],
    "Continual Learning": ["continual learning", "lifelong learning", "incremental learning"],
    "Experience Replay": ["experience replay", "replay buffer"],
    "Quantization": ["quantization"], "Pruning": ["pruning"],
    "Knowledge Graph": ["knowledge graph", "knowledge graphs"],
    "Beam Search": ["beam search"], "Dynamic Programming": ["dynamic programming"],
    "Linear Programming": ["linear programming", "integer programming"],
    "Gradient Descent": ["gradient descent", "stochastic gradient descent", "SGD"],
    "Adam": ["Adam optimizer"], "Bayesian Optimization": ["Bayesian optimization"],
    "Gaussian Process": ["Gaussian process", "Gaussian processes"],
    "Hidden Markov Model": ["hidden Markov model", "HMM"],
    "Learning from Demonstration": ["learning from demonstration", "imitation learning", "behavior cloning"],
    "Model Predictive Control": ["model predictive control", "MPC"],
    "Locality-Sensitive Hashing": ["locality-sensitive hashing", "locality sensitive hashing", "LSH"],
    "Approximation Algorithm": ["approximation algorithm", "approximation scheme"],
    "Depth-First Search": ["depth-first search", "DFS"], "Breadth-First Search": ["breadth-first search", "BFS"],
    "Data Augmentation": ["data augmentation"], "Ensemble Learning": ["ensemble"],
    "Prompt Engineering": ["prompting", "prompt engineering"],
    "Tool Use": ["tool use", "tool-use", "external tools"],
    "Agents": ["LLM agent", "LLM agents", "autonomous agent", "autonomous agents"],
}

_DATASETS: dict[str, list[str]] = {
    "MS MARCO": ["MS MARCO", "MSMARCO"], "Natural Questions": ["Natural Questions"],
    "TriviaQA": ["TriviaQA"], "HotpotQA": ["HotpotQA"], "SQuAD": ["SQuAD"],
    "BEIR": ["BEIR"], "KILT": ["KILT"], "FEVER": ["FEVER"], "WebQuestions": ["WebQuestions"],
    "MMLU": ["MMLU"], "GSM8K": ["GSM8K"], "HumanEval": ["HumanEval"], "MBPP": ["MBPP"],
    "TruthfulQA": ["TruthfulQA"], "HellaSwag": ["HellaSwag"], "BIG-bench": ["BIG-bench"],
    "GLUE": ["GLUE"], "SuperGLUE": ["SuperGLUE"], "SNLI": ["SNLI"], "MultiNLI": ["MultiNLI", "MNLI"],
    "STS Benchmark": ["STS-B", "STS benchmark", "semantic textual similarity benchmark"],
    "ImageNet": ["ImageNet"], "CIFAR-10": ["CIFAR-10"], "CIFAR-100": ["CIFAR-100"],
    "MNIST": ["MNIST"], "COCO": ["MS COCO", "MSCOCO", "COCO"], "PASCAL VOC": ["PASCAL VOC"],
    "Cityscapes": ["Cityscapes"], "ADE20K": ["ADE20K"], "KITTI": ["KITTI"], "nuScenes": ["nuScenes"],
    "LibriSpeech": ["LibriSpeech"], "WikiText-103": ["WikiText-103", "WikiText"],
    "Penn Treebank": ["Penn Treebank"], "C4": ["C4 corpus"], "The Pile": ["the Pile"],
    "Wikipedia": ["Wikipedia"], "PubMed": ["PubMed"], "PubMedQA": ["PubMedQA"], "BioASQ": ["BioASQ"],
    "SciFact": ["SciFact"], "SciDocs": ["SciDocs", "SCIDOCS"], "TREC-COVID": ["TREC-COVID"],
    "TREC DL": ["TREC DL", "TREC Deep Learning"], "Cora": ["Cora"], "Citeseer": ["Citeseer", "CiteSeer"],
    "OGB": ["Open Graph Benchmark", "OGB"], "WMT": ["WMT"], "CNN/DailyMail": ["CNN/DailyMail", "CNN/DM"],
    "XSum": ["XSum"], "CoNLL-2003": ["CoNLL-2003", "CoNLL"], "IMDB": ["IMDB"], "SST-2": ["SST-2"],
    "MovieLens": ["MovieLens"], "Atari": ["Atari"], "MuJoCo": ["MuJoCo"], "D4RL": ["D4RL"],
    "MIMIC-III": ["MIMIC-III", "MIMIC"], "CheXpert": ["CheXpert"], "NSL-KDD": ["NSL-KDD"],
    "CICIDS2017": ["CICIDS2017", "CIC-IDS2017"], "UNSW-NB15": ["UNSW-NB15"], "KDD Cup 99": ["KDD Cup 99", "KDD99"],
    "LongBench": ["LongBench"], "Qasper": ["Qasper"], "Mind2Web": ["Mind2Web"], "ALFWorld": ["ALFWorld"],
    "miniImageNet": ["miniImageNet"], "Omniglot": ["Omniglot"], "ShapeNet": ["ShapeNet"],
    "ScanNet": ["ScanNet"], "CelebA": ["CelebA"], "LSUN": ["LSUN"], "Kinetics": ["Kinetics-400", "Kinetics"],
}

_METRICS: dict[str, list[str]] = {
    "Accuracy": ["accuracy", "top-1 accuracy", "top-5 accuracy"],
    "F1": ["F1", "F1-score", "F1 score", "macro-F1", "micro-F1"],
    "Precision/Recall": ["precision and recall", "precision-recall"],
    "Recall@k": ["recall@k", "recall@1", "recall@5", "recall@10", "recall@20", "recall@100", "recall at"],
    "nDCG": ["nDCG", "NDCG", "nDCG@10"], "MRR": ["MRR", "mean reciprocal rank"],
    "MAP": ["mean average precision", "mAP"], "AUC": ["AUC", "ROC-AUC", "AUROC"],
    "BLEU": ["BLEU"], "ROUGE": ["ROUGE", "ROUGE-L"], "METEOR": ["METEOR"], "BERTScore": ["BERTScore"],
    "Perplexity": ["perplexity"], "Exact Match": ["exact match"],
    "Word Error Rate": ["word error rate", "WER"], "FID": ["FID", "Frechet inception distance"],
    "PSNR": ["PSNR"], "SSIM": ["SSIM"], "IoU": ["IoU", "mIoU", "intersection over union"],
    "Latency": ["latency"], "Throughput": ["throughput"], "Speedup": ["speedup", "speed-up"],
    "Memory Usage": ["memory usage", "memory footprint", "memory consumption"],
    "Energy Efficiency": ["energy consumption", "energy efficiency"],
    "Sample Efficiency": ["sample efficiency", "sample-efficient"],
    "Faithfulness": ["faithfulness", "factuality", "factual consistency"],
    "Hallucination Rate": ["hallucination", "hallucinations", "hallucinate"],
    "Robustness": ["robustness"], "Calibration": ["calibration"],
    "Success Rate": ["success rate"], "Reward": ["cumulative reward", "episodic return"],
    "RMSE": ["RMSE", "root mean squared error"], "MAE": ["MAE", "mean absolute error"],
    "Approximation Ratio": ["approximation ratio", "approximation factor"],
    "Time Complexity": ["running time", "time complexity"],
    "Communication Cost": ["communication cost", "communication overhead"],
}

_TASKS: dict[str, list[str]] = {
    "Question Answering": ["question answering", "QA"],
    "Open-Domain QA": ["open-domain question answering", "open-domain QA"],
    "Multi-hop QA": ["multi-hop question answering", "multi-hop QA", "multi-hop reasoning"],
    "Information Retrieval": ["information retrieval", "document retrieval", "passage retrieval"],
    "Semantic Search": ["semantic search"], "Recommendation": ["recommendation", "recommender system", "recommender systems"],
    "Text Classification": ["text classification"], "Sentiment Analysis": ["sentiment analysis"],
    "Machine Translation": ["machine translation"], "Summarization": ["summarization"],
    "Text Generation": ["text generation", "language generation"], "Code Generation": ["code generation"],
    "Named Entity Recognition": ["named entity recognition", "NER"], "Relation Extraction": ["relation extraction"],
    "Fact Verification": ["fact verification", "fact-checking", "fact checking"],
    "Language Modeling": ["language modeling", "language modelling"],
    "Dialogue": ["dialogue system", "dialogue systems", "conversational agent", "chatbot"],
    "Reasoning": ["mathematical reasoning", "commonsense reasoning", "logical reasoning"],
    "Image Classification": ["image classification"], "Object Detection": ["object detection"],
    "Image Segmentation": ["image segmentation", "semantic segmentation", "instance segmentation"],
    "Image Generation": ["image generation", "text-to-image"], "Image Captioning": ["image captioning"],
    "Visual Question Answering": ["visual question answering", "VQA"],
    "Speech Recognition": ["speech recognition", "ASR"],
    "Link Prediction": ["link prediction"], "Node Classification": ["node classification"],
    "Anomaly Detection": ["anomaly detection"], "Intrusion Detection": ["intrusion detection"],
    "Fraud Detection": ["fraud detection"], "Time Series Forecasting": ["time series forecasting", "forecasting"],
    "Robotic Manipulation": ["robotic manipulation", "robot manipulation", "manipulation skills"],
    "Autonomous Driving": ["autonomous driving", "self-driving"],
    "Drug Discovery": ["drug discovery"], "Protein Structure Prediction": ["protein structure prediction"],
    "Scheduling": ["scheduling"], "Routing": ["routing"], "Clustering": ["clustering"],
    "Graph Partitioning": ["graph partitioning"], "Shortest Paths": ["shortest path", "shortest paths"],
    "Congestion Control": ["congestion control"], "Caching": ["caching"],
    "Program Synthesis": ["program synthesis"], "Theorem Proving": ["theorem proving"],
    "Chunking": ["chunking", "document segmentation"],
}

_ENTITY_TABLES = {
    "method": _METHODS, "dataset": _DATASETS, "metric": _METRICS, "task": _TASKS,
}

_IMPROVE_VERB = re.compile(
    r"\b(?:improv\w*|outperform\w*|boost\w*|increas\w*|reduc\w*|lower\w*|"
    r"surpass\w*|exceed\w*|gains?|better|state[- ]of[- ]the[- ]art|sota|"
    r"speed\w*\s+up|faster|higher)\b",
    re.I,
)
_EXTEND_VERB = re.compile(
    r"\b(?:extend\w*|build\w*\s+(?:up)?on|built\s+(?:up)?on|improv\w*\s+(?:up)?on|"
    r"modif\w*|variant\s+of|generali[sz]\w*|adapt\w*|augment\w*)\b",
    re.I,
)
_PROPOSE = re.compile(
    r"(?:\bwe\s+(?:propose|present|introduce|develop|design|describe|devise|release)|"
    r"^(?:introduces|proposes|presents))\s+"
    r"(?:(?:a|an|the|our|two|three)\s+)?"
    r"(?:(?:novel|new|simple|effective|efficient|general|unified|scalable|first|"
    r"lightweight|principled|systematic|comprehensive|yet|and|robust|practical|"
    r"generic|flexible|fast|end-to-end|training-free|plug-and-play)[\s,]+)*"
    r"(?P<phrase>[^.;:()]{3,120})",
    re.I | re.M,
)
_NAMED = re.compile(
    r"\b(?:called|named|dubbed|termed|coined)\s+(?P<name>[A-Z][\w\-\+\.]*[\w+])"
)
_PHRASE_END = re.compile(
    r"\s*(?:,|\bthat\b|\bwhich\b|\bto\b|\bfor\b|\bin\b|\bby\b|\bwith\b|\bbased\s+on\b|"
    r"\bwhere\b|\busing\b|\bon\b|\bacross\b|\bover\b|\bfrom\b|\bwhose\b|\bthrough\b|\bvia\b|"
    r"\bcalled\b|\bnamed\b|\bdubbed\b|\bthe\b|\bto\b|\band\b)",
    re.I,
)
_LEADING_NAME = re.compile(r"^(?P<name>[A-Z][\w\-\+\.]*[A-Z0-9][\w\-\+]*|[A-Z][a-z]+[A-Z][\w\-]*)\b")
_ACRONYM_DEF = re.compile(
    r"(?P<long>(?:[A-Za-z][\w\-]*\s+){1,6}[A-Za-z][\w\-]*)\s*\((?P<acr>[A-Z][A-Za-z0-9\-]{1,10})\)"
)
_TITLE_NAME = re.compile(r"^\s*(?P<name>[^:]{2,40}?)\s*:\s+\S")
_GENERIC_HEADS = {
    "method", "methods", "approach", "approaches", "framework", "frameworks",
    "model", "models", "system", "systems", "technique", "techniques",
    "algorithm", "algorithms", "architecture", "pipeline", "strategy",
    "solution", "scheme", "paradigm", "mechanism", "way", "study", "analysis",
    "work", "paper", "tool", "library", "set", "family", "series", "variety",
    "number", "survey", "review", "overview",
}
_NON_NAME_HEADS = {
    "goals", "guidelines", "insights", "results", "experiments", "findings",
    "lessons", "routines", "versions", "observations", "experiences",
    "implementations", "ideas", "principles", "challenges", "directions",
    "questions", "problems", "issues", "steps", "ways", "methods", "approaches",
}
_NON_NAME_STARTS = {
    "set", "number", "multiple", "several", "series", "family", "variety",
    "range", "suite", "collection", "list", "some", "various", "many",
    "detailed", "thorough", "brief", "extensive",
}
_DATASET_HEADS = re.compile(r"\b(?:benchmark|dataset|corpus|data\s+set|test\s+suite|leaderboard)s?\b", re.I)
_DATASET_PATTERN = re.compile(
    r"\b(?P<name>[A-Z][\w\-\.]*[A-Za-z0-9](?:[\s\-][A-Z0-9][\w\-\.]*){0,3})\s+"
    r"(?:dataset|benchmark|corpus|data\s+set)s?\b"
)
_NOT_NAMES = {
    "The", "This", "Our", "A", "An", "We", "These", "Large", "New", "Public",
    "Real", "Several", "Many", "Two", "Three", "Multiple", "Existing", "Standard",
    "Each", "Both", "In", "On", "For", "Benchmark", "Extensive", "Popular",
}
_SENT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Extraction:
    arxiv_id: str
    introduces: list[str] = field(default_factory=list)       # methods proposed
    introduces_datasets: list[str] = field(default_factory=list)
    uses: list[str] = field(default_factory=list)             # methods used
    extends: list[str] = field(default_factory=list)          # methods modified
    datasets: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    improves: list[str] = field(default_factory=list)         # metrics improved
    tasks: list[str] = field(default_factory=list)
    aliases: dict[str, str] = field(default_factory=dict)     # ACR -> long form

    def concepts(self) -> list[str]:
        """Every entity name, most specific first (used by Read Next)."""
        return list(dict.fromkeys(
            self.introduces + self.tasks + self.extends + self.uses + self.datasets
        ))


def norm_key(name: str) -> str:
    key = re.sub(r"[^a-z0-9+#@]+", " ", name.lower()).strip()
    if key.endswith("s") and len(key) > 4 and not key.endswith("ss"):
        key = key[:-1]
    return key


def _plausible_name(item: str) -> bool:
    """Reject LaTeX fragments ("R$^3$") and lone adjectives/verbs
    ("Reinforced", "Improving") that pattern rules can pick up."""
    if re.search(r"[$\^{}]", item):
        return False
    if " " not in item and re.fullmatch(r"[A-Z][a-z]+(?:ed|ing|ive|al|ly|ous|able)", item):
        return False
    return True


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for item in items:
        item = re.sub(r"\s+", " ", item).strip(" ,.;:-")
        k = norm_key(item)
        if item and k and k not in seen and _plausible_name(item):
            seen.add(k)
            out.append(item)
    return out


_ALIAS_INDEX: dict[str, tuple[str, str]] = {}
for _kind, _table in _ENTITY_TABLES.items():
    for _display, _aliases in _table.items():
        for _alias in [_display] + _aliases:
            _ALIAS_INDEX.setdefault(norm_key(_alias), (_kind, _display))


def canonical(name: str) -> tuple[str | None, str]:
    """Map a surface name to (kind, canonical display) via the gazetteers.
    "RAG" -> ("method", "Retrieval-Augmented Generation")."""
    base = re.sub(r"\s*\(.*?\)\s*", " ", name).strip()
    for cand in (name, base):
        hit = _ALIAS_INDEX.get(norm_key(cand))
        if hit:
            return hit
    return None, name


def _gazetteer_hits(table: dict[str, list[str]], text: str) -> list[str]:
    return [name for name, aliases in table.items() if any(mentions(a, text) for a in aliases)]


def _clean_phrase(phrase: str) -> tuple[str | None, bool]:
    """Trim a 'we propose ...' phrase to its head. Returns (name, is_dataset)."""
    named = _NAMED.search(phrase)
    if named:
        return named.group("name"), bool(_DATASET_HEADS.search(phrase))
    lead = _LEADING_NAME.match(phrase)
    if lead and len(lead.group("name")) >= 2:
        return lead.group("name"), bool(_DATASET_HEADS.search(phrase[:80]))
    cut = _PHRASE_END.search(phrase)
    head = phrase[: cut.start()] if cut and cut.start() > 0 else phrase
    words = head.split()[:6]
    # "algorithms behind FAISS" -> FAISS: a generic head followed by a name
    if words and words[0].lower() in _GENERIC_HEADS:
        named_tok = next((w for w in words[1:] if re.fullmatch(r"[A-Z][\w\-]*[A-Z0-9][\w\-]*", w)), None)
        if named_tok:
            return named_tok, False
    # drop a trailing participle clause: "dataset constructed (from ...)"
    for i, w in enumerate(words):
        if i >= 2 and re.fullmatch(r"[a-z]+ed", w) and w not in {"based", "learned", "supervised"}:
            words = words[:i]
            break
    while words and words[-1].lower() in {"a", "an", "the", "of", "and", "is", "are"}:
        words.pop()
    if not words:
        return None, False
    if len(words) == 1 and words[0].islower():
        return None, False                     # "hybrid", "efficient": not a name
    if words[-1].lower().strip(",") in _NON_NAME_HEADS or words[0].lower() in _NON_NAME_STARTS:
        return None, False                     # "design goals", "set of guidelines"
    is_dataset = bool(_DATASET_HEADS.search(" ".join(words)))
    content = [w for w in words if w.lower().strip(",") not in _GENERIC_HEADS]
    if not content or (len(content) == 1 and len(content[0]) < 4):
        return None, False
    return " ".join(words), is_dataset


def heuristic_extract(paper: dict) -> Extraction:
    title = paper.get("title", "") or ""
    abstract = paper.get("abstract", "") or ""
    text = f"{title}. {abstract}"
    ex = Extraction(arxiv_id=paper["arxiv_id"])

    # acronym definitions: "Dynamic Alpha Tuning (DAT)"
    for m in _ACRONYM_DEF.finditer(text):
        long = m.group("long").strip()
        acr = m.group("acr")
        # keep only the trailing words that the acronym's letters cover
        words = long.split()
        initials = [c for c in acr if c.isupper()]
        if len(initials) >= 2 and len(words) >= len(initials):
            long = " ".join(words[-len(initials):])
        ex.aliases[acr] = long

    # what this paper introduces
    introduced: list[str] = []
    introduced_ds: list[str] = []
    tm = _TITLE_NAME.match(title)
    if tm:
        name = re.sub(r"^(?:The|A|An)\s+", "", tm.group("name").strip())
        looks_named = (
            len(name.split()) <= 3
            and (re.search(r"[A-Z].*[A-Z]|\d", name[1:]) or name.isupper())
        )
        if looks_named:
            (introduced_ds if _DATASET_HEADS.search(title) else introduced).append(name)
    for m in _PROPOSE.finditer(abstract):
        name, is_ds = _clean_phrase(m.group("phrase"))
        if name:
            (introduced_ds if is_ds else introduced).append(name)
        if len(introduced) + len(introduced_ds) >= 3:
            break
    # an acronym defined inside a "we propose" sentence names the method
    for acr, long in ex.aliases.items():
        if any(norm_key(long) in norm_key(i) or norm_key(i) in norm_key(long) for i in introduced):
            introduced = [i if norm_key(i) != norm_key(long) else f"{long} ({acr})" for i in introduced]
    # canonicalise through the gazetteers so later papers that mention the
    # same thing ("RAG") link to the node this paper introduced
    introduced = [canonical(n)[1] for n in introduced]
    introduced_ds = [canonical(n)[1] for n in introduced_ds]
    ex.introduces = _dedupe(introduced)[:3]
    ex.introduces_datasets = _dedupe(introduced_ds)[:2]

    # known entities
    intro_keys = {norm_key(i) for i in ex.introduces}
    ex.uses = [m for m in _gazetteer_hits(_METHODS, text) if norm_key(m) not in intro_keys]
    ex.datasets = _dedupe(
        _gazetteer_hits(_DATASETS, text)
        + [
            m.group("name") for m in _DATASET_PATTERN.finditer(abstract)
            if m.group("name").split()[0] not in _NOT_NAMES
        ]
    )
    ex.datasets = [
        d for d in ex.datasets
        if norm_key(d) not in {norm_key(x) for x in ex.introduces_datasets}
        # "QA benchmark", "RAG benchmark": a task/method name, not a dataset
        and canonical(d)[0] in (None, "dataset") and len(d) > 2
    ]
    ex.tasks = _gazetteer_hits(_TASKS, text)
    if "Open-Domain QA" in ex.tasks or "Multi-hop QA" in ex.tasks or "Visual Question Answering" in ex.tasks:
        ex.tasks = [t for t in ex.tasks if t != "Question Answering"]
    ex.metrics = _gazetteer_hits(_METRICS, text)

    # claims, sentence by sentence
    for sent in _SENT.split(abstract):
        if _IMPROVE_VERB.search(sent):
            ex.improves += [m for m in ex.metrics if any(mentions(a, sent) for a in _METRICS[m])]
        if _EXTEND_VERB.search(sent):
            ex.extends += [m for m in ex.uses if any(mentions(a, sent) for a in _METHODS[m])]
    ex.improves = _dedupe(ex.improves)
    ex.extends = _dedupe(ex.extends)
    ex.uses = [u for u in ex.uses if u not in ex.extends]
    ex.extends = [u for u in ex.extends if norm_key(u) not in intro_keys]
    return ex


# --------------------------------------------------------------------------
# LLM extraction (optional)
# --------------------------------------------------------------------------
_LLM_PROMPT = """Extract research entities from this paper. Use short names (1-5 words).

Title: {title}
Abstract: {abstract}

Return ONLY JSON with these keys (empty list if none):
{{"introduces": [methods/models this paper proposes],
 "uses": [existing methods it uses or compares with],
 "extends": [existing methods it modifies or builds on],
 "datasets": [datasets/benchmarks it evaluates on],
 "metrics": [metrics it reports],
 "improves": [metrics it claims to improve],
 "tasks": [research tasks/problems it addresses]}}"""


def llm_extract(paper: dict) -> Extraction | None:
    from backend.app.services.llm.generator import get_llm

    llm = get_llm()
    if not llm.generative:
        return None
    try:
        raw = llm.generate(_LLM_PROMPT.format(
            title=paper.get("title", ""), abstract=(paper.get("abstract", "") or "")[:2000]
        ))
    except Exception as exc:
        logger.info("LLM extraction failed for %s: %s", paper.get("arxiv_id"), exc)
        return None
    match = re.search(r"\{.*\}", raw or "", re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return None

    def items(key: str) -> list[str]:
        vals = data.get(key, [])
        return _dedupe([v for v in vals if isinstance(v, str) and 1 < len(v) <= 60])[:6] if isinstance(vals, list) else []

    return Extraction(
        arxiv_id=paper["arxiv_id"],
        introduces=items("introduces")[:3], uses=items("uses"), extends=items("extends"),
        datasets=items("datasets"), metrics=items("metrics"), improves=items("improves"),
        tasks=items("tasks"),
    )


def _merge(a: Extraction, b: Extraction | None) -> Extraction:
    if b is None:
        return a
    out = Extraction(arxiv_id=a.arxiv_id, aliases=dict(a.aliases))
    for f in ("introduces", "introduces_datasets", "uses", "extends", "datasets", "metrics", "improves", "tasks"):
        setattr(out, f, _dedupe(getattr(b, f) + getattr(a, f)))
    out.introduces = out.introduces[:4]
    return out


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------
def extract_papers(papers: list[dict], use_llm: bool = False) -> dict[str, Extraction]:
    """Extract (cached) entities for each paper dict."""
    mode = LLM_VERSION if (use_llm or settings.graph_extractor == "llm") else HEURISTIC_VERSION
    ids = [p["arxiv_id"] for p in papers]
    out: dict[str, Extraction] = {}
    try:  # the cache is an optimisation: never let it break extraction
        with db_session() as s:
            rows = s.execute(
                select(PaperEntities).where(
                    PaperEntities.arxiv_id.in_(ids), PaperEntities.extractor == mode
                )
            ).scalars().all()
            for r in rows:
                try:
                    out[r.arxiv_id] = Extraction(**json.loads(r.data))
                except Exception:
                    pass
    except Exception:
        logger.debug("entity cache read failed", exc_info=True)
    missing = [p for p in papers if p["arxiv_id"] not in out]
    fresh: dict[str, Extraction] = {}
    for p in missing:
        ex = heuristic_extract(p)
        if mode == LLM_VERSION:
            ex = _merge(ex, llm_extract(p))
        fresh[p["arxiv_id"]] = ex
    if fresh:
        try:
            with db_session() as s:
                for pid, ex in fresh.items():
                    s.add(PaperEntities(arxiv_id=pid, extractor=mode, data=json.dumps(asdict(ex))))
        except Exception:
            logger.debug("entity cache write failed", exc_info=True)
    out.update(fresh)
    return out


def load_papers(ids: list[str]) -> list[dict]:
    with db_session() as s:
        rows = s.execute(select(Paper).where(Paper.arxiv_id.in_(ids))).scalars().all()
        by_id = {p.arxiv_id: p.to_dict() for p in rows}
    return [by_id[i] for i in ids if i in by_id]


# --------------------------------------------------------------------------
# graph
# --------------------------------------------------------------------------
def _short(title: str, n: int = 48) -> str:
    return title if len(title) <= n else title[: n - 1].rstrip() + "…"


def build_graph(papers: list[dict], use_llm: bool = False) -> dict:
    """Build nodes/edges/insights for a set of papers."""
    papers = papers[: settings.graph_max_papers]
    ext = extract_papers(papers, use_llm=use_llm)
    nodes: dict[str, dict] = {}
    edges: dict[tuple[str, str, str], dict] = {}
    alias_to_key: dict[str, str] = {}
    for ex in ext.values():
        for acr, long in ex.aliases.items():
            alias_to_key[norm_key(acr)] = norm_key(long)

    def entity(kind: str, name: str, paper_id: str, introduced_by: str | None = None) -> str:
        key = norm_key(name)
        key = alias_to_key.get(key, key)
        nid = f"{kind}:{key}"
        node = nodes.setdefault(nid, {
            "id": nid, "label": name, "type": kind, "papers": [], "introduced_by": None,
        })
        if paper_id not in node["papers"]:
            node["papers"].append(paper_id)
        if introduced_by and not node["introduced_by"]:
            node["introduced_by"] = introduced_by
            node["label"] = name
        return nid

    def edge(src: str, dst: str, rel: str, paper_id: str) -> None:
        if src == dst:
            return
        e = edges.setdefault((src, dst, rel), {"source": src, "target": dst, "relation": rel, "papers": []})
        if paper_id not in e["papers"]:
            e["papers"].append(paper_id)

    by_id = {p["arxiv_id"]: p for p in papers}
    for p in papers:
        pid = p["arxiv_id"]
        ex = ext[pid]
        pnode = f"paper:{pid}"
        nodes[pnode] = {
            "id": pnode, "label": _short(p["title"]), "title": p["title"], "type": "paper",
            "papers": [pid], "published": p.get("published", ""),
            "url": f"https://arxiv.org/abs/{pid}",
        }
        anchors = []
        for m in ex.introduces:
            mid = entity("method", m, pid, introduced_by=pid)
            edge(pnode, mid, "introduces", pid)
            anchors.append(mid)
        for d in ex.introduces_datasets:
            did = entity("dataset", d, pid, introduced_by=pid)
            edge(pnode, did, "introduces", pid)
        anchor = anchors[0] if anchors else pnode
        intro_keys = {norm_key(m) for m in ex.introduces}
        for m in ex.uses:
            if norm_key(m) not in intro_keys:
                edge(pnode, entity("method", m, pid), "uses", pid)
        for m in ex.extends:
            if norm_key(m) not in intro_keys:
                edge(anchor, entity("method", m, pid), "extends", pid)
        for d in ex.datasets:
            edge(anchor, entity("dataset", d, pid), "evaluated on", pid)
        for t in ex.tasks:
            edge(pnode, entity("task", t, pid), "addresses", pid)
        improved = {norm_key(x) for x in ex.improves}
        for m in ex.metrics:
            rel = "improves" if norm_key(m) in improved else "measured by"
            edge(anchor, entity("metric", m, pid), rel, pid)

    # cross-paper links: paper B mentions a method that paper A introduced
    for node in list(nodes.values()):
        if node["type"] != "method" or not node["introduced_by"]:
            continue
        origin = node["introduced_by"]
        name = re.sub(r"\s*\(.*\)\s*$", "", node["label"])
        names = {name}
        acr = re.search(r"\(([^)]+)\)", node["label"])
        if acr:
            names.add(acr.group(1))
        names = {n for n in names if len(n) >= 3}
        for pid, p in by_id.items():
            if pid == origin:
                continue
            text = f"{p['title']}. {p.get('abstract', '')}"
            hit_sents = [s for s in _SENT.split(text) if any(mentions(n, s) for n in names)]
            if not hit_sents:
                continue
            rel = "extends" if any(_EXTEND_VERB.search(s) for s in hit_sents) else "uses"
            own = {f"paper:{pid}"} | {
                n["id"] for n in nodes.values() if n.get("introduced_by") == pid
            }
            if not any(src in own and dst == node["id"] for src, dst, _ in edges):
                edge(f"paper:{pid}", node["id"], rel, pid)
            if pid not in node["papers"]:
                node["papers"].append(pid)

    # one relation per pair: "extends" subsumes "uses"
    for (src, dst, rel) in list(edges):
        if rel == "uses" and (src, dst, "extends") in edges:
            del edges[(src, dst, rel)]

    # prune entity nodes that connect nothing interesting: keep entities
    # shared by 2+ papers, introduced ones, and (for small graphs) all
    small = len(papers) <= 6
    keep = {
        nid for nid, n in nodes.items()
        if n["type"] == "paper" or n["introduced_by"] or len(n["papers"]) >= 2 or small
    }
    node_list = [n for nid, n in nodes.items() if nid in keep]
    edge_list = [e for e in edges.values() if e["source"] in keep and e["target"] in keep]
    for n in node_list:
        n["degree"] = sum(1 for e in edge_list if n["id"] in (e["source"], e["target"]))
    node_list = [n for n in node_list if n["type"] == "paper" or n["degree"] > 0]

    return {
        "nodes": node_list,
        "edges": edge_list,
        "insights": _insights(node_list, edge_list, by_id),
        "extractions": {pid: asdict(ex) for pid, ex in ext.items()},
        "extractor": LLM_VERSION if (use_llm or settings.graph_extractor == "llm") else HEURISTIC_VERSION,
    }


def _insights(nodes: list[dict], edges: list[dict], by_id: dict[str, dict]) -> list[str]:
    out: list[str] = []
    title = {pid: _short(p["title"], 60) for pid, p in by_id.items()}
    by_type: dict[str, list[dict]] = defaultdict(list)
    for n in nodes:
        by_type[n["type"]].append(n)
    for kind, label in (("method", "method"), ("dataset", "benchmark"), ("task", "problem")):
        shared = sorted((n for n in by_type[kind] if len(n["papers"]) >= 2),
                        key=lambda n: len(n["papers"]), reverse=True)
        if shared:
            top = shared[0]
            out.append(f"Most shared {label}: **{top['label']}** appears in {len(top['papers'])} papers.")
    for n in by_type["method"]:
        if n["introduced_by"] and len(n["papers"]) >= 2:
            users = [pid for pid in n["papers"] if pid != n["introduced_by"]]
            out.append(
                f"**{n['label']}** (introduced by “{title.get(n['introduced_by'], n['introduced_by'])}”) "
                f"is reused by {len(users)} other paper(s) in this set."
            )
    # gaps: introduced methods never evaluated on a commonly used benchmark
    common_ds = [n for n in by_type["dataset"] if len(n["papers"]) >= 2]
    eval_pairs = {(e["source"], e["target"]) for e in edges if e["relation"] == "evaluated on"}
    gaps = []
    for m in by_type["method"]:
        if not m["introduced_by"]:
            continue
        for d in common_ds:
            if (m["id"], d["id"]) not in eval_pairs and m["introduced_by"] not in d["papers"]:
                gaps.append(f"Open question: **{m['label']}** has not been evaluated on **{d['label']}** "
                            f"(used by {len(d['papers'])} papers here).")
    out += gaps[:3]
    isolated = [pid for pid in by_id if not any(
        pid in n["papers"] for n in nodes if n["type"] != "paper" and len(n["papers"]) >= 2)]
    if isolated and len(by_id) > 3:
        out.append(f"{len(isolated)} paper(s) share no method, dataset or task with the others: "
                   "they may belong to a different sub-field.")
    counts = Counter(e["relation"] for e in edges)
    if counts:
        out.append("Relations found: " + ", ".join(f"{k} ×{v}" for k, v in counts.most_common()))
    return out[:10]
