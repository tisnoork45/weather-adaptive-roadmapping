"""One-time held-out TEST evaluation of the promoted weather checkpoint.

Runs two independent paths over data/weather_dataset_c3000/test:
  (a) the training evaluation code (ImageFolder + WeatherEvalTransform, batched)
  (b) the deployed WeatherClassifier.infer() on cv2-decoded BGR frames, one by one
Refuses to run if a result file already exists, so the test set is evaluated once.
"""
import json, sys, time
from pathlib import Path
import cv2, numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root, for `import models`
from torch.utils.data import DataLoader
from torchvision import datasets
import models

CKPT, DATA = "weights/efficientnet_weather.pth", Path("data/weather_dataset_c3000")
OUT = Path("outputs/weather_runs/final_test_eval.json")
if OUT.exists():
    sys.exit(f"{OUT} already exists -- the test set has already been evaluated; refusing to re-run.")

# --- (b) setup first: the deployed classifier must load on CUDA with no fallback --------------
wc = models.WeatherClassifier(weights_path=CKPT, device_preference="auto")
if wc._use_fallback or wc.model is None:
    sys.exit("FAIL: WeatherClassifier fell back to the heuristic (see log above).")
device = next(wc.model.parameters()).device
if device.type != "cuda":
    sys.exit(f"FAIL: WeatherClassifier model is on {device}, not CUDA.")
def _no_heuristic(*_a, **_k):
    raise RuntimeError("heuristic fallback was invoked during evaluation")
wc._heuristic_classify = _no_heuristic  # any silent fallback inside infer() now raises
C = wc.classes
print(f"WeatherClassifier: loaded {CKPT} on {device} ({torch.cuda.get_device_name(device)}), fallback=False, "
      f"input_size={wc.input_size}")

# --- (a) training evaluation code, with an independently loaded copy of the weights -----------
ck = torch.load(CKPT, map_location="cpu")
model = models.build_efficientnet_b0(len(C)); model.load_state_dict(ck["model"], strict=True); model.to(device)
pp = ck["preprocessing"]
ds = datasets.ImageFolder(str(DATA / "test"), loader=models.load_weather_rgb,
                          transform=models.WeatherEvalTransform(pp["input_size"], pp["mean"], pp["std"]))
remap = models._build_class_remap(ds, C); ds.target_transform = remap.__getitem__
loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=8)
report_a = models._evaluate_split(model, loader, ds, device, C, "TEST (training evaluation code)")
preds_a, targets = models._predict(model, loader, device)

# --- (b) deployed path, image by image --------------------------------------------------------
preds_b, times = [], []
for path, _ in ds.samples:
    frame = cv2.imread(path)
    t = time.perf_counter(); res = wc.infer(frame); times.append((time.perf_counter() - t) * 1000)
    preds_b.append(C.index(res.label))
preds_b = np.array(preds_b); sources = models._sample_sources(ds)
cm = lambda m: models._confusion_from(preds_b[m], targets[m], len(C))
report_b = models._log_classification_report(cm(np.ones(len(targets), bool)), C, "TEST (WeatherClassifier.infer)")
report_b["by_source"] = {s: models._log_classification_report(cm(sources == s), C, f"TEST WeatherClassifier [{s.upper()}]",
                                                               macro_over_supported=True) for s in ("bdd", "dawn")}
agree = float((preds_a == preds_b).mean())
print(f"prediction agreement (training eval code vs WeatherClassifier.infer): {agree:.4f} "
      f"({int((preds_a != preds_b).sum())} of {len(preds_a)} differ)")
print(f"WeatherClassifier.infer latency on CUDA: median {np.median(times[10:]):.2f} ms/frame")
OUT.write_text(json.dumps({"checkpoint": CKPT, "checkpoint_sha256_prefix": None, "device": str(device),
                           "training_eval_code": report_a, "weather_classifier": report_b,
                           "prediction_agreement": agree, "median_latency_ms": float(np.median(times[10:]))}, indent=2))
print("wrote", OUT)
