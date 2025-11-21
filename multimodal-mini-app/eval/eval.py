
import os, io, json, time, argparse
from PIL import Image
from models.vision import VisionModel
from ocr.ocr import run_ocr

def load_image(path):
    return Image.open(path).convert("RGB")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--provider", default="openai")
    ap.add_argument("--task", choices=["caption","ocr","vqa"], default="caption")
    args = ap.parse_args()

    vm = VisionModel(provider=args.provider)

    total = 0
    correct = 0
    latencies = []

    with open(args.dataset, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            if args.task != item.get("type"): continue
            path = item["file"]
            if not os.path.exists(path):
                print(f"[skip] missing {path}")
                continue
            img = load_image(path)
            t0 = time.time()
            if args.task == "caption":
                out = vm.caption(img)
            elif args.task == "ocr":
                out = run_ocr(img)
            else:
                q = item.get("question","What is in this image?")
                out = vm.vqa(img, q)
            dt = time.time() - t0
            total += 1
            latencies.append(dt)
            expected_exact = item.get("expected_exact")
            expected_contains = item.get("expected_contains", [])
            ok = False
            if expected_exact:
                ok = (out.strip() == expected_exact.strip())
            else:
                out_low = out.lower()
                ok = all(token.lower() in out_low for token in expected_contains)
            print(f"[{'OK' if ok else '..'}] {path}  ({dt:.2f}s)  ->  {out[:120]}")
            if ok:
                correct += 1

    if total == 0:
        print("No matching items evaluated.")
        return
    acc = correct/total*100
    mean_lat = sum(latencies)/len(latencies)
    print(f"Accuracy: {correct}/{total} = {acc:.1f}%")
    print(f"Mean latency: {mean_lat:.2f}s per image")

if __name__ == "__main__":
    main()
