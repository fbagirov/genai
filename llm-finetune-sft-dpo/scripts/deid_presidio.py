
import argparse, json, sys

def try_import_presidio():
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        return AnalyzerEngine(), AnonymizerEngine()
    except Exception as e:
        print("Presidio not installed. Please `pip install -r requirements.extra.txt` and download spaCy model.")
        sys.exit(1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input JSONL path")
    ap.add_argument("--out", dest="out", required=True, help="Output JSONL path")
    args = ap.parse_args()

    analyzer, anonymizer = try_import_presidio()

    with open(args.inp, "r", encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            ex = json.loads(line)
            for key in ["instruction","input","output","prompt","chosen","rejected"]:
                if key in ex and isinstance(ex[key], str) and ex[key]:
                    results = analyzer.analyze(text=ex[key], language="en")
                    anonymized_text = anonymizer.anonymize(text=ex[key], analyzer_results=results).text
                    ex[key] = anonymized_text
            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"De-identified data written to {args.out}")

if __name__ == "__main__":
    main()
