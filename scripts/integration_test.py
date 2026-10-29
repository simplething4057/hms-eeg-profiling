"""E8 앙상블 통합 테스트

Usage:
  python -I scripts/integration_test.py              # 로컬 모델만
  python -I scripts/integration_test.py --api http://localhost:8000
"""
import argparse, json, pathlib, sys, time
import numpy as np

ROOT = pathlib.Path(__file__).parent.parent
CLASSES = ["seizure", "lpd", "gpd", "lrda", "grda", "other"]

def _g(s): return f"\033[32m{s}\033[0m"
def _r(s): return f"\033[31m{s}\033[0m"
def _b(s): return f"\033[1m{s}\033[0m"

def chk(name, ok, detail=""):
    print(f"  [{'PASS' if ok else _r('FAIL')}] {name}" + (f" : {detail}" if detail else ""))
    return ok


def test_weights():
    print(_b("\n[1] 가중치 파일"))
    ok = True
    for m in ("E4", "E5"):
        for f in range(5):
            p = ROOT / "models" / m / f"fold{f}.pt"
            ok &= chk(f"{m}/fold{f}.pt", p.exists())
    meta = ROOT / "models" / "P1" / "p1_meta.json"
    if meta.exists():
        d = json.loads(meta.read_text())
        ok &= chk("p1_meta x_cols=67", len(d.get("x_cols", [])) == 67,
                  str(len(d.get("x_cols", []))))
    return ok


def test_p1():
    print(_b("\n[2] P1 LightGBM 추론"))
    try:
        import lightgbm as lgb
    except ImportError:
        print("  SKIP — lightgbm 없음"); return True
    ok = True
    for f in range(5):
        p = ROOT / "models" / "P1" / f"fold{f}.txt"
        if not p.exists(): continue
        t0 = time.perf_counter()
        bst = lgb.Booster(model_file=str(p))
        feat = np.random.randn(1, 67).astype(np.float32)
        pred = bst.predict(feat)
        ms = (time.perf_counter() - t0) * 1000
        ok &= chk(f"P1 fold{f}", pred.shape == (1, 6), f"{ms:.0f}ms shape={pred.shape}")
    return ok


def test_r1b():
    print(_b("\n[3] R1b 플래그 로직"))
    sys.path.insert(0, str(ROOT))
    from api.predict import _r1b_flag  # noqa: PLC0415
    ok = True
    ok &= chk("저확신 flag", _r1b_flag(np.array([.40,.25,.15,.10,.05,.05]))[0] is True)
    ok &= chk("고확신 정상", _r1b_flag(np.array([.80,.10,.05,.02,.02,.01]))[0] is False)
    ok &= chk("경계 0.47",   _r1b_flag(np.array([.47,.20,.15,.10,.05,.03]))[0] is False)
    return ok


def test_api(base):
    print(_b(f"\n[4] API ({base})"))
    try:
        import requests
    except ImportError:
        print("  SKIP — requests 없음"); return True
    ok = True
    r = requests.get(f"{base}/health", timeout=5)
    ok &= chk("/health 200", r.status_code == 200, str(r.json()))
    feat = np.random.randn(67).tolist()
    t0 = time.perf_counter()
    r = requests.post(f"{base}/predict", json={"segment_id": "t001", "qeeg_features": feat}, timeout=30)
    ms = (time.perf_counter() - t0) * 1000
    ok &= chk("/predict 200", r.status_code == 200, f"{ms:.0f}ms")
    if r.status_code == 200:
        body = r.json()
        ok &= chk("응답 구조", "probabilities" in body and "r1b_flag" in body)
        ok &= chk(f"레이턴시 <500ms", ms < 500, f"{ms:.0f}ms")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api", default=None)
    args = p.parse_args()

    print(_b("=" * 50 + "\n  HMS EEG E8 통합 테스트\n" + "=" * 50))
    results = [test_weights(), test_p1(), test_r1b()]
    if args.api:
        results.append(test_api(args.api))

    passed = sum(results)
    total = len(results)
    line = f"\n  결과: {passed}/{total} 통과"
    print(_b("=" * 50))
    print(_g(line) if passed == total else _r(line))
    print(_b("=" * 50))
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
