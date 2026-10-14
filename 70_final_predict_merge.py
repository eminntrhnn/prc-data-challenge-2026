#!/usr/bin/env python
"""Şubat/Haziran tahmini + Ocak/Temmuz (v2) ile ID bazlı birleştirme -> final_candidate.parquet
Tahmin kodu YENİDEN YAZILMAZ: 40_final_te.py predict kullanılır (cats.json, te_maps.json, Booster(model_file=...))."""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd

W = Path("/content/drive/MyDrive/prc2026_work")
ID, Y = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
RUNS = [("lgb", "run_v9_lgb"), ("lgb", "run_v9_lgbw"), ("xgb", "run_v9_xgb")]   # v9 = eşit ağırlık
N_EXPECT = 670_790

def die(msg): sys.exit(f"[DUR] {msg}")

def load_ids(p, cols=None):
    d = pd.read_parquet(p, columns=cols); d[ID] = d[ID].astype("float64"); return d

def check_features(df):
    for _, rn in RUNS:
        rd = W / rn; meta = json.loads((rd / "meta.json").read_text())
        maps = json.loads((rd / "te_maps.json").read_text()) if (rd / "te_maps.json").exists() else []
        need = set(meta["base_feats"]) | {meta["time_col"]} | {t for m in maps for t in m["tokens"]}
        miss = sorted(c for c in need if c not in df.columns)
        if miss: die(f"{rn}: test dosyasında eksik kolon (predict sessizce NaN yazardı): {miss}")

def predict_v9(test_path, tag):
    """3 modeli 40_final_te.py ile çalıştırır, eşit ortalama döner. Eski pred.parquet dosyalarına DOKUNMAZ."""
    cols = []
    for algo, rn in RUNS:
        out = W / "run_final" / f"{tag}_{rn}.parquet"; out.parent.mkdir(exist_ok=True)
        if not out.exists():
            r = subprocess.run([sys.executable, str(W / "40_final_te.py"), "predict", "--algo", algo,
                                "--run-dir", str(W / rn), "--test", str(test_path), "--out", str(out)])
            if r.returncode: die(f"predict hata verdi: {rn}")
        p = load_ids(out); cols.append(p.rename(columns={"pred": rn}).set_index(ID)[rn])
    P = pd.concat(cols, axis=1)
    if P.isna().any().any(): die(f"{tag}: tahminde NaN var")
    return P.mean(axis=1)                       # index = ID

def last1_for(test_path, last1_dir, n):
    f = Path(last1_dir) / "test_lgb_full_fold2025-12.parquet"
    if not f.exists(): return None
    t = pd.read_parquet(f)[["__trow__", "pred"]].sort_values("__trow__")
    if len(t) != n or not np.array_equal(t["__trow__"].to_numpy(), np.arange(n)):
        die(f"last1 satır düzeni test dosyasıyla uyuşmuyor ({len(t)} vs {n})")
    ids = load_ids(test_path, [ID])[ID].to_numpy()
    return pd.Series(t["pred"].to_numpy(), index=ids)

def merge_final(template_ids, partial, fb_pred, month_of):
    out = pd.DataFrame({ID: template_ids}).merge(partial[[ID, Y]], on=ID, how="left")
    fb_mask = out[ID].isin(fb_pred.index)
    out.loc[fb_mask, Y] = out.loc[fb_mask, ID].map(fb_pred).to_numpy()      # Şubat/Haziran: ID ile yaz
    if out[Y].isna().any(): die(f"birleştirme sonrası NaN: {int(out[Y].isna().sum())}")
    out[Y] = np.clip(out[Y].astype("float64"), 0, 3600)
    return out, fb_mask

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-v8r", default=str(W / "data/final_v8r.parquet"))      # SADECE DEP, 670.790 satır
    ap.add_argument("--partial", default=str(W / "submissions/final_partial_JanJul.parquet"))
    ap.add_argument("--template", default=str(W / "data/final_submitting.parquet"))
    ap.add_argument("--last1-dir", default=str(W / "run_stand_final"))
    ap.add_argument("--allow-v9-only", action="store_true")
    ap.add_argument("--skip-parity", action="store_true")
    ap.add_argument("--out", default=str(W / "submissions/final_candidate.parquet"))
    a = ap.parse_args()

    tpl = load_ids(a.template, [ID]); X = pd.read_parquet(a.final_v8r); X[ID] = X[ID].astype("float64")
    if len(X) != N_EXPECT or not X[ID].is_unique: die(f"final_v8r: {len(X)} satır / ID tekil={X[ID].is_unique} (beklenen {N_EXPECT}, tekil)")
    if set(X[ID]) != set(tpl[ID]): die("final_v8r ID kümesi şablonla aynı değil")
    if "PHASE_mvt" in X and (X["PHASE_mvt"] != "DEP").any(): die("final_v8r'de DEP dışı satır var")
    check_features(X)
    partial = load_ids(a.partial)                                    # Ocak/Temmuz dolu, Şubat/Haziran NaN
    known = set(partial.loc[partial[Y].notna(), ID])
    fb = ~X[ID].isin(known); jj = ~fb                                # bölme AY'a değil, partial'daki doluluğa göre
    if int(jj.sum()) != len(known): die("partial'daki dolu ID'lerin hepsi final_v8r'de yok")
    print("Şubat/Haziran (tahmin edilecek):", int(fb.sum()), "| Ocak/Temmuz (partial'dan):", int(jj.sum()))
    print("aylık satır (bilgi):", X["month"].value_counts().sort_index().to_dict())

    fb_path, jj_path = W / "data/final_v8r_FebJun.parquet", W / "data/final_v8r_JanJul.parquet"
    X[fb].reset_index(drop=True).to_parquet(fb_path, index=False)
    if not a.skip_parity:                                   # parite: eski pipeline = yeni pipeline (Ocak/Temmuz)
        X[jj].reset_index(drop=True).to_parquet(jj_path, index=False)
        new = predict_v9(jj_path, "parity_JanJul")
        old = load_ids(W / "run_final/blend_v9_lgbxgb.parquet").set_index(ID)["pred"]
        j = pd.concat([new.rename("n"), old.rename("o")], axis=1, join="inner")
        corr = float(j["n"].corr(j["o"])); mad = float((j["n"] - j["o"]).abs().mean())
        print(f"PARİTE (Ocak/Temmuz, v9): n={len(j)} corr={corr:.5f} ort.mutlak fark={mad:.2f} sn")
        if len(j) != int(jj.sum()) or corr < 0.999 or mad > 5: die("parite geçmedi; Şubat/Haziran özellikleri güvenilmez")

    v9 = predict_v9(fb_path, "FebJun")
    l1 = last1_for(fb_path, a.last1_dir, int(fb.sum()))
    if l1 is None:
        if not a.allow_v9_only: die(f"last1 tahmini yok ({a.last1_dir}). Önce last1 fold'unu üret ya da --allow-v9-only ver.")
        print("[UYARI] Şubat/Haziran SADECE v9; Ocak/Temmuz 50/50 harman. Tutarsızlık kabul edildi.")
        fbp = v9
    else:
        fbp = 0.5 * v9 + 0.5 * l1.reindex(v9.index)       # v2 ile aynı: 0.5*v9 + 0.5*last1
        print(f"last1 uygulandı; v9-last1 RMS={float(np.sqrt(((v9-l1.reindex(v9.index))**2).mean())):.1f} sn")

    out, fb_mask = merge_final(tpl[ID].to_numpy(), partial, fbp, None)
    if len(out) != N_EXPECT or not out[ID].is_unique or set(out[ID]) != set(tpl[ID]): die("son kontrol: satır/ID uyuşmuyor")
    m = out.merge(X[[ID, "month"]], on=ID)
    print(m.groupby("month")[Y].agg(["count", "mean", "std", "min", "max"]).round(1))
    out[[ID, Y]].to_parquet(a.out, index=False)
    print(f"[ok] {a.out}: {len(out)} satır, NaN=0")

if __name__ == "__main__":
    main()
