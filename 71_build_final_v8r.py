#!/usr/bin/env python
"""final_ranking.parquet (Oca/Şub/Haz/Tem 2026, DEP+ARR) -> final_v8r.parquet (sadece DEP).
Eğitimdeki ranking_v8r zincirinin aynısı: qfeat(3) -> v8b(zaman, wake) -> rwyq(2). HAVA YOK (modelde hava özelliği yok).
Eski ranking_v8r ile Ocak/Temmuz paritesini yazdırır."""
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

ID = "MVT_ID_mvt"; NEW = ["dep_back_30", "arr_back_30", "dep_fwd_30"]; RWYQ = ["rwyq_dep_15", "rwyq_dep_30"]
DROP_ID = ["FLIGHT_ID_mvt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]
WIN = 1800
N_EXPECT = 670_790

def die(m): sys.exit(f"[DUR] {m}")

def sec(s):
    t = pd.to_datetime(s, utc=True, errors="coerce")
    return (t - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()

def cnt(arr, lo, hi):
    return np.searchsorted(arr, hi, "right") - np.searchsorted(arr, lo, "right")

def qfeat(d):
    """Eski 'build' hücresiyle aynı mantık. d = TÜM satırlar (DEP+ARR), ID'ye göre tekilleştirilmiş."""
    mvt, blk = sec(d["MVT_TIME_UTC_mvt"]), sec(d["BLOCK_TIME_UTC_mvt"])
    guess = np.where(np.isnan(mvt) | np.isnan(blk), "UNK", np.where(blk > mvt, "ARR", "DEP"))
    ph0 = d["PHASE_mvt"].astype("object").str.upper()
    known = ph0.notna().to_numpy()
    ph = np.where(known, ph0.to_numpy(), guess)
    dep, arr = (ph == "DEP"), (ph == "ARR")
    ap = np.where(dep, d["ADEP_mvt"].astype(str).to_numpy(), d["ADES_mvt"].astype(str).to_numpy())
    ev = np.where(dep, sec(d["AOBT_3_flt"]), np.where(arr, sec(d["ARVT_3_flt"]), np.nan))
    sch = np.where(dep, sec(d["SCHED_TIME_UTC_mvt"]), np.nan)
    n = len(d); db, ab, fw = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for _, idx in pd.Series(ap).groupby(ap).indices.items():
        e = ev[idx]; D = dep[idx]; A = arr[idx]; s = sch[idx]
        dv = np.sort(e[D & np.isfinite(e)]); av = np.sort(e[A & np.isfinite(e)]); sv = np.sort(s[D & np.isfinite(s)])
        ok = np.isfinite(e); q = e[ok]; ii = idx[ok]
        db[ii] = cnt(dv, q - WIN, q) - dep[ii].astype(int)
        ab[ii] = cnt(av, q - WIN, q) - arr[ii].astype(int)
        own = (dep[ii] & (sch[ii] > q) & (sch[ii] <= q + WIN)).astype(int)
        fw[ii] = cnt(sv, q, q + WIN) - own
    out = pd.DataFrame({ID: d[ID].to_numpy(), "dep_back_30": db, "arr_back_30": ab, "dep_fwd_30": fw})
    return out[dep]                                    # sadece DEP satırları

def add_time(df, tcol="MVT_TIME_UTC_mvt"):
    t = pd.to_datetime(df[tcol], utc=True); df = df.copy()
    df["hour"] = t.dt.hour; df["dow"] = t.dt.dayofweek; df["month"] = t.dt.month
    df["is_weekend"] = (t.dt.dayofweek >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * t.dt.hour / 24.0); df["hour_cos"] = np.cos(2 * np.pi * t.dt.hour / 24.0)
    return df

def same(a, b):
    if is_datetime64_any_dtype(a) and is_datetime64_any_dtype(b):
        diff = (pd.to_datetime(a, utc=True) - pd.to_datetime(b, utc=True)).dt.total_seconds().abs()
        return ((diff <= 30) | (a.isna() & b.isna())).mean()
    if is_numeric_dtype(a) and is_numeric_dtype(b):
        x, y = a.astype("float64").to_numpy(), b.astype("float64").to_numpy()
        return float(np.isclose(x, y, rtol=0, atol=1e-6, equal_nan=True).mean())
    return float(((a.astype(str) == b.astype(str)) | (a.isna() & b.isna())).mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/content/drive/MyDrive/prc2026_work")
    ap.add_argument("--final-ranking"); ap.add_argument("--ref-v8r"); ap.add_argument("--train-v8r")
    ap.add_argument("--template"); ap.add_argument("--out"); ap.add_argument("--skip-parity", action="store_true")
    a = ap.parse_args(); W = Path(a.work)
    fr = a.final_ranking or W / "data/final_ranking.parquet"; ref_p = a.ref_v8r or W / "data/ranking_v8r.parquet"
    trn = a.train_v8r or W / "data/train_v8r.parquet"; out_p = a.out or W / "data/final_v8r.parquet"

    R = pd.read_parquet(fr); R[ID] = R[ID].astype("float64")
    nd = int(R[ID].duplicated().sum()); R = R.drop_duplicates(ID).reset_index(drop=True)
    print(f"final_ranking: {len(R)} satır (tekrar eden ID temizlendi: {nd})")
    Q = qfeat(R)
    D = R[R["PHASE_mvt"].astype(str).str.upper() == "DEP"].copy()
    D = D.drop(columns=[c for c in NEW + ["queue_density_30m"] if c in D.columns])
    D = D.merge(Q, on=ID, how="left", validate="one_to_one")
    D = add_time(D)
    tw = pd.read_parquet(trn, columns=["wake_category", "wake_rank"])          # train'deki sabitleri aynen uygula
    if tw["wake_category"].nunique() != 1 or tw["wake_rank"].nunique() != 1: die("train'de wake sabit değil; builder'ı gözden geçir")
    D["wake_category"] = tw["wake_category"].iloc[0]; D["wake_rank"] = tw["wake_rank"].iloc[0]
    D = D.drop(columns=[c for c in DROP_ID if c in D.columns]).reset_index(drop=True)

    spec = importlib.util.spec_from_file_location("rwyq", str(W / "41_runway_queue.py"))
    rw = importlib.util.module_from_spec(spec); spec.loader.exec_module(rw)
    rq = rw.build(D)                                                          # 41_runway_queue.py ile aynı fonksiyon
    for c in RWYQ: D[c] = rq[c].to_numpy()

    ref = pd.read_parquet(ref_p); ref[ID] = ref[ID].astype("float64")
    miss = [c for c in ref.columns if c not in D.columns]
    if miss: die(f"ranking_v8r'de olup üretilemeyen kolonlar: {miss}")
    D = D[list(ref.columns)]
    print("kuyruk NaN oranı:", D[NEW + RWYQ].isna().mean().round(4).to_dict())
    print("aylık satır:", D["month"].value_counts().sort_index().to_dict())
    ok = D[ID].is_unique          # ay uç satırlarda MVT_TIME yüzünden 3/8 çıkabilir; bilgi amaçlı
    if a.template: ok = ok and set(D[ID]) == set(pd.read_parquet(a.template, columns=[ID])[ID].astype("float64"))
    if not a.skip_parity and len(D) != N_EXPECT: ok = False
    if not ok: print("[UYARI] satır sayısı / tekillik / şablon kontrolü başarısız")
    D.to_parquet(out_p, index=False); print(f"[yazıldı] {out_p}: {D.shape}")

    bad = not ok
    if not a.skip_parity:
        j = D.merge(ref, on=ID, suffixes=("", "_o")); print(f"\nPARİTE: ortak ID (Ocak/Temmuz) = {len(j)} / {len(ref)}")
        if len(j) != len(ref): bad = True
        rows = []
        for c in ref.columns:
            if c == ID or c == "MVT_TIME_UTC_mvt": continue
            s = same(j[c], j[c + "_o"]); thr = 0.99 if c in NEW + RWYQ + ["hour", "hour_sin", "hour_cos"] else 0.999
            rows.append((c, round(s, 5), "OK" if s >= thr else "SORUN")); bad |= s < thr
        print(pd.DataFrame(rows, columns=["kolon", "eşleşme", "durum"]).to_string(index=False))
    if bad: sys.exit("[DUR] parite/kontrol geçmedi - final_v8r'ye güvenme")
    print("[ok] final_v8r hazır; sıradaki: 70_final_predict_merge.py")

if __name__ == "__main__":
    main()
