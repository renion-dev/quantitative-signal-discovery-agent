from pathlib import Path
import json, subprocess, threading, os, time, logging, re, pandas as pd
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

logger = logging.getLogger("dashboard")

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "src" / "signal_discovery_workflow" / "output"
app = FastAPI(title="Signal Discovery Dashboard")
state = {"running": False, "last": None, "log": "", "auto": False}

@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "index.html").read_text()

@app.get("/api/status")
def status():
    return {"running": state["running"], "last": state["last"], "log": state["log"][-2000:], "auto": state["auto"]}

@app.get("/api/signals")
def signals():
    items = []
    if OUTPUT.exists():
        for p in sorted(OUTPUT.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                items.append({"file": p.name, "data": json.loads(p.read_text())})
            except Exception as e:
                items.append({"file": p.name, "error": str(e)})
    return JSONResponse(items)

def _run(inp):
    state["running"] = True
    state["log"] = ""
    try:
        env = dict(os.environ)
        env["SIGNAL_DATA_DIR"] = "binance_futures"
        proc = subprocess.run(
            ["uv", "run", "nat", "run", "--config_file", "configs/config-binance-nvidia.yml", "--input", inp],
            cwd=str(ROOT), capture_output=True, text=True, env=env, timeout=1800,
        )
        state["log"] = proc.stdout + "\n" + proc.stderr
        state["last"] = {"code": proc.returncode}
    except Exception as e:
        state["log"] = str(e)
        state["last"] = {"error": str(e)}
    finally:
        state["running"] = False

@app.post("/api/run")
def run(payload: dict):
    if state["running"]:
        return {"error": "already running"}
    inp = payload.get("input", "momentum signals")
    threading.Thread(target=_run, args=(inp,), daemon=True).start()
    return {"started": True}

@app.post("/api/auto/start")
def auto_start(payload: dict):
    if state["auto"]:
        return {"error": "already auto"}
    state["auto"] = True
    inp = payload.get("input", "momentum signals")
    def loop():
        while state["auto"]:
            if not state["running"]:
                _run(inp)
            time.sleep(60)
    threading.Thread(target=loop, daemon=True).start()
    return {"started": True}

@app.post("/api/auto/stop")
def auto_stop():
    state["auto"] = False
    return {"stopped": True}

@app.post("/api/stop")
def stop():
    state["auto"] = False
    return {"stopped": True}

@app.get("/api/card")
def card(symbol: str):
    ddir = ROOT/"src"/"signal_discovery_workflow"/"data"/"binance_futures"
    try:
        close = pd.read_csv(ddir/"Close.csv", index_col=0, parse_dates=True)[symbol]
    except KeyError:
        return JSONResponse({"error": f"{symbol} not in current data universe"}, status_code=404)
    
    high = pd.read_csv(ddir/"High.csv", index_col=0, parse_dates=True)[symbol]
    low = pd.read_csv(ddir/"Low.csv", index_col=0, parse_dates=True)[symbol]
    vol = pd.read_csv(ddir/"Volume.csv", index_col=0, parse_dates=True)[symbol]
    price = float(close.iloc[-1])
    hi20 = float(high.tail(20).max())
    lo20 = float(low.tail(20).min())
    avgvol = float(vol.tail(20).mean())
    sma20 = float(close.tail(20).mean())
    change24 = round((price/close.iloc[-2]-1)*100,2) if len(close)>1 else 0
    trend7 = round((price/close.iloc[-7]-1)*100,2) if len(close)>7 else 0
    conf = round(min(1.0, abs((price-sma20)/sma20)*10),3)
    range_pct = (hi20-lo20)/price if price else 0
    # inverse volatility sizing: calmer symbol -> bigger size, wilder -> smaller
    pos = round(max(0.5, min(3.0, 3.0/(1.0+range_pct*10))),2)
    return JSONResponse({"symbol":symbol,"price":round(price,4),"high20":round(hi20,4),"low20":round(lo20,4),"sma20":round(sma20,4),"avgvol":round(avgvol,2),"conf":conf,"pos":pos,"change24":change24,"trend7":trend7})
@app.get("/api/price")
def price(symbol: str):
    import urllib.request, json as _j
    url=f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}"
    with urllib.request.urlopen(url, timeout=10) as r:
        d=_j.loads(r.read())
    return JSONResponse({"symbol":symbol,"price":float(d["markPrice"])})
def _load_stock_data():
    ddir = ROOT/"src"/"signal_discovery_workflow"/"data"/"binance_futures"
    stock_data={}
    for f in ["Open","High","Low","Close","Volume"]:
        p=ddir/(f+".csv")
        if p.exists():
            stock_data[f]=pd.read_csv(p,index_col=0,parse_dates=True)
    return stock_data

def _cache_key():
    parts=[]
    if OUTPUT.exists():
        parts+=sorted((p.name, p.stat().st_mtime) for p in OUTPUT.glob("*.json"))
    ddir = ROOT/"src"/"signal_discovery_workflow"/"data"/"binance_futures"
    if ddir.exists():
        parts+=sorted((p.name, p.stat().st_mtime) for p in ddir.glob("*.csv"))
    return tuple(parts)

_top_cache={"key": None, "vals": None}   # symbol -> list of last signal values per signal file

def _sandbox_violations(code):
    """Optional static sandbox check for saved signal code.

    Returns a non-empty list of violation findings when the code violates the
    sandbox policy, an empty list when it is clean, or None when the optional
    ``signal_discovery_workflow.sandbox`` module is unavailable (callers then
    keep the legacy behaviour unchanged). Never raises.
    """
    try:
        from signal_discovery_workflow.sandbox import classify_code_violations
    except ImportError:
        return None
    except Exception as e:
        logger.warning("sandbox module present but not importable: %s", e)
        return None
    try:
        res = classify_code_violations(code)
    except Exception as e:
        # Fail closed: if the classifier errors on importable module, skip execution.
        logger.warning("sandbox classifier failed on saved signal code: %s", e)
        return [{"type": "classifier_error", "detail": str(e)[:200]}]
    if isinstance(res, dict):  # defensive: module may evolve to a report dict
        res = res.get("violations") or []
    if res is None:
        return []
    if isinstance(res, (list, tuple, set)):
        return list(res)
    return [res] if res else []

def _signal_values_all():
    """Exec every saved signal, return {signal_name: last_row Series} + universe."""
    key=_cache_key()
    if _top_cache["key"]==key and _top_cache["vals"] is not None:
        return _top_cache["vals"]
    import sys
    sys.path.insert(0,str(ROOT/"src"))
    stock_data=_load_stock_data()
    if not stock_data:
        _top_cache["key"]=key; _top_cache["vals"]={"series":{}, "universe":[]}
        return _top_cache["vals"]
    universe=list(stock_data["Close"].columns)
    series={}
    files = sorted(OUTPUT.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True) if OUTPUT.exists() else []
    max_signals = int(os.environ.get("MAX_SIGNALS_FOR_CONSENSUS", "60"))
    files = files[:max_signals]
    for fp in files:
        try:
            data=json.loads(fp.read_text())
        except Exception:
            continue
        code=None
        for v in data.values():
            if isinstance(v,str) and "def signal_" in v:
                code=v; break
        if not code:
            continue
        viol=_sandbox_violations(code)
        if viol:
            vtypes=sorted({str(v.get("type","?")) if isinstance(v,dict) else str(v) for v in viol})
            logger.warning("Skipping saved signal %s: %d sandbox violation(s): %s",
                           fp.name, len(viol), ", ".join(vtypes))
            continue
        try:
            ns={}
            exec(code, ns)
            funcs=[(n,o) for n,o in ns.items() if callable(o) and n.startswith("signal_")]
            if not funcs:
                continue
            name,fn=funcs[0]
            kwargs={p:stock_data[p] for p in fn.__code__.co_varnames[:fn.__code__.co_argcount] if p in stock_data}
            if len(kwargs)<fn.__code__.co_argcount:
                continue
            sig=fn(**kwargs)
            if isinstance(sig,pd.Series):
                sig=sig.to_frame()
            last=sig.iloc[-1].dropna()
            if len(last):
                series[name]=last
        except Exception as e:
            logger.warning("Failed to evaluate saved signal %s: %s", fp.name, e)
            continue
    vals={"series":series, "universe":universe}
    _top_cache["key"]=key; _top_cache["vals"]=vals
    return vals

@app.get("/api/top")
def top():
    vals=_signal_values_all()
    if not vals["series"]:
        return JSONResponse({"error": "no signals"})
    name, last = next(iter(vals["series"].items()))
    last=last.sort_values(ascending=False)
    n=len(last); k=max(1,n//2)
    return JSONResponse({"signal":name,"longs":last.head(k).index.tolist(),"shorts":last.tail(k).index.tolist()})

@app.get("/api/signal_detail")
def signal_detail(name: str):
    """Long/short split for one specific saved signal."""
    vals=_signal_values_all()
    series=vals["series"]
    if name not in series:
        return JSONResponse({"error": f"signal {name} not evaluated"}, status_code=404)
    last=series[name].sort_values(ascending=False)
    n=len(last); k=max(1,n//2)
    # find metrics for this signal
    metrics=None
    for fp in OUTPUT.glob("*.json"):
        try:
            d=json.loads(fp.read_text())
            if d.get("selected_signal")==name:
                metrics=d.get("evaluation_metrics"); break
        except Exception:
            continue
    return JSONResponse({"signal":name,"longs":last.head(k).index.tolist(),"shorts":last.tail(k).index.tolist(),"metrics":metrics})

def _consensus_data():
    vals=_signal_values_all()
    series=vals["series"]
    if not series:
        return {"error": "no signals", "rows": [], "signals_used": 0}
    universe=vals["universe"]
    votes={s:{"long":0,"short":0,"net":0.0} for s in universe}
    for name,last in series.items():
        ranked=last.sort_values(ascending=False)
        n=len(ranked)
        if n<2:
            continue
        k=max(1,n//2)
        for sym in ranked.head(k).index:
            if sym in votes: votes[sym]["long"]+=1
        for sym in ranked.tail(k).index:
            if sym in votes: votes[sym]["short"]+=1
    for v in votes.values():
        v["net"]=v["long"]-v["short"]
    rows=[{"symbol":s, **v} for s,v in votes.items()]
    rows.sort(key=lambda r:(r["net"], r["long"]), reverse=True)
    top_buy = rows[:3]
    top_sell = sorted(rows, key=lambda r:(r["net"], r["short"]))[:3]
    total_net = sum(r["net"] for r in rows)
    n_pos = sum(1 for r in rows if r["net"] > 0)
    n_neg = sum(1 for r in rows if r["net"] < 0)
    n_tot = len(rows) or 1
    strength = (n_pos - n_neg) / n_tot
    if strength > 0.1:
        state = "BULLISH"
    elif strength < -0.1:
        state = "BEARISH"
    else:
        state = "NEUTRAL"
    summary = {"top_buy": top_buy, "top_sell": top_sell, "market_state": state,
               "total_net": total_net, "strength": round(strength, 3),
               "pos": n_pos, "neg": n_neg}
    return {"rows": rows, "signals_used": len(series), "summary": summary}

@app.get("/api/consensus")
def consensus():
    d=_consensus_data()
    if d.get("error"):
        return JSONResponse({"error":d["error"]})
    return JSONResponse(d)

import json as _json
import urllib.request as _urllib

@app.get("/api/prices")
def prices():
    """Live 24h ticker for the current universe from Binance USDT-M futures."""
    vals=_signal_values_all()
    syms=vals["universe"] or []
    if not syms:
        return JSONResponse({"error":"no universe"})
    out={}
    try:
        url="https://fapi.binance.com/fapi/v1/ticker/24hr"
        with _urllib.urlopen(url, timeout=10) as r:
            data=_json.loads(r.read())
        wanted=set(syms)
        for t in data:
            sym=t.get("symbol")
            if sym in wanted:
                out[sym]={"price":float(t["lastPrice"]),"change24h":float(t["priceChangePercent"]),
                          "volume":float(t["quoteVolume"])}
    except Exception as e:
        return JSONResponse({"error":str(e)})
    return JSONResponse(out)
@app.post("/api/alert")
def alert():
    """Send top-3 buy/sell summary to Telegram if configured."""
    token=os.environ.get("TELEGRAM_BOT_TOKEN"); chat=os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return JSONResponse({"error":"set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env"})
    d=_consensus_data()
    if d.get("error"):
        return JSONResponse({"error":d["error"]})
    s=d["summary"]
    lines=["📊 IRON SIGNAL — MARKET UPDATE", f"State: {s['market_state']} ({s['strength']*100:.1f}%)", "", "🟢 TOP BUY:"]
    for r in s["top_buy"]: lines.append(f"  {r['symbol']}  net {r['net']}")
    lines.append(""); lines.append("🔴 TOP SELL:")
    for r in s["top_sell"]: lines.append(f"  {r['symbol']}  net {r['net']}")
    text="\n".join(lines)
    try:
        url=f"https://api.telegram.org/bot{token}/sendMessage"
        data=_json.dumps({"chat_id":chat,"text":text}).encode()
        req=_urllib.Request(url,data=data,headers={"Content-Type":"application/json"})
        with _urllib.urlopen(req,timeout=10) as r:
            res=_json.loads(r.read())
        return JSONResponse({"ok":res.get("ok",False)})
    except Exception as e:
        return JSONResponse({"error":str(e)})
from contextlib import contextmanager as _contextmanager
import tempfile as _tempfile
try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Linux is the supported runtime
    _fcntl=None

from signal_discovery_workflow.paper_trading import (
    PaperConfigError,
    load_paper_config,
    resolve_paper_config_path,
    select_open_candidates,
)

PAPER_FILE=Path(__file__).parent/"paper.json"
_PAPER_THREAD_LOCK=threading.Lock()
_PAPER_CONFIG_WARNED=set()

def _atr(sym, period=14):
    try:
        sd=_load_stock_data()
        H=sd["High"][sym]; L=sd["Low"][sym]; C=sd["Close"][sym]
        tr=pd.concat([H-L, (H-C.shift()).abs(), (L-C.shift()).abs()],axis=1).max(axis=1)
        atr=tr.rolling(period).mean().iloc[-1]
        return float(atr) if pd.notna(atr) else None
    except Exception:
        return None

def _load_paper():
    if PAPER_FILE.exists():
        try:
            data=json.loads(PAPER_FILE.read_text())
            if isinstance(data,dict):
                if not isinstance(data.get("positions",[]),list):
                    data["positions"]=[]
                return data
        except Exception:
            pass
    return {"positions":[]}

def _save_paper(d):
    """Atomically persist paper state by replacing a same-directory temporary file."""

    PAPER_FILE.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp_name=_tempfile.mkstemp(prefix=f".{PAPER_FILE.name}.",suffix=".tmp",dir=PAPER_FILE.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as f:
            json.dump(d,f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name,PAPER_FILE)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise

@_contextmanager
def _paper_transaction():
    """Serialize paper-state transactions across dashboard threads and processes.

    Locking the stable dashboard-directory inode avoids a separate untracked lock artifact and
    remains valid when ``paper.json`` is atomically replaced.
    """

    with _PAPER_THREAD_LOCK:
        if _fcntl is None:
            yield
            return
        fd=os.open(str(PAPER_FILE.parent),os.O_RDONLY)
        try:
            _fcntl.flock(fd,_fcntl.LOCK_EX)
            yield
        finally:
            _fcntl.flock(fd,_fcntl.LOCK_UN)
            os.close(fd)

def _load_paper_configuration():
    """Load the paper config and emit deduplicated missing/unknown-key warnings."""

    path=resolve_paper_config_path()
    try:
        config=load_paper_config(path)
    except PaperConfigError as exc:
        warning_key=f"invalid:{path}"
        if warning_key not in _PAPER_CONFIG_WARNED:
            _PAPER_CONFIG_WARNED.add(warning_key)
            logger.error("Invalid paper-trading config: %s",exc)
        raise
    _PAPER_CONFIG_WARNED.discard(f"invalid:{path}")
    if config.file_missing:
        warning_key=f"missing:{path}"
        if warning_key not in _PAPER_CONFIG_WARNED:
            _PAPER_CONFIG_WARNED.add(warning_key)
            logger.warning("Paper-trading config not found at %s; using defaults",path)
    else:
        _PAPER_CONFIG_WARNED.discard(f"missing:{path}")
    for message in config.warnings:
        warning_key=f"warning:{path}:{message}"
        if warning_key not in _PAPER_CONFIG_WARNED:
            _PAPER_CONFIG_WARNED.add(warning_key)
            logger.warning("Paper-trading config warning: %s",message)
    return config

def _paper_limits(config):
    """Return the effective configuration exposed by the paper-trading API."""

    return {
        "max_open_positions":config.max_open_positions,
        "starting_balance":config.starting_balance,
        "max_position_size_pct":config.max_position_size_pct,
        "max_position_size_amount":config.max_position_size_amount,
    }

@app.get("/api/paper/status")
def paper_status():
    try:
        config=_load_paper_configuration()
        config_error=None
    except PaperConfigError as exc:
        config=None; config_error=str(exc)
    prices={}
    try:
        url="https://fapi.binance.com/fapi/v1/ticker/24hr"
        with _urllib.urlopen(url,timeout=10) as r:
            for t in _json.loads(r.read()):
                prices[t["symbol"]]=float(t["lastPrice"])
    except Exception: pass
    with _paper_transaction():
        d=_load_paper()
        # mark-to-market
        total=0.0
        remaining=[]
        for p in d["positions"]:
            px=prices.get(p["symbol"],p["entry"])
            atr=p.get("atr") or (abs(p["entry"]-p["sl"])/1.5)
            # price velocity: change over last 3 bars
            vel=0.0
            try:
                sd=_load_stock_data()
                C=sd["Close"][p["symbol"]]
                if len(C)>=4:
                    vel=abs((C.iloc[-1]-C.iloc[-4])/C.iloc[-4])
            except Exception:
                pass
            # tighten trail on fast moves, widen on slow
            mult=1.5 if vel<0.02 else 1.0
            hit=False
            # trailing stop: ratchet SL in favorable direction
            if p["side"]=="long":
                new_sl=px-atr*mult
                if new_sl>p.get("sl",0):
                    p["sl"]=round(new_sl,6)
                if px<=p["sl"] or px>=p["tp"]: hit=True
                pnl=(px-p["entry"])*p["qty"]
            else:
                new_sl=px+atr*mult
                if new_sl<p.get("sl",float('inf')):
                    p["sl"]=round(new_sl,6)
                if px>=p["sl"] or px<=p["tp"]: hit=True
                pnl=(p["entry"]-px)*p["qty"]
            p["mark"]=px; p["pnl"]=round(pnl,4)
            if hit:
                p["exit"]=px; p["closed_reason"]="SL/TP"
                d.setdefault("closed",[]).append(p)
                d["cumulative_closed_pnl"]=round(d.get("cumulative_closed_pnl",0)+pnl,4)
                d["last_pnl"]=round(pnl,4)
            else:
                remaining.append(p); total+=pnl
        d["positions"]=remaining
        d["total_pnl"]=round(total,4)
        d["combined_pnl"]=round(d.get("cumulative_closed_pnl",0)+total,4)
        hist=d.setdefault("history",[])
        hist.append({"ts":int(time.time()),"pnl":d["total_pnl"],"n":len(remaining),
                     "pos":{p["symbol"]:p["pnl"] for p in remaining}})
        if len(hist)>200: del hist[:-200]
        _save_paper(d)
    response=dict(d)
    if config is not None:
        response["limits"]=_paper_limits(config)
    else:
        response["limits"]=None; response["config_error"]=True; response["error"]=config_error
    return JSONResponse(response)

@app.post("/api/paper/open")
def paper_open():
    try:
        config=_load_paper_configuration()
    except PaperConfigError as exc:
        return JSONResponse({"error":str(exc),"config_error":True},status_code=400)
    d=_consensus_data()
    if d.get("error"): return JSONResponse({"error":d["error"]})
    s=d["summary"]
    prices={}
    try:
        url="https://fapi.binance.com/fapi/v1/ticker/24hr"
        with _urllib.urlopen(url,timeout=10) as r:
            for t in _json.loads(r.read()):
                prices[t["symbol"]]=float(t["lastPrice"])
    except Exception as e:
        return JSONResponse({"error":str(e)})
    desired={}
    for r in s["top_buy"]:
        desired[r["symbol"]]="long"
    for r in s["top_sell"]:
        desired[r["symbol"]]="short"
    with _paper_transaction():
        paper=_load_paper()
        if not desired:
            return JSONResponse({
                "opened":0,"closed":0,"positions":paper["positions"],
                "note":"no desired signals, positions kept",
                "limits":_paper_limits(config),"skipped":[],
            })
        selected,skipped=select_open_candidates(
            desired,paper["positions"],prices,config,paper_state=paper,base_qty=1.0,
        )
        # open new only (hold existing until SL/TP)
        opened=[]
        for candidate in selected:
            sym=candidate["symbol"]; side=candidate["side"]; px=candidate["price"]
            qty=candidate["qty"]; atr=_atr(sym)
            sl_dist = (atr*1.5) if atr else px*0.02
            tp_dist = (atr*3.0) if atr else px*0.04
            pos={
                "symbol":sym,"side":side,"entry":px,"qty":qty,
                "sl":round(px-sl_dist,6) if side=="long" else round(px+sl_dist,6),
                "tp":round(px+tp_dist,6) if side=="long" else round(px-tp_dist,6),
                "atr":atr,"notional":qty*px,"sizing_cap":candidate["sizing_cap"],
            }
            paper["positions"].append(pos); opened.append(pos)
        if opened:
            _save_paper(paper)
    return JSONResponse({
        "opened":len(opened),"closed":0,"positions":paper["positions"],
        "limits":_paper_limits(config),"skipped":skipped,
    })

@app.post("/api/paper/close")
def paper_close():
    prices={}
    try:
        url="https://fapi.binance.com/fapi/v1/ticker/24hr"
        with _urllib.urlopen(url,timeout=10) as r:
            for t in _json.loads(r.read()):
                prices[t["symbol"]]=float(t["lastPrice"])
    except Exception as e:
        return JSONResponse({"error":str(e)})
    with _paper_transaction():
        paper=_load_paper()
        total=0.0
        for p in paper["positions"]:
            px=prices.get(p["symbol"],p["entry"])
            pnl=(px-p["entry"])*p["qty"] if p["side"]=="long" else (p["entry"]-px)*p["qty"]
            p["exit"]=px; p["pnl"]=round(pnl,4); total+=pnl
        paper["closed"]=paper["positions"]; paper["positions"]=[]; paper["last_pnl"]=round(total,4)
        paper["cumulative_closed_pnl"]=round(paper.get("cumulative_closed_pnl",0)+total,4)
        _save_paper(paper)
    return JSONResponse({"closed":len(paper["closed"]),"total_pnl":paper["last_pnl"]})
@app.get("/api/paper/history")
def paper_history():
    d=_load_paper()
    return JSONResponse({"history":d.get("history",[]),"last_pnl":d.get("last_pnl",0)})
@app.get("/api/top9")
def top9():
    d=_consensus_data()
    if d.get("error"):
        return JSONResponse({"error":d["error"]})
    rows=d["rows"][:9]
    return JSONResponse({"symbols":[r["symbol"] for r in rows],"rows":rows,"signals_used":d["signals_used"]})

# ---------------------------------------------------------------------------
# Signal evolution / lineage API
#
# Parses saved output/signal_*_iter<n>.json files into runs of iteration
# nodes. Grouping: by `run_id` when present in the file, else by the legacy
# filename prefix before `_iter` (e.g. `signal_20261108_190141_iter1.json`
# -> run "20261108_190141"). Pure JSON parsing — never executes signal code.
# ---------------------------------------------------------------------------

_EVO_FILE_RE = re.compile(r"^signal_(.+)_iter(\d+)\.json$")
_EVO_NON_METRIC_KEYS = {
    "iteration", "timestamp", "run_id", "selected_signal", "signal_code",
    "signal_description", "request", "lineage", "last_feedback", "error",
    "status", "saved_path", "headline", "oos",
}
# Acceptance thresholds mirror the optimizer configs (configs/*.yml).
_IC_THRESHOLD = float(os.environ.get("SIGNAL_IC_THRESHOLD", "0.02"))
_P_THRESHOLD = float(os.environ.get("SIGNAL_P_THRESHOLD", "0.05"))
_EVO_MAX_RUNS = int(os.environ.get("EVOLUTION_MAX_RUNS", "200"))
_EVO_CACHE = {"key": None, "parsed": None}

def _evo_num(v):
    """float(v) for real numbers (bool excluded), else None."""
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

def _evo_ts_iso(ts):
    """'20261108_190141' -> '2026-11-08T19:01:41', else None."""
    try:
        return datetime.strptime(str(ts), "%Y%m%d_%H%M%S").isoformat()
    except Exception:
        return None

def _parse_signal_description(desc) -> list:
    """Extract signal descriptor dicts from a ``signal_description`` value.

    Handles: raw JSON array, a single JSON dict, and ```json fenced blocks
    embedded in LLM prose. Fenced ```python blocks are skipped. A balanced
    bracket scan is used as a last resort. Invalid/missing JSON -> [].
    """
    out = []
    if not isinstance(desc, str):
        return out
    found_fence = False

    def _collect(obj):
        if isinstance(obj, dict):
            out.append(obj)
        elif isinstance(obj, list):
            out.extend(o for o in obj if isinstance(o, dict))

    text = desc.strip()
    if text[:1] in ("{", "["):
        try:
            _collect(json.loads(text))
            if out:
                return out
        except Exception:
            pass
    for m in re.finditer(r"```([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)```", desc, re.S):
        found_fence = True
        lang = (m.group(1) or "").lower()
        if lang in ("python", "py", "python3"):
            continue
        try:
            _collect(json.loads(m.group(2).strip()))
        except Exception:
            continue
    if out:
        return out
    if found_fence:
        # Fenced content present but nothing parseable (e.g. ```python blocks
        # are skipped by design) — do not rescue JSON out of fenced code.
        return []
    # Last resort (unfenced prose): first balanced {...} or [...] blob.
    for opener, closer in (("[", "]"), ("{", "}")):
        start = desc.find(opener)
        if start < 0:
            continue
        depth, in_str, esc_ch = 0, False, False
        for i in range(start, len(desc)):
            ch = desc[i]
            if in_str:
                if esc_ch:
                    esc_ch = False
                elif ch == "\\":
                    esc_ch = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        _collect(json.loads(desc[start:i + 1]))
                    except Exception:
                        pass
                    break
        if out:
            return out
    return out

def _evo_accepted(mean_ic, p_value):
    """Mirror of the workflow's is_acceptable(): bool, or None when unknown."""
    if mean_ic is None:
        return None
    if abs(mean_ic) < _IC_THRESHOLD:
        return False
    if p_value is not None and p_value > _P_THRESHOLD:
        return False
    return True

_EVO_OVERFIT_FALSE = {"", "0", "0.0", "false", "no", "none", "null", "low"}

def _evo_overfit_flag(data, metrics):
    """True when overfit_risk (top level / metrics) or oos.is_overfit is set."""
    v = data.get("overfit_risk", metrics.get("overfit_risk"))
    if isinstance(v, str) and v.strip().lower() in _EVO_OVERFIT_FALSE:
        v = False
    oos = data.get("oos") if isinstance(data.get("oos"), dict) else {}
    return bool(v) or bool(oos.get("is_overfit"))

def _evo_parse_files():
    """Parse every output/*.json once into evolution node records."""
    records, skipped = [], 0
    if not OUTPUT.exists():
        return records, skipped
    for p in OUTPUT.glob("*.json"):
        m = _EVO_FILE_RE.match(p.name)
        if not m:
            skipped += 1
            continue
        try:
            data = json.loads(p.read_text())
        except Exception:
            skipped += 1
            continue
        if not isinstance(data, dict):
            skipped += 1
            continue
        try:
            file_iter = int(m.group(2))
        except Exception:
            file_iter = 1
        try:
            iteration = int(data.get("iteration", file_iter))
        except (TypeError, ValueError):
            iteration = file_iter
        em = data.get("evaluation_metrics")
        if not isinstance(em, dict):
            em = {}
        # All numeric metric keys found (evaluation_metrics + top level).
        metrics = {}
        for src in (em, data):
            for k, v in src.items():
                if k in _EVO_NON_METRIC_KEYS or isinstance(v, bool):
                    continue
                if isinstance(v, (int, float)):
                    metrics[k] = v
        lineage = data.get("lineage") if isinstance(data.get("lineage"), dict) else {}
        oos = data.get("oos") if isinstance(data.get("oos"), dict) else {}
        # Optional skeptic-gate verdicts (new-format artifacts only; absent for
        # legacy files). Tolerant: missing/malformed -> empty list, kept as dicts.
        skeptic_reviews = [rv for rv in (data.get("skeptic_reviews") or [])
                           if isinstance(rv, dict)]
        mean_ic = _evo_num(em.get("mean_ic"))
        p_value = _evo_num(em.get("p_value"))
        entries = _parse_signal_description(data.get("signal_description"))
        records.append({
            "id": p.stem,
            "file": p.name,
            "run_id": str(data.get("run_id") or m.group(1)),
            "iteration": iteration,
            "signal_name": data.get("selected_signal"),
            "timestamp": data.get("timestamp") if isinstance(data.get("timestamp"), str) else None,
            "ts_iso": _evo_ts_iso(data.get("timestamp")) or _evo_ts_iso(m.group(1)),
            "mean_ic": mean_ic,
            "ic_ir": _evo_num(em.get("ic_ir")),
            "t_stat": _evo_num(em.get("t_stat")),
            "p_value": p_value,
            "formula": entries[0].get("formula") if entries else None,
            "category": entries[0].get("category") if entries else None,
            "source": lineage.get("source"),
            "parent_signal": lineage.get("parent_signal"),
            "parent_run_id": lineage.get("parent_run_id"),
            "code_len": len(data["signal_code"]) if isinstance(data.get("signal_code"), str) else 0,
            "metrics": metrics,
            "accepted": _evo_accepted(mean_ic, p_value),
            "overfit": _evo_overfit_flag(data, metrics),
            "oos_is_overfit": bool(oos.get("is_overfit")),
            "overfit_risk": data.get("overfit_risk", metrics.get("overfit_risk")),
            "request": data.get("request") if isinstance(data.get("request"), str) else None,
            "skeptic_reviews": skeptic_reviews,
        })
    return records, skipped

def _evo_data():
    """Cached parse of the output directory, invalidated when it changes."""
    key = _cache_key()
    if _EVO_CACHE["key"] == key and _EVO_CACHE["parsed"] is not None:
        return _EVO_CACHE["parsed"]
    records, skipped = _evo_parse_files()
    data = {"records": records, "skipped": skipped}
    _EVO_CACHE["key"] = key
    _EVO_CACHE["parsed"] = data
    return data

def _evo_node_public(r):
    """Node summary shape shared by /api/evolution and run detail."""
    node = {
        "id": r["id"], "run_id": r["run_id"], "iteration": r["iteration"],
        "signal_name": r["signal_name"], "timestamp": r["timestamp"],
        "mean_ic": r["mean_ic"], "ic_ir": r["ic_ir"], "t_stat": r["t_stat"],
        "p_value": r["p_value"], "formula": r["formula"],
        "source": r["source"], "parent_signal": r["parent_signal"],
        "code_len": r["code_len"], "metrics": dict(r["metrics"]),
        "accepted": r["accepted"],
        # extras consumed by the UI
        "file": r["file"], "ts_iso": r["ts_iso"], "category": r["category"],
        "parent_run_id": r["parent_run_id"], "overfit": r["overfit"],
        "oos_is_overfit": r["oos_is_overfit"], "overfit_risk": r["overfit_risk"],
    }
    if r.get("skeptic_reviews"):  # optional: absent for legacy artifacts
        node["skeptic_reviews"] = r["skeptic_reviews"]
    return node

def _evo_sort_key(n):
    return (n["iteration"], n["ts_iso"] or "", n["id"])

def _evo_group_runs(records):
    """Group node records into runs (run_id), newest first."""
    groups = {}
    for r in records:
        g = groups.setdefault(r["run_id"], {"run_id": r["run_id"], "request": None, "nodes": []})
        g["nodes"].append(r)
        if g["request"] is None and r["request"]:
            g["request"] = r["request"]
    out = []
    for g in groups.values():
        g["nodes"].sort(key=_evo_sort_key)
        g["started_at"] = min((n["ts_iso"] for n in g["nodes"] if n["ts_iso"]), default=None)
        best = None
        for n in g["nodes"]:
            if best is None or (abs(n["mean_ic"]) if n["mean_ic"] is not None else -1.0) > \
               (abs(best["mean_ic"]) if best["mean_ic"] is not None else -1.0):
                best = n
        g["best"] = best
        g["status"] = "accepted" if any(n["accepted"] for n in g["nodes"]) else "best_effort"
        out.append(g)
    out.sort(key=lambda g: g["started_at"] or "", reverse=True)
    return out

def _evo_build_tree(nodes):
    """Nested lineage tree for one run; supports branching (crossover siblings).

    Root = node whose parent_signal is null/absent or points outside this run
    (lowest iteration wins); children are matched by parent_signal name.
    """
    if not nodes:
        return {"root": None, "roots": []}
    by_name = {}
    for n in nodes:
        by_name.setdefault(n["signal_name"], []).append(n)
    children = {n["id"]: [] for n in nodes}
    has_parent = set()
    for n in sorted(nodes, key=_evo_sort_key):
        parent = n["parent_signal"]
        if not parent:
            continue
        cands = [c for c in by_name.get(parent, []) if c["id"] != n["id"]]
        if not cands:
            continue  # parent outside this run -> root
        earlier = [c for c in cands if _evo_sort_key(c) < _evo_sort_key(n)]
        pn = (earlier or cands)[-1]
        children[pn["id"]].append(n)
        has_parent.add(n["id"])
    roots = [n for n in nodes if n["id"] not in has_parent]
    if not roots:  # defensive: cyclic lineage -> break it at the earliest node
        roots = [min(nodes, key=_evo_sort_key)]

    def wrap(n, seen):
        if n["id"] in seen:
            return {**_evo_node_public(n), "children": []}
        seen = seen | {n["id"]}
        return {**_evo_node_public(n),
                "children": [wrap(c, seen) for c in sorted(children.get(n["id"], []), key=_evo_sort_key)]}

    roots.sort(key=_evo_sort_key)
    root = wrap(roots[0], set())
    if len(roots) > 1:  # multiple roots -> synthetic wrapper keeps one render root
        root = {"id": None, "signal_name": None,
                "children": [root] + [wrap(r, set()) for r in roots[1:]]}
    return {"root": root, "roots": [r["id"] for r in roots]}

def _evo_summary(records, groups):
    best = None
    accepted = 0
    overfit = 0
    for r in records:
        if r["accepted"] is True:
            accepted += 1
        if r["overfit"]:
            overfit += 1
        if r["mean_ic"] is not None and (best is None or abs(r["mean_ic"]) > abs(best["mean_ic"])):
            best = r
    return {
        "total_runs": len(groups),
        "total_signals": len(records),
        "best_ic": best["mean_ic"] if best else None,
        "best_signal": best["signal_name"] if best else None,
        "best_run_id": best["run_id"] if best else None,
        "accepted_count": accepted,
        "overfit_count": overfit,
    }

@app.get("/api/evolution")
@app.get("/api/evolution/{run_id}")
def evolution(run_id: str | None = None):
    """Signal evolution: runs of iteration nodes with lineage trees.

    GET /api/evolution        -> {"runs":[...], "tree":[...], "summary":{...}}
    GET /api/evolution/{id}   -> one run incl. per-iteration signal code,
                                 formulas and optimization feedback.
    Only parses saved JSON files; never executes signal code.
    """
    try:
        parsed = _evo_data()
    except Exception as e:
        logger.exception("evolution: failed to parse output dir")
        empty = {"total_runs": 0, "total_signals": 0, "best_ic": None,
                 "best_signal": None, "accepted_count": 0, "overfit_count": 0}
        return JSONResponse({"error": str(e), "runs": [], "tree": [], "summary": empty})
    records = parsed["records"]
    groups = _evo_group_runs(records)
    if run_id is None:
        runs, trees = [], []
        for g in groups[:_EVO_MAX_RUNS]:
            tree = _evo_build_tree(g["nodes"])
            trees.append({"run_id": g["run_id"], "root": tree["root"]})
            runs.append({
                "run_id": g["run_id"], "request": g["request"],
                "started_at": g["started_at"], "status": g["status"],
                "iterations": [_evo_node_public(n) for n in g["nodes"]],
                "best": _evo_node_public(g["best"]) if g["best"] else None,
                "tree": tree,
            })
        return JSONResponse({
            "runs": runs, "tree": trees,
            "summary": _evo_summary(records, groups),
            "runs_shown": len(runs), "runs_total": len(groups),
            "skipped_files": parsed["skipped"],
        })
    # Single-run detail (with code + feedback per iteration).
    g = next((g for g in groups if g["run_id"] == run_id), None)
    if g is None:
        return JSONResponse({"error": f"run {run_id} not found"}, status_code=404)
    details = []
    for n in g["nodes"]:
        node = _evo_node_public(n)
        try:
            raw = json.loads((OUTPUT / n["file"]).read_text())
        except Exception as e:
            raw = {}
            node["read_error"] = str(e)
        node["code"] = raw.get("signal_code") if isinstance(raw.get("signal_code"), str) else None
        node["descriptors"] = _parse_signal_description(raw.get("signal_description"))
        node["feedback"] = raw.get("last_feedback") if isinstance(raw.get("last_feedback"), str) else None
        details.append(node)
    best_id = g["best"]["id"] if g["best"] else None
    best_detail = next((d for d in details if d["id"] == best_id), None)
    feedbacks = [d["feedback"] for d in details if d["feedback"]]
    return JSONResponse({
        "run_id": run_id, "request": g["request"], "started_at": g["started_at"],
        "status": g["status"], "iterations": details,
        "best": best_detail, "tree": _evo_build_tree(g["nodes"]),
        "last_feedback": feedbacks[-1] if feedbacks else None,
    })

@app.get("/api/metrics/summary")
def metrics_summary():
    """Aggregated quantitative metrics over all saved results.

    Per-metric availability counts, min/max values, and runs flagged as
    overfit (overfit_risk / oos.is_overfit). Defensive: missing keys never
    break the response.
    """
    try:
        parsed = _evo_data()
    except Exception as e:
        logger.exception("metrics/summary: failed to parse output dir")
        return JSONResponse({"error": str(e), "total_results": 0,
                             "metric_availability": {}, "metric_stats": {},
                             "overfit_runs": [], "overfit_count": 0, "summary": {}})
    records = parsed["records"]
    groups = _evo_group_runs(records)
    stats = {}
    for r in records:
        for k, v in r["metrics"].items():
            st = stats.setdefault(k, {"count": 0, "max": None, "min": None,
                                      "max_signal": None, "min_signal": None,
                                      "max_run_id": None, "min_run_id": None})
            st["count"] += 1
            if st["max"] is None or v > st["max"]:
                st["max"], st["max_signal"], st["max_run_id"] = v, r["signal_name"], r["run_id"]
            if st["min"] is None or v < st["min"]:
                st["min"], st["min_signal"], st["min_run_id"] = v, r["signal_name"], r["run_id"]
    total = len(records)
    availability = {k: {"count": st["count"],
                        "pct": round(st["count"] * 100.0 / total, 2) if total else 0.0}
                    for k, st in sorted(stats.items())}
    metric_stats = {k: {kk: vv for kk, vv in st.items() if kk != "count"}
                    for k, st in sorted(stats.items())}
    overfit_runs = [{"run_id": r["run_id"], "signal": r["signal_name"],
                     "iteration": r["iteration"], "file": r["file"],
                     "overfit_risk": r["overfit_risk"], "oos_is_overfit": r["oos_is_overfit"]}
                    for r in records if r["overfit"]]
    return JSONResponse({
        "total_results": total,
        "metric_availability": availability,
        "metric_stats": metric_stats,
        "overfit_runs": overfit_runs[:200],
        "overfit_count": len(overfit_runs),
        "summary": _evo_summary(records, groups),
    })
