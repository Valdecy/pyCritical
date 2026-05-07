###############################################################################
# pyCritical - Web Launcher
#
# Provides:
#   - web_app(host='127.0.0.1', port=5000, open_browser=True)
#       Starts a local Flask server in a background thread serving the
#       interactive CPM / PERT workbench. Returns the server URL.
#
#   - web_stop()
#       Cleanly shuts down a server previously started with web_app().
#
# The server exposes JSON endpoints that wrap the four core analytical
# functions and the two probability helpers from cpm_pert.py.
###############################################################################

import ast
import json
import math
import os
import re
import threading
import time
import webbrowser

from flask import Flask, jsonify, render_template, request
from werkzeug.serving import make_server

from ..cpm_pert import (
    critical_path_method,
    critical_path_method_dep,
    pert_method,
    pert_method_dep,
    date_prob,
    date_required,
)

# ---------------------------------------------------------------------------
# Server-state container (module-level)
# ---------------------------------------------------------------------------

_state = {
    "server": None,   # werkzeug BaseWSGIServer
    "thread": None,   # serving thread
    "url": None,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df_to_records(df):
    """Convert a pandas DataFrame (indexed by task id) into a list of dicts.

    Critical tasks are flagged via Slack == 0 (matches the original library's
    convention). NaN / inf values are sanitised so JSON encoding never fails.
    """
    records = []
    for tid, row in df.iterrows():
        rec = {"id": str(tid)}
        for col in df.columns:
            v = row[col]
            try:
                fv = float(v)
                if math.isnan(fv) or math.isinf(fv):
                    fv = None
                rec[col] = fv
            except (TypeError, ValueError):
                rec[col] = v
        rec["critical"] = (rec.get("Slack") == 0.0)
        records.append(rec)
    return records


def _normalise_simple_dataset(tasks, mode):
    """Convert JSON task list -> dataset tuple expected by cpm_pert functions.

    mode == 'cpm'  -> (id, [pred_id, ...], duration)
    mode == 'pert' -> (id, [pred_id, ...], O, M, P)
    """
    out = []
    for t in tasks:
        tid = str(t["id"]).strip()
        preds = [str(p).strip() for p in t.get("preds", []) if str(p).strip()]
        if mode == "cpm":
            out.append((tid, preds, float(t["duration"])))
        else:
            out.append((tid, preds, float(t["o"]), float(t["m"]), float(t["p"])))
    return out


def _normalise_dep_dataset(tasks, mode):
    """Same as above but for the dependency-aware variants.

    Each predecessor entry is {id, type ('FS'|'SS'|'FF'|'SF'), lag (number)}.
    """
    out = []
    for t in tasks:
        tid = str(t["id"]).strip()
        preds = []
        for p in t.get("preds", []):
            pid = str(p["id"]).strip()
            if not pid:
                continue
            ptype = str(p.get("type", "FS")).upper()
            if ptype not in ("FS", "SS", "FF", "SF"):
                ptype = "FS"
            lag = float(p.get("lag", 0) or 0)
            preds.append((pid, ptype, lag))
        if mode == "cpm":
            out.append((tid, preds, float(t["duration"])))
        else:
            out.append((tid, preds, float(t["o"]), float(t["m"]), float(t["p"])))
    return out


def _to_number(value, label):
    try:
        num = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be numeric.")
    if not math.isfinite(num):
        raise ValueError(f"{label} must be a finite number.")
    return num


def _clean_literal_text(text):
    cleaned_lines = []
    for raw_line in (text or '').splitlines():
        line = raw_line.split('#', 1)[0].rstrip()
        if line.strip():
            cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines).strip()
    match = re.search(r"\bdataset\s*=\s*(\[.*\])\s*$", cleaned, flags=re.IGNORECASE | re.DOTALL)
    return (match.group(1) if match else cleaned).strip()


def _normalise_preds_literal(preds_raw):
    if preds_raw is None:
        return [], False
    if not isinstance(preds_raw, (list, tuple)):
        raise ValueError("Dependencies must be provided as a list.")
    if len(preds_raw) == 0:
        return [], False

    if all(not isinstance(p, (list, tuple)) for p in preds_raw):
        preds = [str(p).strip() for p in preds_raw if str(p).strip()]
        return preds, False

    if any(not isinstance(p, (list, tuple)) for p in preds_raw):
        raise ValueError("Dependencies cannot mix plain predecessor ids with typed dependency tuples in the same row.")

    preds = []
    for i, p in enumerate(preds_raw, start=1):
        if len(p) not in (2, 3):
            raise ValueError(f"Typed dependency #{i} must have the form (id, type[, lag]).")
        pid = str(p[0]).strip()
        ptype = str(p[1]).strip().upper()
        if not pid:
            raise ValueError("Dependency predecessor id cannot be empty.")
        if ptype not in ("FS", "SS", "FF", "SF"):
            raise ValueError(f"Unsupported dependency type: {ptype}.")
        lag = 0 if len(p) == 2 else _to_number(p[2], f"Lag for dependency {pid}")
        preds.append({"id": pid, "type": ptype, "lag": lag})
    return preds, True


def _coerce_simple_preds_to_typed(tasks):
    for task in tasks:
        if task.get("preds") and task["preds"] and isinstance(task["preds"][0], str):
            task["preds"] = [
                {"id": pid, "type": "FS", "lag": 0}
                for pid in task["preds"]
            ]


def _parse_literal_dataset(text):
    cleaned = _clean_literal_text(text)
    if not cleaned:
        raise ValueError("Paste a dataset before parsing.")
    rows = ast.literal_eval(cleaned)
    if not isinstance(rows, (list, tuple)) or len(rows) == 0:
        raise ValueError("Dataset must be a non-empty list.")

    widths = set()
    parsed = []
    has_typed_deps = False
    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, (list, tuple)):
            raise ValueError(f"Row {idx} must be a list or tuple.")
        if len(row) not in (3, 5):
            raise ValueError(f"Row {idx} must have 3 or 5 elements.")
        widths.add(len(row))
        tid = str(row[0]).strip()
        if not tid:
            raise ValueError(f"Row {idx} has an empty task id.")
        preds, typed = _normalise_preds_literal(row[1])
        has_typed_deps = has_typed_deps or typed
        nums = [_to_number(v, f"Row {idx} duration field") for v in row[2:]]
        if len(row) == 3:
            parsed.append({"id": tid, "preds": preds, "duration": nums[0]})
        else:
            parsed.append({"id": tid, "preds": preds, "o": nums[0], "m": nums[1], "p": nums[2]})

    if len(widths) != 1:
        raise ValueError("All rows must share the same schema length (3 or 5 columns).")

    is_pert = (5 in widths)
    if has_typed_deps:
        _coerce_simple_preds_to_typed(parsed)
    method = ("pert" if is_pert else "cpm") + ("_dep" if has_typed_deps else "")
    return method, parsed


def _split_spreadsheet_row(line):
    if "\t" in line:
        return [cell.strip() for cell in line.split("\t")]
    cols = [cell.strip() for cell in re.split(r"\s{2,}", line.strip()) if cell.strip() or cell == ""]
    return cols


def _parse_pred_cell(cell):
    cell = (cell or "").strip()
    if not cell:
        return [], False

    if ":" in cell:
        raw_parts = [p.strip() for p in re.split(r"[;]+", cell) if p.strip()]
        if len(raw_parts) == 1 and "," in cell:
            raw_parts = [p.strip() for p in cell.split(",") if p.strip()]
        preds = []
        for part in raw_parts:
            pieces = [x.strip() for x in part.split(":")]
            if len(pieces) not in (2, 3):
                raise ValueError(f"Typed dependency '{part}' must look like predecessor:type[:lag].")
            pid, ptype = pieces[0], pieces[1].upper()
            if ptype not in ("FS", "SS", "FF", "SF"):
                raise ValueError(f"Unsupported dependency type: {ptype}.")
            lag = 0 if len(pieces) == 2 or pieces[2] == "" else _to_number(pieces[2], f"Lag for dependency {pid}")
            preds.append({"id": pid, "type": ptype, "lag": lag})
        return preds, True

    return [p.strip() for p in re.split(r"[;,]", cell) if p.strip()], False


def _parse_spreadsheet_dataset(text):
    lines = []
    for raw_line in (text or '').splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        lines.append(raw_line.rstrip("\n"))
    if not lines:
        raise ValueError("Paste tab-separated spreadsheet rows before parsing.")

    rows = []
    for line in lines:
        cols = _split_spreadsheet_row(line)
        while cols and cols[-1] == "":
            cols.pop()
        rows.append(cols)

    if rows and len(rows[0]) >= 2:
        header0 = rows[0][0].strip().lower()
        header1 = rows[0][1].strip().lower()
        if header0 in ("id", "task", "activity") and header1 in ("dependencies", "dependency", "preds", "predecessors"):
            rows = rows[1:]

    if not rows:
        raise ValueError("No spreadsheet rows were found after removing the header.")

    widths = {len(r) for r in rows}
    if len(widths) != 1 or list(widths)[0] not in (3, 5):
        raise ValueError("Spreadsheet paste must have exactly 3 columns (ID, Dependencies, Duration) or 5 columns (ID, Dependencies, O, M, P).")

    width = list(widths)[0]
    is_pert = width == 5
    has_typed_deps = False
    parsed = []

    for idx, cols in enumerate(rows, start=1):
        tid = str(cols[0]).strip()
        if not tid:
            raise ValueError(f"Spreadsheet row {idx} has an empty task id.")
        preds, typed = _parse_pred_cell(cols[1] if len(cols) > 1 else "")
        has_typed_deps = has_typed_deps or typed
        nums = [_to_number(v, f"Spreadsheet row {idx} duration field") for v in cols[2:]]
        if is_pert:
            parsed.append({"id": tid, "preds": preds, "o": nums[0], "m": nums[1], "p": nums[2]})
        else:
            parsed.append({"id": tid, "preds": preds, "duration": nums[0]})

    if has_typed_deps:
        _coerce_simple_preds_to_typed(parsed)

    method = ("pert" if is_pert else "cpm") + ("_dep" if has_typed_deps else "")
    return method, parsed


def _format_python_dataset(method, tasks):
    lines = ["dataset = ["]
    dep = method.endswith("_dep")
    pert = method.startswith("pert")
    for task in tasks:
        if dep:
            preds = [(p["id"], p.get("type", "FS"), p.get("lag", 0)) for p in task.get("preds", [])]
            row = (task["id"], preds, task["o"], task["m"], task["p"]) if pert else (task["id"], preds, task["duration"])
        else:
            preds = list(task.get("preds", []))
            row = [task["id"], preds, task["o"], task["m"], task["p"]] if pert else [task["id"], preds, task["duration"]]
        lines.append(f"    {repr(row)},")
    lines.append("]")
    return "\n".join(lines)


def _parse_pasted_dataset(text):
    literal_error = None
    try:
        method, tasks = _parse_literal_dataset(text)
        return {
            "method": method,
            "tasks": tasks,
            "source": "literal",
            "normalized_text": _format_python_dataset(method, tasks),
        }
    except Exception as exc:
        literal_error = exc

    try:
        method, tasks = _parse_spreadsheet_dataset(text)
        return {
            "method": method,
            "tasks": tasks,
            "source": "spreadsheet",
            "normalized_text": _format_python_dataset(method, tasks),
        }
    except Exception as exc:
        raise ValueError(f"Could not parse the pasted dataset. Literal parse failed with: {literal_error}. Spreadsheet parse failed with: {exc}.")


# ---------------------------------------------------------------------------
# Flask application factory
# ---------------------------------------------------------------------------

def _create_app():
    here = os.path.dirname(os.path.abspath(__file__))
    app = Flask(
        __name__,
        template_folder=os.path.join(here, "templates"),
        static_folder=os.path.join(here, "static"),
    )

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/parse_dataset", methods=["POST"])
    def api_parse_dataset():
        try:
            payload = request.get_json(force=True)
            parsed = _parse_pasted_dataset(payload.get("text", ""))
            return jsonify({
                "ok": True,
                "method": parsed["method"],
                "tasks": parsed["tasks"],
                "source": parsed["source"],
                "normalized_text": parsed["normalized_text"],
                "count": len(parsed["tasks"]),
            })
        except Exception as e:
            return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}" if str(e) else type(e).__name__}), 400

    # ---- CPM (no dependencies) ------------------------------------------
    @app.route("/api/cpm", methods=["POST"])
    def api_cpm():
        try:
            payload = request.get_json(force=True)
            dataset = _normalise_simple_dataset(payload["tasks"], "cpm")
            df = critical_path_method(dataset)
            finish = float(df["EF"].max())
            return jsonify({
                "ok": True,
                "method": "cpm",
                "tasks": _df_to_records(df),
                "finish_time": finish,
                "dataset": [
                    {"id": tid, "preds": preds, "duration": dur}
                    for (tid, preds, dur) in dataset
                ],
            })
        except Exception as e:
            return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}" if str(e) else type(e).__name__}), 400

    # ---- PERT (no dependencies) -----------------------------------------
    @app.route("/api/pert", methods=["POST"])
    def api_pert():
        try:
            payload = request.get_json(force=True)
            dataset = _normalise_simple_dataset(payload["tasks"], "pert")
            df, finish, std_dev = pert_method(dataset)
            return jsonify({
                "ok": True,
                "method": "pert",
                "tasks": _df_to_records(df),
                "finish_time": float(finish),
                "std_dev": float(std_dev),
                "dataset": [
                    {"id": tid, "preds": preds, "o": o, "m": m, "p": p}
                    for (tid, preds, o, m, p) in dataset
                ],
            })
        except Exception as e:
            return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}" if str(e) else type(e).__name__}), 400

    # ---- CPM with typed dependencies + lags -----------------------------
    @app.route("/api/cpm_dep", methods=["POST"])
    def api_cpm_dep():
        try:
            payload = request.get_json(force=True)
            dataset = _normalise_dep_dataset(payload["tasks"], "cpm")
            df = critical_path_method_dep(dataset)
            finish = float(df["EF"].max())
            return jsonify({
                "ok": True,
                "method": "cpm_dep",
                "tasks": _df_to_records(df),
                "finish_time": finish,
                "dataset": [
                    {
                        "id": tid,
                        "preds": [
                            {"id": pid, "type": ptype, "lag": lag}
                            for (pid, ptype, lag) in preds
                        ],
                        "duration": dur,
                    }
                    for (tid, preds, dur) in dataset
                ],
            })
        except Exception as e:
            return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}" if str(e) else type(e).__name__}), 400

    # ---- PERT with typed dependencies + lags ----------------------------
    @app.route("/api/pert_dep", methods=["POST"])
    def api_pert_dep():
        try:
            payload = request.get_json(force=True)
            dataset = _normalise_dep_dataset(payload["tasks"], "pert")
            df, finish, std_dev = pert_method_dep(dataset)
            return jsonify({
                "ok": True,
                "method": "pert_dep",
                "tasks": _df_to_records(df),
                "finish_time": float(finish),
                "std_dev": float(std_dev),
                "dataset": [
                    {
                        "id": tid,
                        "preds": [
                            {"id": pid, "type": ptype, "lag": lag}
                            for (pid, ptype, lag) in preds
                        ],
                        "o": o, "m": m, "p": p,
                    }
                    for (tid, preds, o, m, p) in dataset
                ],
            })
        except Exception as e:
            return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}" if str(e) else type(e).__name__}), 400

    # ---- Probability helpers --------------------------------------------
    @app.route("/api/probability", methods=["POST"])
    def api_probability():
        try:
            payload = request.get_json(force=True)
            mean = float(payload["mean"])
            dp = float(payload["std_dev"])
            if dp <= 0:
                return jsonify({"ok": False, "error": "Standard deviation must be > 0."}), 400
            mode = payload.get("mode", "date")
            if mode == "date":
                date = float(payload["date"])
                p = float(date_prob(date, mean, dp))
                return jsonify({"ok": True, "mode": "date", "date": date, "probability": p})
            else:
                prob = float(payload["probability"])
                prob = max(min(prob, 0.9999), 0.0001)
                d = float(date_required(prob, mean, dp))
                return jsonify({"ok": True, "mode": "probability", "probability": prob, "date": d})
        except Exception as e:
            return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}" if str(e) else type(e).__name__}), 400

    # ---- Health & shutdown ----------------------------------------------
    @app.route("/api/health")
    def api_health():
        return jsonify({"ok": True, "service": "pyCritical web", "url": _state.get("url")})

    return app


# ---------------------------------------------------------------------------
# Public API: web_app / web_stop
# ---------------------------------------------------------------------------

def web_app(host="127.0.0.1", port=5000, open_browser=True, verbose=True):
    """Launch the pyCritical web workbench in a background thread.

    Parameters
    ----------
    host : str
        Interface to bind. Defaults to localhost (recommended).
    port : int
        TCP port. If already serving, returns the existing URL.
    open_browser : bool
        If True, opens the user's default browser at the served URL.
    verbose : bool
        If True, prints a friendly status line.

    Returns
    -------
    str
        The URL the server is listening on (e.g. 'http://127.0.0.1:5000').
    """
    if _state["server"] is not None:
        url = _state["url"]
        if verbose:
            print(f"[pyCritical] web app already running at {url}")
        return url

    app = _create_app()
    server = make_server(host, port, app, threaded=True)
    thread = threading.Thread(
        target=server.serve_forever,
        name="pyCritical-web",
        daemon=True,
    )
    thread.start()

    actual_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    url = f"http://{actual_host}:{port}"

    _state["server"] = server
    _state["thread"] = thread
    _state["url"] = url

    # Tiny wait to ensure socket is accepting connections before we open the browser.
    time.sleep(0.25)

    if verbose:
        print(f"[pyCritical] web app running at {url}")
        print("[pyCritical] call pyCritical.web_stop() to shut it down.")

    if open_browser:
        try:
            webbrowser.open(url, new=2)
        except Exception:
            pass

    return url


def web_stop(verbose=True):
    """Stop the pyCritical web workbench started by web_app()."""
    server = _state.get("server")
    thread = _state.get("thread")
    if server is None:
        if verbose:
            print("[pyCritical] no web app is running.")
        return False
    try:
        server.shutdown()        # signals serve_forever() to exit
        server.server_close()    # releases the socket
    except Exception as e:
        if verbose:
            print(f"[pyCritical] shutdown raised: {e}")
    if thread is not None:
        thread.join(timeout=3)
    _state["server"] = None
    _state["thread"] = None
    url = _state["url"]
    _state["url"] = None
    if verbose:
        print(f"[pyCritical] web app stopped ({url}).")
    return True
