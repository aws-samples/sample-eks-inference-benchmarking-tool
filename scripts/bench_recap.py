#!/usr/bin/env python3
"""Write a markdown recap table of every benchmark in a campaign, and keep it current.

Each row comes from a local rows file (see --rows), one planned or finished benchmark per line,
fields separated by a spaced pipe " | " so regexes inside a field may use a bare "|":

    model_hf | instance | weights | kv | backend | shape | result_glob | lane | stop_check | fallback | note

Status is decided in this order, so the table never claims more than the evidence shows:
  1. A result file matching result_glob exists -> Complete, with tok/s at concurrency 512 read
     from its AGGREGATE table, input tokens from its MEASURED TOKEN SHAPE block, mean output
     tokens as tok/s over req/s, and the KV dtype from its provenance block.
  2. stop_check ("file~regex~status") matches -> that status (an orchestrator gave up or refused).
  3. lane names an entry in the bench_status lanes file -> its live state (Measuring, Loading,
     Queued) with an ETA, computed exactly as bench_status.py does.
  4. Otherwise the row's fallback status (Refused, Not possible, Not run, Skipped).

Measured results are private (see .gitignore), so write the output under docs/reports/, which is
ignored, and keep the rows file under docs/results/. The report's prose quotes measured numbers,
so it lives in a local JSON notes file (--notes, also under docs/results/) rather than in this script.
"""
import argparse
import datetime as dt
import glob
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_status  # noqa: E402  (shares lane parsing, live state and ETA logic)

STATE_LABEL = {"DONE": "Complete", "MEASURING": "Measuring", "GATING": "Measuring (accuracy check)",
               "DEPLOYING": "Loading", "QUEUED": "Queued", "UNKNOWN": "Unknown (cluster unreachable)"}
COLUMNS = ["#", "Model ID (Hugging Face)", "Instance type", "Input tokens", "Output tokens",
           "Model weights", "KV cache", "Attention backend", "tok/s (c=512)", "TTFT p50 (c=1)", "Serving layout"]
PROFILE = "app/bench/data/prompt_profiles/coding_repo_context.json"


SKIP_PREFIXES = ("Skipped", "Not run", "Not possible", "Refused", "Failed", "Cancelled")
INFLIGHT_PREFIXES = ("Measuring", "Loading", "Queued", "Unknown")
SECTIONS = ("main", "rejected")


def load_rows(path: str) -> list[dict]:
    """Parse the rows file, failing loudly on a malformed line rather than dropping it.

    An optional 12th field names the section: "rejected" for a finished run whose instance is too
    small for the model. Rows with no result that were skipped, refused or not possible are sorted
    into the Skipped section by their status, so they need no marking here.
    """
    names = ["model", "instance", "weights", "kv", "backend", "shape", "glob", "lane", "stop", "fallback", "note"]
    rows = []
    for number, raw in enumerate(open(path), 1):
        line = raw.strip()
        if not line or line.startswith(("#", "exclude:")):
            continue
        # Fields split on " | " (spaced), so a stop_check regex can use an unspaced "|" for alternation.
        fields = [f.strip() for f in line.split(" | ")]
        if len(fields) not in (len(names), len(names) + 1):
            sys.exit(f"{path}:{number}: expected {len(names)} or {len(names) + 1} '|'-separated fields, "
                     f"got {len(fields)}")
        row = {k: (None if v == "-" else v) for k, v in zip(names, fields)}
        row["section"] = fields[len(names)] if len(fields) > len(names) else "main"
        if row["section"] not in SECTIONS:
            sys.exit(f"{path}:{number}: section must be one of {SECTIONS}, got {row['section']!r}")
        rows.append(row)
    return rows


def load_excludes(path: str) -> list[str]:
    """Result-file globs deliberately left out of every table ("exclude: <glob>" lines in the rows file).

    Without these the unlisted-results check would report a file that was removed on purpose.
    """
    return [line.split(":", 1)[1].split("#", 1)[0].strip() for line in open(path)
            if line.startswith("exclude:")]


def numeric_tok(cell: dict) -> float | None:
    """The c=512 tok/s as a number, or None for a dash or an n/a."""
    try:
        return float(cell["tok"].replace(",", ""))
    except ValueError:
        return None


def below_floor(cell: dict, floor: float) -> bool:
    """True for a finished run whose c=512 tok/s falls under the rejection floor."""
    tok = numeric_tok(cell)
    return cell["status"].startswith("Complete") and tok is not None and tok < floor


def section_of(row: dict, cell: dict, floor: float) -> str:
    """Skipped when there is no result and never will be; In flight while a lane is still working on it;
    Rejected when marked or under the floor; otherwise the row's own section."""
    if cell["status"].startswith(SKIP_PREFIXES):
        return "skipped"
    if cell["status"].startswith(INFLIGHT_PREFIXES):
        return "inflight"
    if below_floor(cell, floor):
        return "rejected"
    return row["section"]


def newest(results_dir: str, pattern: str | None) -> str | None:
    """Newest result file matching the pattern, skipping superseded copies unless the pattern names one."""
    if not pattern:
        return None
    keep_superseded = ".superseded-" in pattern
    files = [f for f in glob.glob(os.path.join(results_dir, pattern))
             if keep_superseded or ".superseded-" not in f]
    return max(files, key=os.path.getmtime) if files else None


def is_open_loop(text: str) -> bool:
    """True for an arrival-rate (server_qps) result, which has no concurrency-512 row."""
    return re.search(r"^\s+protocol\s+:\s+server_qps", text, re.M) is not None


def c512_row(text: str) -> tuple[float, float] | None:
    """(output tokens/s, requests/s) at concurrency 512 from the throughput table."""
    table = text.split("AGGREGATE", 1)[-1].split("REQUEST SUCCESS", 1)[0]
    match = re.search(r"^\s+512\s+([0-9.]+)\s+([0-9.]+)", table, re.M)
    return (float(match.group(1)), float(match.group(2))) if match else None


def c512_tokens(text: str) -> str:
    """Output tokens/s at concurrency 512, or why there is none."""
    if is_open_loop(text):
        return "n/a (open-loop)"
    row = c512_row(text)
    return f"{row[0]:,.0f}" if row else "—"


def c1_ttft(text: str) -> str:
    """Median time to first token with one request in flight, from the streaming latency table."""
    table = text.split("LATENCY (streaming)", 1)[-1].split("per-rep raw values", 1)[0]
    match = re.search(r"^\s+1\s+([0-9.]+)\s+[0-9.]+", table, re.M) if "LATENCY (streaming)" in text else None
    return f"{float(match.group(1)):,.2f} s" if match else "—"


def output_cap(text: str) -> int | None:
    """The per-request output cap the run used."""
    match = re.search(r"output cap:\s+(\d+) tok", text)
    return int(match.group(1)) if match else None


def mean_output(text: str) -> int | None:
    """Mean completion tokens per request at c=512: tok/s over req/s (req/s has 2 decimals, so ±0.5%)."""
    row = c512_row(text)
    return round(row[0] / row[1]) if row and row[1] > 0 else None


def input_tokens(text: str) -> int | None:
    """Server-counted median prompt tokens, chat template included."""
    if is_open_loop(text):
        header = re.search(r"^\s*tgt_qps.*$", text, re.M)
        first = re.search(r"^\s+[0-9.]+(?:\s+[0-9.]+)+\s*$", text[header.end():], re.M) if header else None
        if not first:
            return None
        column = header.group(0).split().index("input_tokens_median")
        return round(float(first.group(0).split()[column]))
    shape = text.split("MEASURED TOKEN SHAPE", 1)[-1]
    match = re.search(r"^\s+512\s+(\d+)\s+\d+\s+\d+\s+\d+", shape, re.M)
    return int(match.group(1)) if match else None


def reaches_cap(text: str, cap: int) -> bool:
    """True when req/s rounding (2 decimals) cannot rule out every request hitting the cap."""
    row = c512_row(text)
    return bool(row) and row[1] > 0.005 and row[0] / (row[1] - 0.005) >= cap


def output_cell(text: str) -> str:
    """Mean output against the cap, so early EOS stops are visible next to tok/s."""
    cap, mean = output_cap(text), mean_output(text)
    if cap and mean and reaches_cap(text, cap):
        return f"≈{cap:,} (at cap)"
    if cap and mean:
        return f"≈{mean:,} avg (cap {cap:,})"
    return f"cap {cap:,}" if cap else "—"


def recorded_kv(text: str) -> str | None:
    """KV dtype recorded by the serving pod's provenance, if the file carries it."""
    match = re.search(r"^\s+KV cache dtype\s+:\s+(\S+)", text, re.M)
    return match.group(1).upper() if match else None


def complete_row(row: dict, path: str) -> dict:
    """Fill a row from its result file."""
    text = open(path).read()
    kv = recorded_kv(text)
    kv_cell = kv if kv else f"{row['kv']} †"
    inp = input_tokens(text)
    status = "Complete (superseded)" if ".superseded-" in path else "Complete"
    return {"status": status, "tok": c512_tokens(text), "ttft": c1_ttft(text), "kv": kv_cell,
            "inp": f"{inp:,}" if inp else "—", "out": output_cell(text), "measured_inp": inp,
            "note": row["note"] or "", "path": os.path.basename(path), "dagger": kv is None,
            "c512": c512_row(text)}


def stop_status(spec: str | None) -> str | None:
    """Return the stop status when the named log matches its regex."""
    if not spec:
        return None
    path, pattern, label = spec.split("~")
    if os.path.exists(path) and re.search(pattern, open(path).read()):
        return label
    return None


def live_states(lanes_path: str, results_dir: str) -> dict:
    """Current state and ETA of every lane, exactly as the status monitor computes them."""
    lanes = bench_status.load_lanes(lanes_path)
    now = time.time()
    states = {}
    for lane in lanes:
        try:
            states[lane["name"]] = bench_status.lane_status(lane, results_dir, now)
        except bench_status.ClusterUnreachable as why:
            states[lane["name"]] = {"state": "UNKNOWN", "detail": str(why), "eta_s": None}
    bench_status.chain_etas(lanes, states)
    return states


def eta_suffix(state: dict, now: float, tz: tuple) -> str:
    """", ETA ~HH:MM TZ" for a lane with a known ETA, else nothing."""
    if state["state"] == "DONE" or not state.get("eta_s"):
        return ""
    clock = dt.datetime.fromtimestamp(now + state["eta_s"], dt.timezone.utc) + dt.timedelta(hours=tz[1])
    return f", ETA ~{clock:%H:%M} {tz[0]}"


def row_status(row: dict, results_dir: str, states: dict, now: float, tz: tuple) -> dict:
    """Decide one row's status, following the precedence in the module docstring."""
    path = newest(results_dir, row["glob"])
    if path:
        return complete_row(row, path)
    base = {"tok": "—", "ttft": "—", "path": None, "kv": row["kv"], "note": row["note"] or "", "dagger": False,
            "inp": "—", "out": "—", "measured_inp": None}
    stopped = stop_status(row["stop"])
    if stopped:
        return {**base, "status": stopped}
    if row["lane"] and row["lane"] in states:
        state = states[row["lane"]]
        label = STATE_LABEL.get(state["state"], state["state"].title())
        return {**base, "status": label + eta_suffix(state, now, tz)}
    return {**base, "status": row["fallback"] or "Unknown"}


def profile_cap(path: str) -> int | None:
    """The profile's output cap, for rows that have no result file to read it from."""
    try:
        return int(json.load(open(path))["output_tokens"])
    except (OSError, KeyError, ValueError):
        return None


def profile_of(row: dict) -> str:
    """The prompt profile a row's result file is named for; rows with no file use the default."""
    match = re.search(r"-corpus-([A-Za-z0-9_]+)(?:\.|-\*|\*)", row["glob"] or "")
    return match.group(1) if match else "coding_repo_context"


def fill_unmeasured(rows: list[dict], cells: list[dict], cap: int | None) -> None:
    """Give rows without a result the input size a finished run of the same model measured.

    The profile's prompts are pinned, so every run of one model sends the same tokens; the ‡ marks
    that the figure was measured on a sibling row rather than this one.
    """
    measured = {(r["model"], profile_of(r)): c["measured_inp"] for r, c in zip(rows, cells) if c["measured_inp"]}
    for row, cell in zip(rows, cells):
        key = (row["model"], profile_of(row))
        if cell["inp"] == "—" and key in measured:
            cell["inp"] = f"{measured[key]:,} ‡"
        if cell["out"] == "—" and cap:
            cell["out"] = f"cap {cap:,}"


INFLIGHT_COLUMNS = ["#", "Status", "Model ID (Hugging Face)", "Instance type", "Model weights", "KV cache",
                    "Attention backend", "Serving layout"]
SKIPPED_COLUMNS = ["#", "Status", "Model ID (Hugging Face)", "Instance type", "Model weights", "KV cache",
                   "Attention backend", "Serving layout", "Reason"]


def full_row(number: int, row: dict, cell: dict) -> list[str]:
    """One Results- or Rejected-table row, in COLUMNS order. Every row here is complete, so there is no
    Status column; a superseded run is marked § on its instance type instead."""
    instance = row["instance"] + (" §" if "superseded" in cell["status"] else "")
    return [str(number), f"`{row['model']}`", instance, cell["inp"], cell["out"],
            row["weights"], cell["kv"], row["backend"], cell["tok"], cell["ttft"], row["shape"]]


def inflight_row(number: int, row: dict, cell: dict) -> list[str]:
    """One In flight row: identity and status only, since nothing has been measured yet."""
    return [str(number), cell["status"], f"`{row['model']}`", row["instance"], row["weights"], row["kv"],
            row["backend"], row["shape"]]


def skipped_row(number: int, row: dict, cell: dict) -> list[str]:
    """One Skipped row: no measurement columns, since nothing was measured."""
    return [str(number), cell["status"], f"`{row['model']}`", row["instance"], row["weights"], row["kv"],
            row["backend"], row["shape"], cell["note"].replace("|", "\\|")]


def table(columns: list[str], lines: list[list[str]]) -> list[str]:
    """A markdown table, or a placeholder line when the section is empty."""
    if not lines:
        return ["_None._"]
    return ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)] + \
           ["| " + " | ".join(values) + " |" for values in lines]


def gpus_of(shape: str) -> int | None:
    """GPUs a serving layout uses: "8 copies × 1 GPU" -> 8, "1 copy × 8 GPUs (TP8)" -> 8, "1 GPU" -> 1."""
    numbers = [int(n) for n in re.findall(r"(\d+) (?:cop(?:y|ies)|GPUs?)", shape)]
    if not numbers:
        return None
    total = 1
    for n in numbers:
        total *= n
    return total


def input_rates(pairs: list[tuple[dict, dict]]) -> list[tuple[str, float, float, float]]:
    """(model, input tokens/s per node, per GPU, input tokens per output token) at c=512, per finished row."""
    rates = []
    for row, cell in pairs:
        c512, inp, gpus = cell.get("c512"), cell.get("measured_inp"), gpus_of(row["shape"])
        if not (c512 and inp and gpus and c512[0] > 0):
            continue
        per_node = c512[1] * inp
        rates.append((row["model"], per_node, per_node / gpus, per_node / c512[0]))
    return rates


def rates_sentence(main: list[tuple[dict, dict]]) -> str:
    """The generated input-token-rate sentence for the Methodology section, or "" before any result."""
    rates = input_rates(main)
    if not rates:
        return ""
    lo_node, hi_node = min(r[1] for r in rates), max(r[1] for r in rates)
    lo_gpu, hi_gpu = min(r[2] for r in rates), max(r[2] for r in rates)
    lo_ratio, hi_ratio = min(r[3] for r in rates), max(r[3] for r in rates)
    return (f" At c=512 the Results rows are also processing {lo_node:,.0f}–{hi_node:,.0f} input "
            f"tokens/s per node ({lo_gpu:,.0f}–{hi_gpu:,.0f} per GPU), {lo_ratio:.0f}–{hi_ratio:.0f} "
            "input tokens for every output token (req/s × the input-token median).")


def load_notes(path: str | None) -> dict:
    """The report's prose, kept in a local notes file because it quotes measured numbers.

    Measured results stay out of git, so this script carries only the table mechanics; the intro,
    Methodology, Conclusions and Capacity text live in a JSON file beside the rows file. Without
    one the report is the tables alone.
    """
    if not path or not os.path.exists(path):
        return {}
    try:
        return json.load(open(path))
    except json.JSONDecodeError as why:
        sys.exit(f"{path}: not valid JSON ({why})")


def finished(model: str, rows: list[dict], cells: list[dict]) -> list[tuple[dict, dict]]:
    """One model's completed, non-superseded runs that have a c=512 figure."""
    return [(r, c) for r, c in zip(rows, cells)
            if r["model"] == model and c["status"] == "Complete" and numeric_tok(c) is not None]


def best_line(model: str, rows: list[dict], cells: list[dict]) -> str | None:
    """"- Best measured: ..." for the model's highest-throughput finished run, or None before any finish."""
    done = finished(model, rows, cells)
    if not done:
        return None
    row, cell = max(done, key=lambda pair: numeric_tok(pair[1]))
    return (f"- Best measured: {row['instance']}, {row['backend']}, {row['shape']}: {cell['tok']} tok/s "
            f"at c=512, TTFT p50 {cell['ttft']} at c=1.")


def backend_comparison(pairs: list[tuple[dict, dict]]) -> list[str]:
    """One line per instance and layout measured with more than one backend, fastest first."""
    groups: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for row, cell in pairs:
        tok = numeric_tok(cell)
        if tok is not None:
            groups.setdefault((row["instance"], row["shape"]), []).append((row["backend"], tok))
    lines = []
    for (instance, shape), runs in groups.items():
        if len(runs) < 2:
            continue
        runs.sort(key=lambda r: -r[1])
        slowest = runs[-1][1]
        parts = [f"{b} {t:,.0f}" + (f" (+{t / slowest - 1:.0%})" if t > slowest else "") for b, t in runs]
        lines.append(f"{instance}, {shape}: " + ", ".join(parts) + " tok/s.")
    return lines


def model_summary(model: str, rows: list[dict], cells: list[dict]) -> list[str]:
    """Generated lines for one model: best finished run, backend comparisons, runs still going."""
    done = finished(model, rows, cells)
    going = [r for r, c in zip(rows, cells) if r["model"] == model and c["status"].startswith(INFLIGHT_PREFIXES)]
    lines = []
    if done:
        lines.append(best_line(model, rows, cells))
        lines += [f"- Backends on {line}" for line in backend_comparison(done)]
    if going:
        lines.append("- Still running: " + "; ".join(f"{r['instance']} {r['backend']}" for r in going) + ".")
    return lines


def methodology(notes: dict, main: list[tuple[dict, dict]]) -> list[str]:
    """The Methodology section from the notes, with the input-token rates filled in from the results."""
    lines = notes.get("methodology")
    if not lines:
        return []
    return ["", "## Methodology", ""] + [line.replace("{input_rates}", rates_sentence(main)) for line in lines]


def conclusions(notes: dict, rows: list[dict], cells: list[dict]) -> list[str]:
    """The Conclusions section: each model's note, then its generated results lines."""
    entries = notes.get("conclusions") or []
    out = ["", "## Conclusions"] if entries else []
    for entry in entries:
        out += ["", f"**{entry['title']}** (`{entry['model']}`). {entry['note']}", ""]
        out += model_summary(entry["model"], rows, cells)
    return out


def capacity(notes: dict, rows: list[dict], cells: list[dict]) -> list[str]:
    """The Capacity section: each model's sizing note, then its generated best result."""
    entries = notes.get("capacity") or []
    if not entries:
        return []
    model_ids = {e["title"]: e["model"] for e in notes.get("conclusions") or []}
    out = ["", "## Capacity", "", notes.get("capacity_intro", "")]
    for entry in entries:
        out += ["", f"**{entry['title']}.** {entry['note']}"]
        best = best_line(model_ids.get(entry["title"], ""), rows, cells)
        if best:
            out += ["", best]
    return out


def by_throughput(pairs: list[tuple[dict, dict]]) -> list[tuple[dict, dict]]:
    """Highest c=512 tok/s first; rows without a number keep their order at the end."""
    return sorted(pairs, key=lambda pair: -(numeric_tok(pair[1]) or float("-inf")))


def footnotes(cells: list[dict]) -> list[str]:
    """The ‡ § † markers' explanations, for whichever markers the tables used."""
    out = []
    if any(c["inp"].endswith("‡") for c in cells):
        out += ["", "‡ Not measured on this row: the median a finished run of the same model measured. The "
                "profile's prompts are pinned, so every run of one model sends the same tokens."]
    if any("superseded" in c["status"] for c in cells):
        out += ["", "§ Superseded: the first run of a configuration that was later re-run with the same settings; "
                "kept for the record next to the rerun."]
    if any(c["dagger"] for c in cells):
        out += ["", "† KV dtype not recorded in the result file (provenance capture empty or skipped). The recipe "
                "sets FP8 and its deploy script refuses to measure unless the live server reports "
                "`cache_dtype=\"fp8\"`."]
    return out


def render(rows: list[dict], cells: list[dict], now: float, tz: tuple, floor: float, notes: dict) -> str:
    """Build the markdown document: In flight, Results, Rejected, Skipped, then the notes' sections."""
    local = dt.datetime.fromtimestamp(now, dt.timezone.utc) + dt.timedelta(hours=tz[1])
    grouped = {"inflight": [], "main": [], "rejected": [], "skipped": []}
    for row, cell in zip(rows, cells):
        grouped[section_of(row, cell, floor)].append((row, cell))
    for section in ("main", "rejected"):
        grouped[section] = by_throughput(grouped[section])
    main = grouped["main"]
    out = [f"# {notes.get('title', 'Benchmark recap')}", "",
           f"Last updated {local:%Y-%m-%d %H:%M} {tz[0]} (regenerated automatically). "
           f"{len(grouped['inflight'])} in flight; {len(main)} in Results; {len(grouped['rejected'])} rejected; "
           f"{len(grouped['skipped'])} skipped.", ""]
    out += notes.get("intro", [])
    if grouped["inflight"]:  # the section disappears once every run has landed
        out += ["## In flight", "",
                "Runs still loading, measuring or queued, with ETAs. Each moves to Results or Rejected when "
                "its result file lands, or to Skipped if it fails or is refused.", ""]
        out += table(INFLIGHT_COLUMNS, [inflight_row(n, r, c) for n, (r, c) in enumerate(grouped["inflight"], 1)])
        out += [""]
    out += ["## Results", ""]
    out += table(COLUMNS, [full_row(n, r, c) for n, (r, c) in enumerate(main, 1)])
    rejected_intro = notes.get("rejected_intro", "Finished runs below {floor} tok/s (c=512).")
    out += ["", "## Rejected", "", rejected_intro.replace("{floor}", f"{floor:,.0f}"), ""]
    out += table(COLUMNS, [full_row(n, r, c) for n, (r, c) in enumerate(grouped["rejected"], 1)])
    out += ["", "## Skipped", "", notes.get("skipped_intro", "Configurations with no result."), ""]
    out += table(SKIPPED_COLUMNS, [skipped_row(n, r, c) for n, (r, c) in enumerate(grouped["skipped"], 1)])
    out += footnotes(cells)
    if notes.get("weights_note"):
        out += ["", notes["weights_note"]]
    out += methodology(notes, main)
    out += conclusions(notes, rows, cells)
    out += capacity(notes, rows, cells)
    out += [""]
    return "\n".join(out)


def unlisted_results(results_dir: str, cells: list[dict], excludes: list[str]) -> list[str]:
    """Campaign result files that no row reads and no exclude names, so a missing row shows up."""
    cited = {c["path"] for c in cells if c.get("path")}
    excluded = {os.path.basename(f) for pattern in excludes for f in glob.glob(os.path.join(results_dir, pattern))}
    files = {os.path.basename(f) for f in glob.glob(os.path.join(results_dir, "*coding_repo_context*.txt"))}
    return sorted(files - cited - excluded)


def generate(args) -> str:
    rows = load_rows(args.rows)
    states = live_states(args.lanes, args.results_dir)
    now = time.time()
    tz = (args.tz_label, args.tz_offset)
    cells = [row_status(r, args.results_dir, states, now, tz) for r in rows]
    fill_unmeasured(rows, cells, profile_cap(args.profile))
    doc = render(rows, cells, now, tz, args.reject_below, load_notes(args.notes))
    missing = unlisted_results(args.results_dir, cells, load_excludes(args.rows))
    if missing:
        doc += "\n**Result files with no row in this table** (add them to the rows file):\n\n" + \
               "\n".join(f"- `{name}`" for name in missing) + "\n"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # Written in place, not via a temp file and rename: a rename gives the path a new inode, and an
    # editor tab that already has the file open can keep showing the old one.
    with open(args.out, "w") as handle:
        handle.write(doc)
    return doc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--lanes", required=True, help="the bench_status.py lanes file")
    parser.add_argument("--results-dir", default="docs/results")
    parser.add_argument("--out", default="docs/reports/benchmark-recap.md")
    parser.add_argument("--profile", default=PROFILE, help="prompt profile, for the output cap of unrun rows")
    parser.add_argument("--notes", default="docs/results/bench-recap-notes.txt",
                        help="local JSON with the report's prose (title, intro, methodology, conclusions, capacity)")
    parser.add_argument("--reject-below", type=float, default=2000,
                        help="finished runs under this c=512 tok/s go to the Rejected table")
    parser.add_argument("--tz-label", default="UTC")
    parser.add_argument("--tz-offset", type=float, default=0.0)
    parser.add_argument("--interval-min", type=float, default=10)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    while True:
        generate(args)
        print(f"[{time.strftime('%H:%M:%S')}] wrote {args.out}", flush=True)
        if args.once:
            return
        time.sleep(args.interval_min * 60)


if __name__ == "__main__":
    main()
