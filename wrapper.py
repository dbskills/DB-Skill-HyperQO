#!/usr/bin/env python3.7
"""HyperQO Query Optimizer Skill — single-file long-lived HTTP server wrapper.

Hybrid MCTS + TreeLSTM optimizer. The model, KNN cache, replay memory, alias
map, and counters live in memory for the lifetime of the server process.

Lock-free, parallel optimize:
  * Training runs in a SEPARATE PROCESS (`wrapper.py --train --model-dir ...`),
    not a thread — it never touches the foreground's net, so it can't block or
    race optimize. Training-mode queries are appended to a pending-queries
    file; once `train_trigger` accumulate, a training subprocess is spawned
    that re-executes each pending query with `hinterRun(optimize_only=False)`
    (EXPLAIN ANALYZE + net/MCTS update) and saves atomically.
  * The foreground `optimize` path is cost-only (`hinterRun(optimize_only=True)`)
    and READ-ONLY w.r.t. the net/knn. It picks up the subprocess's new weights
    via an mtime check → snapshot-swap (load into fresh modules, atomically
    reassign references; in-flight readers keep the old object).
  * No inference lock. Thread-safety of concurrent optimize rests on:
      - Hinter.hinterRun uses per-call LOCAL bookkeeping lists (Hinter.py),
      - `PGUtils.pgrunner` is a thread-local proxy → each thread its own cursor,
      - `sql2fea.getColumnId` and `PGUtils.addLatency` are guarded by locks.
    Same-DSN workloads are fully parallel; the shared `config` DSN fields are
    set per call (benign for a single DSN).

Endpoints: GET /health, POST /optimize, GET /state, POST /shutdown.
"""
import argparse
import copy
import json
import os
import pickle
import signal
import subprocess
import sys
import threading
import time
from contextlib import redirect_stdout
import io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_DIR = os.path.join(SCRIPT_DIR, "state")
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")
os.makedirs(STATE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

MAX_KVS_SIZE_DEFAULT = 200000
# Batch N pending training queries per subprocess spawn (amortizes torch import
# + model load). The original code trained every query in-process; per-query
# subprocess spawn would be dominated by interpreter startup, so we batch.
TRAIN_TRIGGER_DEFAULT = 25
# Safe-gate for background training: if the chosen (hinted) plan's cost is
# more than this many times the PostgreSQL default plan's cost, discard it
# without enqueueing for training. A catastrophically worse plan would hang the
# training subprocess on a multi-minute execution; the cost ratio is a cheap
# foreground-visible proxy.
_MAX_COST_RATIO = 50.0


def _cost_ratio_too_high(chosen_cost, default_cost, max_ratio=_MAX_COST_RATIO):
    """Return True if the chosen plan's cost is catastrophically higher than
    the default plan's (so training should be skipped for this query).
    Unknown / non-positive costs → False (let the background try)."""
    if not chosen_cost or not default_cost:
        return False
    try:
        d = float(default_cost)
        if d <= 0:
            return False
        return float(chosen_cost) / d > max_ratio
    except (TypeError, ValueError, ZeroDivisionError):
        return False


def _atomic_write_bytes(path, data):
    tmp = "{}.tmp.{}".format(path, os.getpid())
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_torch_save(obj, path):
    tmp = "{}.tmp.{}".format(path, os.getpid())
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass


def _clone_state_dict(sd):
    """Snapshot a state_dict for unlocked disk writes. Model state_dicts are
    flat {key: tensor}; optimizer state_dicts are nested {state: {int: {...}},
    param_groups: [...]} whose values are dicts/lists, not tensors. A plain
    `v.detach().clone()` comprehension raises AttributeError on the nested
    dict values, so deepcopy handles arbitrary nesting and clones tensor
    storage. Returns None on failure so callers save what they can."""
    try:
        return copy.deepcopy(sd)
    except Exception:
        return None


def parse_dsn(dsn):
    if "://" in dsn:
        parsed = urlparse(dsn)
        return (parsed.path.lstrip("/"), parsed.username or "postgres",
                parsed.password or "", parsed.hostname or "127.0.0.1",
                parsed.port or 5432)
    parts = {}
    for item in dsn.split():
        k, _, v = item.partition("=")
        parts[k] = v
    return (parts.get("dbname", parts.get("database", "imdb")),
            parts.get("user", "postgres"), parts.get("password", ""),
            parts.get("host", "127.0.0.1"), int(parts.get("port", "5432")))


def extract_aliases_from_query(sql):
    from psqlparse import parse_dict
    try:
        parse_result = parse_dict(sql)[0]["SelectStmt"]
        aliases = []
        for item in parse_result.get("fromClause", []):
            range_var = item.get("RangeVar", {})
            if "alias" in range_var:
                alias_name = range_var["alias"]["Alias"]["aliasname"]
            else:
                alias_name = range_var.get("relname", "")
            aliases.append(alias_name)
        return aliases
    except Exception:
        return []


def build_dynamic_alias_map(base_id2aliasname, base_aliasname2id, query_aliases, config_alias_overrides):
    id2aliasname = dict(base_id2aliasname)
    aliasname2id = dict(base_aliasname2id)
    if config_alias_overrides and "aliasname2id" in config_alias_overrides:
        for alias_name, alias_id in config_alias_overrides["aliasname2id"].items():
            aliasname2id[alias_name] = alias_id
            id2aliasname[alias_id] = alias_name
    next_id = (max(id2aliasname.keys()) + 1) if id2aliasname else 1
    for alias_name in query_aliases:
        if alias_name and alias_name not in aliasname2id:
            aliasname2id[alias_name] = next_id
            id2aliasname[next_id] = alias_name
            next_id += 1
    return id2aliasname, aliasname2id


# --------------------------------------------------------------------------- #
# Skill: holds all in-memory state; one instance per server process.
# --------------------------------------------------------------------------- #
class HyperQOSkill:
    def __init__(self):
        self.loaded = False
        self.max_kvs_size = MAX_KVS_SIZE_DEFAULT
        self.train_trigger = TRAIN_TRIGGER_DEFAULT
        self.model_dir = MODEL_DIR
        self.state_dir = STATE_DIR
        self.config = None
        self.net = None
        self.value_network = None
        self.mcts_searcher = None
        self.hinter = None
        self.tree_builder = None
        self.sql2vec = None
        # _load_lock: serializes the rare model_dir-change reload (initial load
        # + config-driven reload). The common optimize path never takes it.
        self._load_lock = threading.Lock()
        # _state_lock: brief mutex for the pending-queue / training-spawn flag.
        # Never held during net access or training.
        self._state_lock = threading.Lock()
        self._training_in_progress = False
        self._train_proc = None  # Popen of the in-flight training subprocess
        self._shutting_down = False  # set by persist(); /optimize then 503s
        self._persist_lock = threading.Lock()  # single-flight: concurrent /shutdown runs persist once
        self._last_mtime = 0  # max mtime across persisted files at last reload
        self.trained_samples = 0  # queries processed by training subprocesses (synced via reload)
        self._drain_thread = None  # catch-up drainer; started by start_drainer() in the server only

    def _paths(self):
        state_file = os.path.join(self.state_dir, "hinter_state.pkl")
        column_id_file = os.path.join(self.state_dir, "column_id.pkl")
        alias_map_file = os.path.join(self.state_dir, "alias_map.pkl")
        value_net = os.path.join(self.model_dir, "value_network.pth")
        value_net_opt = os.path.join(self.model_dir, "value_network_optimizer.pth")
        mcts_opt = os.path.join(self.model_dir, "mcts_optimizer.pth")
        mcts_model = os.path.join(
            self.model_dir,
            self.config.log_file.split("/")[-1].split(".txt")[0] + ".pth",
        )
        return (state_file, column_id_file, alias_map_file, value_net,
                value_net_opt, mcts_opt, mcts_model)

    def _pending_file(self):
        return os.path.join(self.state_dir, "pending_queries.jsonl")

    def _load(self):
        sys.path.insert(0, SCRIPT_DIR)
        import ImportantConfig

        config = ImportantConfig.Config()
        config.usegpu = False
        config.device = torch.device("cpu")
        config.cpudevice = torch.device("cpu")
        config.latency_file = os.path.join(self.state_dir, "latency_record.txt")
        config.log_file = os.path.join(self.state_dir, "log_hypqo.txt")
        config.modelpath = os.path.join(self.model_dir, "")
        # Full-order Leading hints (leading_length=-1). try_hint_num (source
        # default 3) caps how many MCTS candidate orders are evaluated per query
        # — each needs a cost-only EXPLAIN + a forward + a KNN lookup, so the
        # per-query cost and the 17-table tail scale ~linearly with it. Left at
        # the source default; raise via --config for more hint coverage at the
        # cost of latency. Both are --config-overridable (see SKILL.md).
        config.leading_length = -1
        self.config = config

        alias_map_file = os.path.join(self.state_dir, "alias_map.pkl")
        if os.path.exists(alias_map_file):
            try:
                with open(alias_map_file, "rb") as f:
                    persisted = pickle.load(f)
                config.id2aliasname = persisted.get("id2aliasname", config.id2aliasname)
                config.aliasname2id = persisted.get("aliasname2id", config.aliasname2id)
            except Exception:
                pass

        ImportantConfig.config = config
        ImportantConfig.Config = lambda: config

        from sql2fea import TreeBuilder, value_extractor, Sql2Vec
        from NET import TreeNet
        from TreeLSTM import SPINN
        from Hinter import Hinter
        from mcts import MCTSHinterSearch

        self.tree_builder = TreeBuilder()
        self.sql2vec = Sql2Vec()
        self.value_network = SPINN(
            head_num=config.head_num, input_size=7 + 2,
            hidden_size=config.hidden_size, table_num=50,
            sql_size=40 * 40 + config.max_column,
        ).to(config.device)

        import torch.nn as init_mod
        for name, param in self.value_network.named_parameters():
            if len(param.shape) == 2:
                init_mod.init.xavier_normal(param)
            else:
                init_mod.init.uniform(param)

        self.net = TreeNet(tree_builder=self.tree_builder, value_network=self.value_network)
        self.mcts_searcher = MCTSHinterSearch()

        self.hinter = Hinter(
            model=self.net, optimize_only=False, sql2vec=self.sql2vec,
            value_extractor=value_extractor, mcts_searcher=self.mcts_searcher,
        )

        self._restore_state_from_disk()
        self.loaded = True
        self._last_mtime = self._model_mtime()

    def _restore_state_from_disk(self):
        """In-place load of model weights + KNN + replay + alias map + column_id
        into the existing objects. Used by `_load` (initial hydrate / model_dir
        change, under _load_lock or at first request). NOT safe for concurrent
        readers (use _maybe_reload_model for that)."""
        config = self.config
        (state_file, column_id_file, alias_map_file, value_net, value_net_opt,
         mcts_opt, mcts_model) = self._paths()

        if os.path.exists(value_net):
            try:
                self.value_network.load_state_dict(torch.load(value_net, map_location=config.device))
            except Exception:
                pass
        if os.path.exists(value_net_opt):
            try:
                self.net.optimizer.load_state_dict(torch.load(value_net_opt, map_location=config.device))
            except Exception:
                pass
        if os.path.exists(mcts_model):
            try:
                import mcts as mcts_mod
                mcts_mod.predictionNet.load_state_dict(torch.load(mcts_model, map_location=config.device))
                mcts_mod.predictionNet.to(config.cpudevice)
            except Exception:
                pass
        if os.path.exists(mcts_opt):
            try:
                import mcts as mcts_mod
                mcts_mod.optimizer.load_state_dict(torch.load(mcts_opt, map_location=config.device))
            except Exception:
                pass

        import sql2fea as sql2fea_mod
        if os.path.exists(column_id_file):
            try:
                with open(column_id_file, "rb") as f:
                    sql2fea_mod.column_id = pickle.load(f)
            except Exception:
                pass
        if os.path.exists(alias_map_file):
            try:
                with open(alias_map_file, "rb") as f:
                    persisted = pickle.load(f)
                config.id2aliasname = persisted.get("id2aliasname", config.id2aliasname)
                config.aliasname2id = persisted.get("aliasname2id", config.aliasname2id)
            except Exception:
                pass

        if os.path.exists(state_file):
            try:
                with open(state_file, "rb") as f:
                    saved = pickle.load(f)
                self.hinter.knn = saved.get("knn", self.hinter.knn)
                self.hinter.hinter_times = saved.get("hinter_times", self.hinter.hinter_times)
                self.trained_samples = saved.get("trained_samples", self.trained_samples)
                self.net.memory.memory = saved.get("net_memory", self.net.memory.memory)
                self.net.memory.position = saved.get("net_memory_pos", self.net.memory.position)
                self.mcts_searcher.memory.memory = saved.get("mcts_memory", self.mcts_searcher.memory.memory)
                self.mcts_searcher.memory.position = saved.get("mcts_memory_pos", self.mcts_searcher.memory.position)
            except Exception as e:
                print(f"state restore failed: {e}", file=sys.stderr)

    def _model_mtime(self):
        """Max mtime across all persisted files; 0 if none exist yet."""
        try:
            mt = 0
            for p in self._paths():
                if os.path.exists(p):
                    mt = max(mt, os.path.getmtime(p))
            return mt
        except Exception:
            return 0

    def _maybe_reload_model(self):
        """Lock-free foreground reload: if the training subprocess saved new
        state (mtime advanced), pick up the new weights via SNAPSHOT-SWAP —
        load into freshly-constructed modules, then atomically reassign the
        references. In-flight optimize calls keep reading the old object (still
        valid); new calls use the new one. No half-loaded state, no lock."""
        mt = self._model_mtime()
        if mt == self._last_mtime:
            return
        config = self.config
        (state_file, column_id_file, alias_map_file, value_net, value_net_opt,
         mcts_opt, mcts_model) = self._paths()
        import sql2fea as sql2fea_mod

        # value_network: fresh SPINN, load isolated, then atomic swap.
        if os.path.exists(value_net):
            try:
                from TreeLSTM import SPINN
                fresh = SPINN(
                    head_num=config.head_num, input_size=7 + 2,
                    hidden_size=config.hidden_size, table_num=50,
                    sql_size=40 * 40 + config.max_column,
                ).to(config.device)
                fresh.load_state_dict(torch.load(value_net, map_location=config.device))
                self.value_network = fresh
                self.net.value_network = fresh  # self.hinter.model is self.net
            except Exception as e:
                print(f"value_network reload failed: {e}", file=sys.stderr)
        # mcts.predictionNet: fresh ValueNet, swap module global.
        if os.path.exists(mcts_model):
            try:
                import mcts as mcts_mod
                from NET import ValueNet
                fresh_m = ValueNet(config.mcts_input_size).to(config.cpudevice)
                fresh_m.load_state_dict(torch.load(mcts_model, map_location=config.cpudevice))
                mcts_mod.predictionNet = fresh_m
            except Exception as e:
                print(f"predictionNet reload failed: {e}", file=sys.stderr)
        # knn / replay / counters: object reassign (readers get old-or-new, both valid).
        if os.path.exists(state_file):
            try:
                with open(state_file, "rb") as f:
                    saved = pickle.load(f)
                self.hinter.knn = saved.get("knn", self.hinter.knn)
                # hinter_times is not reloaded on the foreground path (the
                # foreground doesn't use it). trained_samples is a training-side
                # counter, synced via reload.
                self.trained_samples = saved.get("trained_samples", self.trained_samples)
                self.net.memory.memory = saved.get("net_memory", self.net.memory.memory)
                self.net.memory.position = saved.get("net_memory_pos", self.net.memory.position)
                self.mcts_searcher.memory.memory = saved.get("mcts_memory", self.mcts_searcher.memory.memory)
                self.mcts_searcher.memory.position = saved.get("mcts_memory_pos", self.mcts_searcher.memory.position)
            except Exception as e:
                print(f"state reload failed: {e}", file=sys.stderr)
        # column_id: replace the module-global dict under its lock.
        if os.path.exists(column_id_file):
            try:
                with open(column_id_file, "rb") as f:
                    cid = pickle.load(f)
                with sql2fea_mod._column_id_lock:
                    sql2fea_mod.column_id = cid
            except Exception as e:
                print(f"column_id reload failed: {e}", file=sys.stderr)
        # alias map: reassign config attrs (atomic).
        if os.path.exists(alias_map_file):
            try:
                with open(alias_map_file, "rb") as f:
                    persisted = pickle.load(f)
                config.id2aliasname = persisted.get("id2aliasname", config.id2aliasname)
                config.aliasname2id = persisted.get("aliasname2id", config.aliasname2id)
            except Exception as e:
                print(f"alias_map reload failed: {e}", file=sys.stderr)
        self._last_mtime = mt

    def ensure_loaded(self, config_overrides):
        model_dir = config_overrides.get("model_dir") or MODEL_DIR
        self.max_kvs_size = int(config_overrides.get("max_kvs_size", MAX_KVS_SIZE_DEFAULT))
        self.train_trigger = int(config_overrides.get("train_trigger", TRAIN_TRIGGER_DEFAULT))
        if not self.loaded or model_dir != self.model_dir:
            with self._load_lock:
                if not self.loaded or model_dir != self.model_dir:
                    self.model_dir = model_dir
                    os.makedirs(self.model_dir, exist_ok=True)
                    self._load()

    def optimize(self, dsn, query, optimize_only, config_overrides, inspect_sql):
        # Lock-free critical path. _maybe_reload_model is a snapshot-swap (no
        # blocking); hinterRun(optimize_only=True) is read-only w.r.t. the net
        # and uses per-call local bookkeeping; the thread-local pgrunner proxy
        # gives this thread its own cursor.
        self._maybe_reload_model()
        import PGUtils
        import sql2fea as sql2fea_mod

        database, user, password, host, port = parse_dsn(dsn)
        self.config.database = database
        self.config.user = user
        self.config.password = password
        self.config.ip = host
        self.config.port = port
        # The thread-local pgrunner proxy connects from `config` on first use
        # in this thread → own connection + cursor.

        query_aliases = extract_aliases_from_query(query)
        id2aliasname, aliasname2id = build_dynamic_alias_map(
            self.config.id2aliasname, self.config.aliasname2id,
            query_aliases, config_overrides,
        )
        self.config.id2aliasname = id2aliasname
        self.config.aliasname2id = aliasname2id
        for k, v in config_overrides.items():
            if k == "aliasname2id":
                continue
            if hasattr(self.config, k):
                setattr(self.config, k, v)

        self.hinter.optimize_only = True  # foreground is ALWAYS cost-only: no
        # EXPLAIN ANALYZE, no execution, no training on the critical path. The
        # actual execution + training runs in a separate process (see
        # _execute_training / run_training) when this is a training-mode
        # request and enough pending queries accumulate.

        start_time = time.time()
        try:
            (pg_plan_time, pg_latency, mcts_time, hinter_plan_time, MPHE_time,
             hinter_latency, chosen_plan, hinter_time) = self.hinter.hinterRun(query)
            optimization_time = time.time() - start_time
            if chosen_plan and chosen_plan[0] != "PG":
                optimized_query = chosen_plan[0] + " " + query
            else:
                optimized_query = query
            # pg_latency / hinter_latency are COST estimates in both modes
            # (foreground never executes). Actual latencies are collected by
            # the training subprocess and are not reported per-call.
            if pg_latency > 0 and hinter_latency > 0:
                estimated_impact = max(0, (pg_latency - hinter_latency) / pg_latency * 100)
            else:
                estimated_impact = 0.0
            result = {
                "optimized_query": optimized_query,
                "metadata": {
                    "strategy_type": "hybrid-mcts-learning",
                    "optimization_time": round(optimization_time, 4),
                    "estimated_impact": round(estimated_impact, 2),
                    "pg_latency_ms": round(pg_latency, 2),
                    "hinter_latency_ms": round(hinter_latency, 2),
                    "chosen_plan": chosen_plan,
                },
            }
        except Exception as e:
            result = {
                "optimized_query": query,
                "metadata": {
                    "strategy_type": "hybrid-mcts-learning",
                    "optimization_time": round(time.time() - start_time, 4),
                    "estimated_impact": 0.0,
                    "error": str(e),
                },
            }

        if inspect_sql:
            result["sql_log"] = PGUtils.sql_log[:]

        # Training-mode: append the query to the pending queue so the training
        # subprocess will re-execute it (EXPLAIN ANALYZE) + train. Optimize-only
        # enqueues nothing (no execution). The safe-gate skips enqueueing when
        # the chosen plan is catastrophically costlier than PG's default (would
        # hang the subprocess on a doomed execution). Spawning (and background
        # catch-up) is handled by _try_start_training.
        if not optimize_only and not _cost_ratio_too_high(hinter_latency, pg_latency):
            self._append_pending(query, dsn, sql2fea_mod)
            self._try_start_training()
        return result

    # -- Pending training queue + subprocess spawn --
    def _append_pending(self, query, dsn, sql2fea_mod):
        """Persist lightweight state (alias_map + column_id) so the training
        subprocess trains with ids consistent with prediction, and append the
        query to the pending file. Spawning is decided by _try_start_training
        (called from the foreground and the background drainer). column_id is
        read under its lock so the pickle isn't torn by a concurrent
        getColumnId."""
        self._persist_lightweight(sql2fea_mod)
        pending = self._pending_file()
        try:
            with open(pending, "a") as f:
                f.write(json.dumps({"query": query, "dsn": dsn}) + "\n")
        except Exception as e:
            print(f"pending append failed: {e}", file=sys.stderr)

    def _try_start_training(self):
        """Atomically check the spawn condition and start a training
        subprocess if pending >= train_trigger and none is in progress. Safe
        to call from both the foreground /optimize path and the background
        drainer — the check-and-set is under _state_lock, so at most one
        subprocess runs at a time (a losing caller's subprocess no-ops via
        _claim_pending's rename)."""
        if not self.loaded or self._shutting_down:
            return False
        try:
            with open(self._pending_file()) as f:
                n = sum(1 for line in f if line.strip())
        except (FileNotFoundError, OSError):
            n = 0
        with self._state_lock:
            if self._training_in_progress or n < self.train_trigger:
                return False
            self._training_in_progress = True
        self._spawn_training_worker()
        return True

    def start_drainer(self):
        """Start the background catch-up drainer. Called once from the server
        main() only — never from a `wrapper.py --train` subprocess. A --train
        subprocess that started its own drainer would spawn more --train
        children: the _training_in_progress flag is in-process memory and does
        not cross process boundaries, so a subprocess's drainer can't see the
        foreground's flag, forks uncontrollably, and concurrent trainers
        clobber each other's state writes (trained_samples/knn_len regress)."""
        if self._drain_thread is None:
            self._drain_thread = threading.Thread(target=self._drain_loop, daemon=True)
            self._drain_thread.start()

    def _drain_loop(self):
        """Background catch-up: keep starting training subprocesses while
        pending >= train_trigger, independent of incoming /optimize calls.
        Drains the pending queue even after the client stops sending queries
        (the case where pending balloons while the slow training subprocess is
        busy, then the client disconnects and nothing re-spawns). No-op while
        not loaded / shutting down."""
        while not self._shutting_down:
            try:
                self._try_start_training()
            except Exception as e:
                print(f"drain loop error: {e}", file=sys.stderr)
            with self._state_lock:
                in_progress = self._training_in_progress
            # Poll quickly while training is in progress (chain the next batch
            # as soon as it finishes); back off when idle.
            time.sleep(2 if in_progress else 5)

    def _persist_lightweight(self, sql2fea_mod):
        """Atomically persist only the alias map + column_id (cheap, training-
        mode per query) so the training subprocess loads the foreground's
        latest alias/column ids and trains with ids consistent with prediction."""
        alias_map_file = os.path.join(self.state_dir, "alias_map.pkl")
        column_id_file = os.path.join(self.state_dir, "column_id.pkl")
        try:
            alias_bytes = pickle.dumps({
                "id2aliasname": self.config.id2aliasname,
                "aliasname2id": self.config.aliasname2id,
            })
            _atomic_write_bytes(alias_map_file, alias_bytes)
        except Exception:
            pass
        try:
            with sql2fea_mod._column_id_lock:
                column_bytes = pickle.dumps(sql2fea_mod.column_id)
            _atomic_write_bytes(column_id_file, column_bytes)
        except Exception:
            pass

    def _claim_pending(self):
        """Atomically rename the pending file to a pid-suffixed processing file
        and return (queries, processing_path). New appends go to a fresh
        pending file, so nothing is lost while the subprocess trains. Returns
        ([], None) if there is nothing pending."""
        pending = self._pending_file()
        processing = os.path.join(
            self.state_dir, f"pending_queries.processing.{os.getpid()}.jsonl")
        try:
            os.rename(pending, processing)
        except (OSError, FileNotFoundError):
            return [], None
        queries = []
        try:
            with open(processing) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        queries.append(json.loads(line))
                    except Exception:
                        pass
        except Exception:
            pass
        return queries, processing

    def _execute_training(self, sql, dsn):
        """Re-run hinterRun with optimize_only=False to collect EXPLAIN ANALYZE
        latencies and train the value/MCTS networks. Runs in the training
        subprocess (single-threaded): no lock needed. The MCTS search is
        cost-only and deterministic given the current net state, so it picks
        the same plan the foreground already returned; execution + training
        happen here."""
        import PGUtils
        database, user, password, host, port = parse_dsn(dsn)
        self.config.database = database
        self.config.user = user
        self.config.password = password
        self.config.ip = host
        self.config.port = port
        # Re-merge aliases defensively (the foreground persisted them already,
        # but this keeps ids consistent if a query alias was missed on disk).
        query_aliases = extract_aliases_from_query(sql)
        id2aliasname, aliasname2id = build_dynamic_alias_map(
            self.config.id2aliasname, self.config.aliasname2id, query_aliases, None)
        self.config.id2aliasname = id2aliasname
        self.config.aliasname2id = aliasname2id
        self.hinter.optimize_only = False
        try:
            with redirect_stdout(io.StringIO()):
                self.hinter.hinterRun(sql)
        except Exception as e:
            print(f"training hinterRun failed: {e}", file=sys.stderr)

    def _spawn_training_worker(self):
        """Spawn `wrapper.py --train --model-dir <dir>` in a child process and
        wait for it. The subprocess loads state from disk, processes all
        pending queries, and saves atomically; the foreground picks up the new
        weights via _maybe_reload_model on the next request. Runs on a daemon
        thread so the foreground /optimize returns immediately."""
        model_dir = self.model_dir

        def _run():
            try:
                p = subprocess.Popen(
                    [sys.executable, os.path.abspath(__file__),
                     "--train", "--model-dir", model_dir]
                )
                with self._state_lock:
                    self._train_proc = p
                p.wait()
            except Exception as e:
                print(f"training worker failed: {e}", file=sys.stderr)
            finally:
                with self._state_lock:
                    self._training_in_progress = False
                    self._train_proc = None

        threading.Thread(target=_run, daemon=True).start()

    def _wait_for_training(self, timeout=300):
        """Block until the in-flight training subprocess finishes (its model
        save completes) or the timeout expires."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._state_lock:
                in_progress = self._training_in_progress
            if not in_progress:
                return
            time.sleep(0.5)

    def _persist(self, sql2fea_mod):
        """Full snapshot + atomic write of all state. Called from the training
        subprocess (single-threaded) and from persist()'s final-training
        subprocess — never from a context with concurrent readers, so no lock."""
        (state_file, column_id_file, alias_map_file, value_net, value_net_opt,
         mcts_opt, mcts_model) = self._paths()

        try:
            kvs = self.hinter.knn.kvs
            if len(kvs) > self.max_kvs_size:
                self.hinter.knn.kvs = kvs[-self.max_kvs_size:]
        except Exception:
            pass
        state_bytes = pickle.dumps({
            "knn": self.hinter.knn,
            "hinter_times": self.hinter.hinter_times,
            "trained_samples": self.trained_samples,
            "net_memory": self.net.memory.memory,
            "net_memory_pos": self.net.memory.position,
            "mcts_memory": self.mcts_searcher.memory.memory,
            "mcts_memory_pos": self.mcts_searcher.memory.position,
        })
        with sql2fea_mod._column_id_lock:
            column_bytes = pickle.dumps(sql2fea_mod.column_id)
        alias_bytes = pickle.dumps({
            "id2aliasname": self.config.id2aliasname,
            "aliasname2id": self.config.aliasname2id,
        })
        # Clone each state_dict independently. Optimizer state_dicts are nested
        # (dict/list values, not tensors), so _clone_state_dict deep-copies
        # them; a failure on one must not discard the others.
        try:
            vw_sd = _clone_state_dict(self.value_network.state_dict())
        except Exception:
            vw_sd = None
        try:
            opt_sd = _clone_state_dict(self.net.optimizer.state_dict())
        except Exception:
            opt_sd = None
        try:
            import mcts as mcts_mod
            mo_sd = _clone_state_dict(mcts_mod.optimizer.state_dict())
        except Exception:
            mo_sd = None
        try:
            import mcts as mcts_mod
            mm_sd = _clone_state_dict(mcts_mod.predictionNet.state_dict())
        except Exception:
            mm_sd = None

        _atomic_write_bytes(state_file, state_bytes)
        _atomic_write_bytes(column_id_file, column_bytes)
        _atomic_write_bytes(alias_map_file, alias_bytes)
        if vw_sd is not None:
            _atomic_torch_save(vw_sd, value_net)
        if opt_sd is not None:
            _atomic_torch_save(opt_sd, value_net_opt)
        if mo_sd is not None:
            _atomic_torch_save(mo_sd, mcts_opt)
        if mm_sd is not None:
            _atomic_torch_save(mm_sd, mcts_model)

    def persist(self):
        """Shutdown persist: single-flight (concurrent /shutdown or SIGTERM
        racing a skill_runner stop_wrapper runs this once). Wait for any
        in-flight training subprocess so its save isn't abandoned, then if
        pending queries remain (collected since the last spawn), run a FINAL
        training subprocess and wait for it. The final training is a subprocess
        (not in-process) so it never mutates the foreground net while in-flight
        optimize calls (past the 503 check) finish. SERVER.shutdown is started
        in the finally so only the lock-owner shuts the server down."""
        if not self._persist_lock.acquire(blocking=False):
            return
        try:
            self._shutting_down = True
            self._wait_for_training(timeout=300)
            if not self.loaded:
                return
            # If pending queries remain, run a final training subprocess.
            pending = self._pending_file()
            n = 0
            try:
                with open(pending) as f:
                    n = sum(1 for line in f if line.strip())
            except FileNotFoundError:
                n = 0
            if n > 0:
                try:
                    p = subprocess.Popen(
                        [sys.executable, os.path.abspath(__file__),
                         "--train", "--model-dir", self.model_dir]
                    )
                    p.wait(timeout=600)
                except subprocess.TimeoutExpired:
                    try:
                        p.kill()
                    except Exception:
                        pass
                    print("final training subprocess timed out", file=sys.stderr)
                except Exception as e:
                    print(f"final training subprocess failed: {e}", file=sys.stderr)
        finally:
            self._persist_lock.release()
            # Only the persist that acquired the lock shuts down the server —
            # a guarded-out persist (concurrent /shutdown) must NOT start
            # SERVER.shutdown, or daemon_threads=True would let main() exit
            # and kill this in-flight save.
            if SERVER is not None:
                threading.Thread(target=SERVER.shutdown, daemon=True).start()

    def state_summary(self):
        if not self.loaded:
            return {"loaded": False}
        knn_len = 0
        try:
            knn_len = len(self.hinter.knn.kvs)
        except Exception:
            pass
        summary = {
            "loaded": True,
            "knn_len": knn_len,
            "trained_samples": self.trained_samples,
            "net_memory_len": len(self.net.memory.memory),
            "mcts_memory_len": len(self.mcts_searcher.memory.memory),
            "alias_count": len(self.config.aliasname2id),
            "model_dir": self.model_dir,
            "train_trigger": self.train_trigger,
        }
        try:
            with open(self._pending_file()) as f:
                pending_count = sum(1 for line in f if line.strip())
        except (FileNotFoundError, Exception):
            pending_count = 0
        with self._state_lock:
            summary["pending_count"] = pending_count
            # `idle`: True only when, with no new prompts arriving, this skill
            # does no background work — no training subprocess and pending queue
            # below train_trigger (so the drainer won't spawn). Collection runs
            # only inside the training subprocess, so no in-flight check applies.
            summary["idle"] = (not self._training_in_progress
                               and pending_count < self.train_trigger)
        return summary


# --------------------------------------------------------------------------- #
# Training subprocess entry: load state from disk, process all pending
# training queries, save atomically. Invoked as `wrapper.py --train`.
# --------------------------------------------------------------------------- #
def run_training(model_dir):
    os.makedirs(model_dir, exist_ok=True)
    runner = HyperQOSkill()
    runner.model_dir = model_dir
    runner._load()  # fresh Hinter + net + mcts, state hydrated from disk
    queries, processing = runner._claim_pending()
    if not queries:
        if processing is not None:
            try:
                os.remove(processing)
            except OSError:
                pass
        return 0
    for q in queries:
        try:
            runner._execute_training(q["query"], q["dsn"])
            runner.trained_samples += 1
        except Exception as e:
            print(f"training query failed: {e}", file=sys.stderr)
    try:
        import sql2fea as sql2fea_mod
        runner._persist(sql2fea_mod)
    except Exception as e:
        print(f"training persist failed: {e}", file=sys.stderr)
    try:
        if processing is not None:
            os.remove(processing)
    except OSError:
        pass
    return 0


# --------------------------------------------------------------------------- #
# HTTP server.
# --------------------------------------------------------------------------- #
SERVER = None
SKILL = HyperQOSkill()


def _drain_and_stop():
    try:
        SKILL.persist()  # persist() starts SERVER.shutdown in its finally (only if it owns the lock)
    except Exception as e:
        print(f"persist failed: {e}", file=sys.stderr)
        if SERVER is not None:
            threading.Thread(target=SERVER.shutdown, daemon=True).start()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send_json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send_json(200, {"status": "ok"})
        elif self.path.startswith("/state"):
            self._send_json(200, SKILL.state_summary())
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/optimize":
            if SKILL._shutting_down:
                self._send_json(503, {"error": "server is shutting down"})
                return
            try:
                n = int(self.headers.get("Content-Length", "0") or "0")
                raw = self.rfile.read(n).decode() if n else "{}"
                req = json.loads(raw) if raw else {}
            except Exception as e:
                self._send_json(400, {"error": f"bad request body: {e}"})
                return
            dsn = req.get("dsn")
            query = req.get("query")
            optimize_only = bool(req.get("optimize_only"))
            inspect_sql = bool(req.get("inspect_sql"))
            cfg = req.get("config")
            if isinstance(cfg, str) and cfg:
                try:
                    cfg = json.loads(cfg)
                except Exception:
                    cfg = {"model_dir": cfg}
            cfg = cfg or {}
            if not isinstance(cfg, dict):
                self._send_json(400, {"error": "config must be a JSON object"})
                return
            if not dsn or not query:
                self._send_json(400, {"error": "dsn and query are required"})
                return
            try:
                SKILL.ensure_loaded(cfg)
                result = SKILL.optimize(dsn, query, optimize_only, cfg, inspect_sql)
            except Exception as e:
                self._send_json(500, {"error": str(e)})
            else:
                self._send_json(200, result)
        elif self.path == "/shutdown":
            self._send_json(200, {"status": "shutting down"})
            _drain_and_stop()
        else:
            self._send_json(404, {"error": "not found"})


def _signal_handler(signum, frame):
    _drain_and_stop()


def main():
    parser = argparse.ArgumentParser(description="HyperQO Query Optimizer server")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "0")))
    parser.add_argument("--train", action="store_true",
                        help="Run one training cycle on pending queries and exit (subprocess mode).")
    parser.add_argument("--model-dir", default=None,
                        help="Model directory (for --train mode).")
    args = parser.parse_args()

    if args.train:
        sys.exit(run_training(args.model_dir or MODEL_DIR))

    if not args.port:
        print("error: --port or PORT env required", file=sys.stderr)
        sys.exit(1)

    global SERVER
    SERVER = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    SKILL.start_drainer()
    print(f"HyperQO skill server listening on 127.0.0.1:{args.port}", flush=True)
    SERVER.serve_forever()


if __name__ == "__main__":
    main()
