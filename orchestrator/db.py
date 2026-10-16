"""Compact program database: OpenEvolve v0.4.0's storage and sampling (openevolve/database.py at a1e9d91),
reimplemented for this project with OpenEvolve's default parameters and 3 islands.

  storage     every evaluated child is kept (invalid ones with score 0); the global best is tracked separately
  islands     a child goes to the island it was sampled for; an island also keeps programs that lost their
              MAP-Elites cell competition (OpenEvolve adds every child to the island set)
  MAP-Elites  one grid per island, (complexity, diversity) x 10 bins, both min-max scaled over the values seen
              so far; the higher score keeps a cell; a displaced program leaves the island (and the database,
              unless it is the global best)
  archive     global elite set of 20 (MAP-Elites replacements swap archive membership, as in OpenEvolve)
  sampling    parent: exploration 0.2 (uniform over the island) / exploitation 0.7 (archive members of the island,
              else the whole archive) / weighted 0.1 (fitness-proportional within the island);
              inspirations: island best, island top max(1, int(0.1 n)), nearby MAP-Elites cells, random fill
  migration   ring topology, top 10% of each island copied to both neighbours when
              max(generation) - last migration >= 10; copies are never migrated again

Determinism: one seeded random.Random; programs, islands and the archive are insertion-ordered.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

FEATURES = ("complexity", "diversity")


@dataclass
class Program:
    id: str
    code: str
    score: float
    artifacts: dict = field(default_factory=dict)
    parent_id: str | None = None
    island: int = 0
    iteration: int = 0
    generation: int = 0
    feature_coords: list = field(default_factory=list)
    status: str = "ok"                 # evaluation status (ok | compile_error | invalid | timeout | runtime_error)
    changes: str = ""                  # diff summary shown as "Changes" in later prompts
    parent_score: float | None = None
    migrant: bool = False
    code_hash: str = ""
    eval_hash: str | None = None       # solutions/{eval_hash}.json holds the full evaluation


def fast_code_diversity(a: str, b: str) -> float:
    """OpenEvolve's _fast_code_diversity."""
    if a == b:
        return 0.0
    return (abs(len(a) - len(b)) * 0.1 + abs(a.count("\n") - b.count("\n")) * 10
            + len(set(a) ^ set(b)) * 0.5)


class ProgramDB:
    def __init__(self, seed: int, num_islands: int = 3, feature_bins: int = 10, archive_size: int = 20,
                 exploration_ratio: float = 0.2, exploitation_ratio: float = 0.7, elite_selection_ratio: float = 0.1,
                 migration_interval: int = 10, migration_rate: float = 0.1, diversity_reference_size: int = 20,
                 num_inspirations: int = 2):
        self.p = dict(num_islands=num_islands, feature_bins=feature_bins, archive_size=archive_size,
                      exploration_ratio=exploration_ratio, exploitation_ratio=exploitation_ratio,
                      elite_selection_ratio=elite_selection_ratio, migration_interval=migration_interval,
                      migration_rate=migration_rate, diversity_reference_size=diversity_reference_size,
                      num_inspirations=num_inspirations)
        self.n = num_islands
        # OpenEvolve: bins = max(feature_bins, ceil-ish(archive_size ** (1 / n_features)))
        self.bins = max(feature_bins, int(pow(archive_size, 1 / len(FEATURES)) + 0.99))
        self.rng = random.Random(seed)
        self.programs: dict[str, Program] = {}
        self.islands: list[dict[str, None]] = [{} for _ in range(num_islands)]
        self.grids: list[dict[str, str]] = [{} for _ in range(num_islands)]
        self.archive: dict[str, None] = {}
        self.best_id: str | None = None
        self.generations = [0] * num_islands
        self.last_migration = 0
        self.n_migrations = 0
        self.n_migrant_copies = 0
        self.stats: dict[str, list[float]] = {}     # feature -> [min, max]
        self.div_cache: dict[str, float] = {}
        self.ref_set: list[str] = []
        self.next_id = 0

    # ------------------------------------------------------------------ features
    def new_id(self) -> str:
        self.next_id += 1
        return f"p{self.next_id:04d}"

    def _bin(self, feat: str, v: float) -> int:
        st = self.stats.setdefault(feat, [v, v])
        st[0], st[1] = min(st[0], v), max(st[1], v)
        x = 0.5 if st[1] == st[0] else min(1.0, max(0.0, (v - st[0]) / (st[1] - st[0])))
        return max(0, min(self.bins - 1, int(x * self.bins)))

    def _diversity(self, code: str) -> float:
        if code in self.div_cache:
            return self.div_cache[code]
        k = self.p["diversity_reference_size"]
        if len(self.ref_set) < k:  # OpenEvolve rebuilds the reference set until it holds k programs
            progs = list(self.programs.values())
            if len(progs) <= k:
                self.ref_set = [q.code for q in progs]
            else:  # greedy max-min selection from a random start
                sel = [progs.pop(self.rng.randint(0, len(progs) - 1))]
                while len(sel) < k and progs:
                    i = max(range(len(progs)), key=lambda j: min(fast_code_diversity(progs[j].code, s.code)
                                                                   for s in sel))
                    sel.append(progs.pop(i))
                self.ref_set = [q.code for q in sel]
        ds = [fast_code_diversity(code, r) for r in self.ref_set if r != code]
        self.div_cache[code] = sum(ds) / len(ds) if ds else 0.0
        return self.div_cache[code]

    def coords(self, prog: Program) -> list[int]:
        c = self._bin("complexity", float(len(prog.code)))
        d = 0 if len(self.programs) < 2 else self._bin("diversity", self._diversity(prog.code))
        return [c, d]

    @staticmethod
    def _key(coords) -> str:
        return "-".join(map(str, coords))

    # ------------------------------------------------------------------ add
    def add(self, prog: Program, island: int) -> None:
        island %= self.n
        self.programs[prog.id] = prog
        prog.feature_coords = self.coords(prog)
        key, grid = self._key(prog.feature_coords), self.grids[island]
        cur = grid.get(key)
        replaced = None
        if cur is None or cur not in self.programs or prog.score > self.programs[cur].score:
            if cur is not None:
                if cur in self.programs and cur in self.archive:
                    self.archive.pop(cur)
                    self.archive[prog.id] = None
                self.islands[island].pop(cur, None)
                replaced = cur
            grid[key] = prog.id
        self.islands[island][prog.id] = None
        prog.island = island
        self._update_archive(prog)
        if self.best_id is None or self.best_id not in self.programs or prog.score > self.best.score:
            self.best_id = prog.id
        if replaced is not None and replaced != prog.id:
            self._remove_if_orphaned(replaced)

    def _update_archive(self, prog: Program) -> None:
        size = self.p["archive_size"]
        if len(self.archive) < size:
            self.archive[prog.id] = None
            return
        for pid in [pid for pid in self.archive if pid not in self.programs]:
            self.archive.pop(pid)
        if len(self.archive) < size:
            self.archive[prog.id] = None
            return
        worst = min((self.programs[pid] for pid in self.archive), key=lambda q: q.score)
        if prog.score > worst.score:
            self.archive.pop(worst.id)
            self.archive[prog.id] = None

    def _remove_if_orphaned(self, pid: str) -> None:
        if pid not in self.programs or pid == self.best_id:
            return
        if any(pid in g.values() for g in self.grids) or any(pid in isl for isl in self.islands):
            return
        del self.programs[pid]
        self.archive.pop(pid, None)

    # ------------------------------------------------------------------ queries
    @property
    def best(self) -> Program:
        return self.programs[self.best_id]

    def members(self, island: int) -> list[Program]:
        return [self.programs[pid] for pid in self.islands[island % self.n] if pid in self.programs]

    def top(self, island: int, k: int) -> list[Program]:
        return sorted(self.members(island), key=lambda q: -q.score)[:k]   # stable: ties by insertion order

    def island_best(self, island: int) -> Program | None:
        ms = self.members(island)
        return max(ms, key=lambda q: q.score) if ms else None            # first max = earliest added

    # ------------------------------------------------------------------ sampling
    def _weighted(self, island: int) -> Program:
        ms = self.members(island)
        if not ms:
            return self.rng.choice(list(self.programs.values()))
        if len(ms) == 1:
            return ms[0]
        return self.rng.choices(ms, weights=[max(q.score, 0.001) for q in ms], k=1)[0]

    def _from_archive(self, island: int) -> Program:
        valid = [self.programs[pid] for pid in self.archive if pid in self.programs]
        if not valid:
            return self._weighted(island)
        here = [q for q in valid if q.island == island]
        return self.rng.choice(here or valid)

    def sample(self, island: int) -> tuple[Program, list[Program], str]:
        """Parent + inspirations for `island`; returns the sampling mode as well."""
        island %= self.n
        r = self.rng.random()
        expl, expt = self.p["exploration_ratio"], self.p["exploitation_ratio"]
        mode = "exploration" if r < expl else "exploitation" if r < expl + expt else "weighted"
        ms = self.members(island)
        if not ms:
            # OpenEvolve falls back to sample(): same draw, on its current island (0), "random" over all programs
            if mode == "exploration":
                parent = self.rng.choice(self.members(0) or list(self.programs.values()))
            elif mode == "exploitation":
                parent = self._from_archive(0)
            else:
                parent = self.rng.choice(list(self.programs.values()))
            return parent, self._inspirations(parent, parent.island), mode + "(empty_island)"
        if mode == "exploration":
            parent = self.rng.choice(ms)
        elif mode == "exploitation":
            parent = self._from_archive(island)
        else:
            parent = self._weighted(island)
        return parent, self._inspirations(parent, island), mode

    def _inspirations(self, parent: Program, island: int) -> list[Program]:
        n = self.p["num_inspirations"]
        ids = [q.id for q in self.members(island)]
        if not ids:
            return []
        out: list[str] = []
        ib = self.island_best(island)
        if ib is not None and ib.id != parent.id:
            out.append(ib.id)
        for q in self.top(island, max(1, int(n * self.p["elite_selection_ratio"]))):
            if q.id not in out and q.id != parent.id:
                out.append(q.id)
        if len(out) < n:
            remaining = n - len(out)
            pc = self.coords(parent)
            cell = {self._key(self.coords(self.programs[pid])): pid for pid in ids}
            nearby: list[str] = []
            for _ in range(remaining * 3):
                k = self._key([max(0, min(self.bins - 1, c + self.rng.randint(-2, 2))) for c in pc])
                pid = cell.get(k)
                if pid is not None and pid != parent.id and pid not in out and pid not in nearby:
                    nearby.append(pid)
                    if len(nearby) >= remaining:
                        break
            if len(out) + len(nearby) < n:
                skip = {parent.id, *out, *nearby}
                avail = [pid for pid in ids if pid not in skip]
                nearby += self.rng.sample(avail, min(n - len(out) - len(nearby), len(avail)))
            out += nearby
        return [self.programs[pid] for pid in out[:n]]

    # ------------------------------------------------------------------ generations / migration
    def tick(self, island: int) -> bool:
        """Count a child added to `island`; migrate if due. Returns True if a migration happened."""
        self.generations[island % self.n] += 1
        if max(self.generations) - self.last_migration < self.p["migration_interval"]:
            return False
        self.migrate()
        return True

    def migrate(self) -> None:
        for i in range(self.n):
            ms = sorted(self.members(i), key=lambda q: -q.score)
            if not ms:
                continue
            for m in ms[:max(1, int(len(ms) * self.p["migration_rate"]))]:
                for tgt in ((i + 1) % self.n, (i - 1) % self.n):
                    self._migrate_one(m.id, tgt)
        self.last_migration = max(self.generations)
        self.n_migrations += 1

    def _migrate_one(self, pid: str, tgt: int) -> None:
        m = self.programs.get(pid)
        if m is None or m.migrant or any(q.code == m.code for q in self.members(tgt)):
            return
        copy = Program(**{**asdict(m), "id": self.new_id(), "parent_id": m.id, "island": tgt, "migrant": True,
                          "feature_coords": []})
        self.add(copy, tgt)
        self.n_migrant_copies += 1

    # ------------------------------------------------------------------ reporting / persistence
    def health(self) -> dict:
        return {"programs": len(self.programs), "programs_per_island": [len(self.members(i)) for i in range(self.n)],
                "cells_per_island": [len(g) for g in self.grids], "archive": len(self.archive),
                "migrations": self.n_migrations, "migrant_copies": self.n_migrant_copies,
                "generations": list(self.generations), "best_id": self.best_id,
                "best_score": self.best.score if self.best_id else None}

    def to_dict(self, with_code: bool = True) -> dict:
        progs = {pid: {k: v for k, v in asdict(q).items() if with_code or k != "code"}
                 for pid, q in self.programs.items()}
        st = self.rng.getstate()
        return {"params": self.p, "programs": progs, "islands": [list(i) for i in self.islands],
                "grids": self.grids, "archive": list(self.archive), "best_id": self.best_id,
                "generations": self.generations, "last_migration": self.last_migration,
                "n_migrations": self.n_migrations, "n_migrant_copies": self.n_migrant_copies,
                "stats": self.stats, "ref_set": self.ref_set if with_code else len(self.ref_set),
                "div_cache": self.div_cache if with_code else len(self.div_cache), "next_id": self.next_id,
                "rng": [st[0], list(st[1]), st[2]]}

    @classmethod
    def from_dict(cls, d: dict, seed: int) -> "ProgramDB":
        db = cls(seed, **d["params"])
        db.programs = {pid: Program(**q) for pid, q in d["programs"].items()}
        db.islands = [dict.fromkeys(i) for i in d["islands"]]
        db.grids, db.archive, db.best_id = d["grids"], dict.fromkeys(d["archive"]), d["best_id"]
        db.generations, db.last_migration = d["generations"], d["last_migration"]
        db.n_migrations, db.n_migrant_copies = d["n_migrations"], d["n_migrant_copies"]
        db.stats, db.ref_set, db.div_cache, db.next_id = d["stats"], d["ref_set"], d["div_cache"], d["next_id"]
        db.rng.setstate((d["rng"][0], tuple(d["rng"][1]), d["rng"][2]))
        return db
