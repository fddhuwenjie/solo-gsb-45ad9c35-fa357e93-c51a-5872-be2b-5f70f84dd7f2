"""删除影响预演（只读分析，不修改草稿）。

把用户勾选的存活条款（父条款按现有级联规则连同后代）加入删除集，比较删除
前后两类绑定：

1. 引用绑定：删除前可解析且目标存活、删除后目标落入删除集的引用（区间按
   展开后的单条目标逐个比较，仅报告「新增失效」部分）；
2. 定义术语绑定：删除前**唯一**绑定某定义的使用，删除后失去该唯一定义
   （未定义 / 无存活同名定义）、命中多个同名定义（歧义）或改绑其他定义。

直接受影响的存活条款为根，沿「反向引用」边做多源 BFS，得到每个受影响条款
到根因的最短原因链；等长链按文档顺序取字典序最小者；同一受影响条款（及其
重复影响）只列一次。链上任一边处于引用环上时，整条链合并标注为循环链。
"""
from __future__ import annotations

from collections import deque

from .parser import Clause
from .references import (
    _scan,
    _canonical_order,
    cascade_deleted,
    expand_range,
)
from .terms import review_terms

# 明细类型 -> 次级排序（同一偏移处引用排在术语前）
_DETAIL_ORDER = {
    "ref_broken": 0,
    "term_lost_unique": 1,
    "term_ambiguous": 1,
    "term_rebound": 1,
}


def _clause_brief(c: Clause | None) -> dict:
    if c is None:
        return {"id": "", "label": "", "heading": ""}
    return {"id": c.id, "label": c.label, "heading": c.heading or ""}


# ---------------------------------------------------------------- 术语绑定对照

def _use_index(review: dict) -> dict[str, dict]:
    """occ_id -> 使用标记（附 block_id，便于定位所在条款）。"""
    out: dict[str, dict] = {}
    for b in review["blocks"]:
        for m in b["marks"]:
            if m["kind"] == "use":
                out[m["occ_id"]] = dict(m, block_id=b["id"])
    return out


def _def_index(review: dict) -> dict[str, dict]:
    return {d["id"]: d for d in review["definitions"]}


def _term_details(preamble: str, clauses: list[Clause],
                  cur_del: set[str], new_del: set[str]
                  ) -> tuple[dict[str, list[dict]], list[dict]]:
    """对比删除前后的术语绑定。

    返回 (存活条款 id -> 直接影响明细, 前置说明的直接影响明细)。
    只统计删除前「唯一绑定」的使用；歧义/未定义不因删除而计入。
    """
    cur_rev = review_terms(preamble, clauses, cur_del)
    new_rev = review_terms(preamble, clauses, new_del)
    cur_uses, new_uses = _use_index(cur_rev), _use_index(new_rev)
    cur_defs, new_defs = _def_index(cur_rev), _def_index(new_rev)

    by_clause: dict[str, list[dict]] = {}
    preamble_details: list[dict] = []

    def add(owner: str, d: dict) -> None:
        (preamble_details if owner == "preamble"
         else by_clause.setdefault(owner, [])).append(d)

    for uid, um in cur_uses.items():
        if um["status"] != "ok" or len(um.get("target_ids", [])) != 1:
            continue
        owner = um["block_id"]
        if owner != "preamble" and owner in new_del:
            continue  # 使用处自身将被删除，无从失效
        old_did = um["target_ids"][0]
        old_def = cur_defs.get(old_did, {})
        post = new_uses.get(uid)

        kind: str | None = None
        if post is None or post["status"] == "undefined":
            kind = "term_lost_unique"
        elif post["status"] == "ambiguous":
            kind = "term_ambiguous"
        elif (post["status"] == "ok" and post.get("target_ids")
              and post["target_ids"][0] != old_did):
            kind = "term_rebound"
        if kind is None:
            continue

        name = um["name"]
        old_where = old_def.get("scope_label", "")
        old_label = f"{old_where}局部定义" if old_def.get("scope") == "local" \
            else "全文定义"
        if kind == "term_lost_unique":
            message = (
                f"术语“{name}”失去唯一定义：其绑定的{old_where}定义"
                f"（{old_def.get('label', '')}）将随删除移除，本处变为未定义"
            )
        elif kind == "term_ambiguous":
            cand = [new_defs[i]["label"] for i in post["target_ids"]
                    if i in new_defs]
            message = (
                f"术语“{name}”原绑定的{old_label}（{old_def.get('label', '')}）"
                f"删除后，本处命中 {len(post['target_ids'])} 个同名定义"
                f"（{'、'.join(cand)}），变为歧义绑定"
            )
        else:
            new_def = new_defs.get(post["target_ids"][0], {})
            message = (
                f"术语“{name}”将改绑其他定义："
                f"{old_def.get('label', '')} → {new_def.get('label', '')}"
            )

        # 根因落到「被移除的旧定义」所在条款（前置说明中的全文定义不可删）
        terminal = old_def.get("clause_id") or owner
        add(owner, {
            "kind": kind,
            "message": message,
            "start": um["start"],
            "target_id": terminal if terminal != "preamble" else None,
        })

    return by_clause, preamble_details


# ---------------------------------------------------------------- 强连通分量（Tarjan，迭代版）

def _scc(nodes: list[str], edges: dict[str, set[str]],
         doc: dict[str, int]) -> dict[str, int]:
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    comp: dict[str, int] = {}
    counter = 0

    for start in nodes:
        if start in index:
            continue
        index[start] = low[start] = counter
        counter += 1
        stack.append(start)
        on_stack.add(start)
        work: list[tuple[str, object]] = [
            (start, iter(sorted(edges[start], key=lambda x: doc[x])))]
        while work:
            v, it = work[-1]
            descended = False
            for w in it:  # type: ignore[union-attr]
                if w not in index:
                    index[w] = low[w] = counter
                    counter += 1
                    stack.append(w)
                    on_stack.add(w)
                    work.append((w, iter(sorted(edges[w],
                                                key=lambda x: doc[x]))))
                    descended = True
                    break
                if w in on_stack:
                    low[v] = min(low[v], index[w])
            if descended:
                continue
            if low[v] == index[v]:
                cid = len(comp)
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    comp[w] = cid
                    if w == v:
                        break
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[v])
    return comp


# ---------------------------------------------------------------- 主入口

def preview_deletion(preamble: str,
                     clauses: list[Clause],
                     order: list[str],
                     deleted: list[str],
                     candidates: list[str]) -> dict:
    by_id = {c.id: c for c in clauses}
    doc = {c.id: i for i, c in enumerate(clauses)}
    path_to_id: dict[tuple[int, ...], str] = {}
    for c in clauses:
        path_to_id.setdefault(c.path, c.id)

    cur_del = cascade_deleted(clauses, deleted)
    valid_pick = [cid for cid in candidates
                  if cid in by_id and cid not in cur_del]
    if not valid_pick:
        raise ValueError("没有可预演删除的存活条款")
    pick_set = set(valid_pick)
    new_del = cascade_deleted(clauses, deleted + valid_pick)
    added = new_del - cur_del

    _, survivors_cur, _ = _canonical_order(clauses, order, cur_del)
    _, survivors_new, _ = _canonical_order(clauses, order, new_del)
    survivors_new_set = set(survivors_new)

    raw_by = {c.id: _scan(c.text) for c in clauses}

    # ---- 1) 引用绑定：仅报告删除前存活、删除后落入删除集的目标 ----
    direct: dict[str, list[dict]] = {}
    seen: dict[str, set] = {}

    def add_direct(owner: str, key: tuple, d: dict) -> None:
        bucket = seen.setdefault(owner, set())
        if key in bucket:
            return
        bucket.add(key)
        direct.setdefault(owner, []).append(d)

    for cid in survivors_cur:
        if cid in new_del:
            continue
        for r in raw_by[cid]:
            if r.kind == "single":
                tid = path_to_id.get(r.a)  # type: ignore[arg-type]
                if tid and tid not in cur_del and tid in added:
                    t = by_id[tid]
                    add_direct(cid, ("ref", r.text, tid), {
                        "kind": "ref_broken",
                        "message": (
                            f"引用失效：「{r.text}」指向的{t.label}"
                            f"（{t.heading or '该条款'}）将被删除"
                        ),
                        "start": r.start,
                        "target_id": tid,
                    })
            elif r.kind == "range":
                ids, _ = expand_range(r.a, r.b, path_to_id)  # type: ignore
                lost = [t for t in ids
                        if t not in cur_del and t in added]
                if lost:
                    labels = "、".join(
                        f"{by_id[t].label}（{by_id[t].heading or '该条款'}）"
                        for t in lost)
                    add_direct(cid, ("ref", r.text, tuple(lost)), {
                        "kind": "ref_broken",
                        "message": f"区间引用「{r.text}」中 {labels} 将失效",
                        "start": r.start,
                        "target_id": lost[0],
                    })

    # ---- 2) 术语绑定对照 ----
    term_by_clause, preamble_details = _term_details(
        preamble, clauses, cur_del, new_del)
    for owner, dets in term_by_clause.items():
        for d in dets:
            add_direct(owner, ("term", d["kind"], d["message"], d["start"]), d)

    # 根因条款（旧定义/被引目标）：取该根最小偏移明细
    root_terminal: dict[str, str | None] = {}
    for owner, dets in direct.items():
        first = min(dets, key=lambda d: (d["start"],
                                         _DETAIL_ORDER.get(d["kind"], 9)))
        root_terminal[owner] = first.get("target_id")

    # ---- 3) 删除后存活条款上的引用图（含显式自引用，区间自指不计边） ----
    edges: dict[str, set[str]] = {cid: set() for cid in survivors_new}
    for cid in survivors_new:
        for r in raw_by[cid]:
            if r.kind == "single":
                tid = path_to_id.get(r.a)  # type: ignore[arg-type]
                if tid and tid in survivors_new_set:
                    edges[cid].add(tid)
            elif r.kind == "range":
                ids, _ = expand_range(r.a, r.b, path_to_id)  # type: ignore
                for tid in ids:
                    if tid != cid and tid in survivors_new_set:
                        edges[cid].add(tid)

    # ---- 4) 多源 BFS（反向边）求最短距离 ----
    roots = set(direct)
    rev: dict[str, list[str]] = {cid: [] for cid in survivors_new}
    for u, vs in edges.items():
        for v in vs:
            rev[v].append(u)
    for vs in rev.values():
        vs.sort(key=lambda x: doc[x])

    dist: dict[str, int] = {r: 0 for r in roots}
    dq: deque[str] = deque(sorted(roots, key=lambda x: doc[x]))
    while dq:
        cur = dq.popleft()
        for nxt in rev[cur]:
            if nxt not in dist:
                dist[nxt] = dist[cur] + 1
                dq.append(nxt)

    # 等长链取文档顺序字典序最小者：按距离分层，沿出边选 best 链最小的前驱
    best: dict[str, list[str]] = {r: [r] for r in roots}
    max_dist = max(dist.values(), default=0)
    for level in range(1, max_dist + 1):
        layer = sorted((n for n in dist if dist[n] == level),
                       key=lambda x: doc[x])
        for n in layer:
            preds = [p for p in edges[n] if dist.get(p) == level - 1]
            chosen = min(preds,
                         key=lambda p: [doc[q] for q in best[p]])
            best[n] = [n] + best[chosen]

    # ---- 5) 循环边：同属一个 SCC（含自环）的边 ----
    comp = _scc(survivors_new, edges, doc)
    comp_size: dict[int, int] = {}
    for c in comp.values():
        comp_size[c] = comp_size.get(c, 0) + 1

    def is_cycle_edge(u: str, v: str) -> bool:
        cu, cv = comp.get(u), comp.get(v)
        if cu is None or cu != cv:
            return False
        return comp_size[cu] > 1 or u == v

    # ---- 6) 组装影响行（每个存活条款至多一行，明细去重后按偏移排序）----
    def chain_for(cid: str) -> tuple[list[dict], bool]:
        ids = list(best[cid])
        terminal = root_terminal.get(ids[-1])
        cyclic = any(is_cycle_edge(ids[i], ids[i + 1])
                     for i in range(len(ids) - 1))
        entries = [_clause_brief(by_id.get(x)) for x in ids]
        if terminal and terminal != ids[-1]:
            entries.append(_clause_brief(by_id.get(terminal)))
            cyclic = cyclic or is_cycle_edge(ids[-1], terminal)
        return entries, cyclic

    impacts: list[dict] = []
    for cid in sorted(dist, key=lambda x: (dist[x], doc[x])):
        chain, cyclic = chain_for(cid)
        c = by_id[cid]
        dets = sorted(direct.get(cid, []),
                      key=lambda d: (d["start"],
                                     _DETAIL_ORDER.get(d["kind"], 9)))
        impacts.append({
            "clause_id": cid,
            "label": c.label,
            "heading": c.heading or "",
            "distance": dist[cid],
            "cycle": cyclic,
            "chain": chain,
            "details": [{
                "kind": d["kind"],
                "message": d["message"],
                "target_id": d.get("target_id"),
            } for d in dets],
        })

    # 前置说明中的直接影响（术语使用；不能成为引用边节点）
    if preamble_details:
        dets = sorted(preamble_details,
                      key=lambda d: (d["start"],
                                     _DETAIL_ORDER.get(d["kind"], 9)))
        terminal = dets[0].get("target_id")
        chain = [{"id": "preamble", "label": "前置说明", "heading": ""}]
        if terminal:
            chain.append(_clause_brief(by_id.get(terminal)))
        impacts.insert(0, {
            "clause_id": "preamble",
            "label": "前置说明",
            "heading": "",
            "distance": 0,
            "cycle": False,
            "chain": chain,
            "details": [{
                "kind": d["kind"],
                "message": d["message"],
                "target_id": d.get("target_id"),
            } for d in dets],
        })

    to_delete = []
    for c in clauses:
        if c.id in added:
            item = _clause_brief(c)
            item["selected"] = c.id in pick_set
            to_delete.append(item)
    already = [
        _clause_brief(by_id[c.id]) for c in clauses if c.id in cur_del
    ]

    n_ref = sum(1 for row in impacts for d in row["details"]
                if d["kind"] == "ref_broken")
    n_term = sum(len(row["details"]) for row in impacts) - n_ref
    return {
        "candidates": [c.id for c in clauses if c.id in pick_set],
        "to_delete": to_delete,
        "already_deleted": already,
        "impacts": impacts,
        "counts": {
            "to_delete": len(to_delete),
            "selected": len(pick_set),
            "cascade": len(to_delete) - len(pick_set),
            "impacts": len(impacts),
            "direct": sum(1 for r in impacts if r["distance"] == 0),
            "transitive": sum(1 for r in impacts if r["distance"] > 0),
            "ref": n_ref,
            "term": n_term,
            "cycles": sum(1 for r in impacts if r["cycle"]),
        },
    }
