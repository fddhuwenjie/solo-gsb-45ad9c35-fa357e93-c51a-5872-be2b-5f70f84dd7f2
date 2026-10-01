"""删除影响预演（只读分析，不修改草稿）。

把用户勾选的存活条款（父条款按现有级联规则连同后代）加入删除集，比较删除
前后两类绑定：

1. 引用绑定：删除前可解析且目标存活、删除后目标落入删除集的引用（区间按
   展开后的单条目标逐个比较，仅报告「新增失效」部分）；
2. 定义术语绑定：删除前**唯一**绑定某定义的使用，删除后失去该唯一定义
   （未定义 / 无存活同名定义）、命中多个同名定义（歧义）或改绑其他定义。

直接受影响的存活条款为根，先把删除后存活引用图按**强连通分量（SCC）合并**：
同一循环组只生成一个影响项、组内跨成员的同一直接影响去重；再在收缩后的
SCC DAG 上沿「反向边」多源 BFS，得到每个受影响条款/循环组到根因的最短
原因链；等长链按文档顺序取字典序最小者；链上任一节点为循环 SCC 时整条链
标注循环。
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

    # ---- 4) 存活引用图按强连通分量合并：同组节点只产生一个影响项 ----
    comp = _scc(survivors_new, edges, doc)
    groups: dict[int, list[str]] = {}
    for cid in survivors_new:
        groups.setdefault(comp[cid], []).append(cid)
    for members in groups.values():
        members.sort(key=lambda x: doc[x])
    # 循环 SCC：多成员分量，或单成员上存在显式自环
    cyclic_group: dict[int, bool] = {}
    for cid, k in comp.items():
        if len(groups[k]) > 1 or cid in edges[cid]:
            cyclic_group[k] = True
    # 分量代表条款（文档顺序最靠前）与分量文档顺序
    rep = {k: members[0] for k, members in groups.items()}
    comp_doc = {k: doc[members[0]] for k, members in groups.items()}

    # 组内直接影响合并：跨成员的同一影响去重
    # （key 取 (种类, 根因, 消息)：同组中对同一被删条款的同款失效只算一次）
    comp_details: dict[int, list[dict]] = {}
    for k in sorted(groups, key=lambda g: comp_doc[g]):
        merged: list[dict] = []
        seen_key: set[tuple] = set()
        for cid in groups[k]:
            for d in direct.get(cid, []):
                key = (d["kind"], d.get("target_id"), d["message"])
                if key in seen_key:
                    continue
                seen_key.add(key)
                md = dict(d)
                md["owner_id"] = cid
                merged.append(md)
        if merged:
            merged.sort(key=lambda d: (doc[d["owner_id"]], d["start"],
                                       _DETAIL_ORDER.get(d["kind"], 9)))
            comp_details[k] = merged

    # 每个根分量（含直接影响的分量）的根因条款（去重，按文档顺序）
    root_terminal: dict[int, list[str]] = {}
    for k, dets in comp_details.items():
        terms = sorted({d.get("target_id") for d in dets if d.get("target_id")},
                       key=lambda t: doc.get(t, 10 ** 9))
        root_terminal[k] = terms  # type: ignore[assignment]

    # ---- 5) SCC DAG 上多源 BFS（反向边）求最短分量距离 ----
    c_edges: dict[int, set[int]] = {k: set() for k in groups}
    for u, vs in edges.items():
        for v in vs:
            if comp[u] != comp[v]:
                c_edges[comp[u]].add(comp[v])
    c_rev: dict[int, list[int]] = {k: [] for k in groups}
    for u, vs in c_edges.items():
        for v in vs:
            c_rev[v].append(u)
    for ks in c_rev.values():
        ks.sort(key=lambda k: comp_doc[k])

    roots = set(comp_details)
    dist: dict[int, int] = {r: 0 for r in roots}
    dq: deque[int] = deque(sorted(roots, key=lambda k: comp_doc[k]))
    while dq:
        cur = dq.popleft()
        for nxt in c_rev[cur]:
            if nxt not in dist:
                dist[nxt] = dist[cur] + 1
                dq.append(nxt)

    # 等长链取文档顺序字典序最小者：按距离分层，沿收缩出边选 best 链最小的前驱
    best: dict[int, list[int]] = {r: [r] for r in roots}
    max_dist = max(dist.values(), default=0)
    for level in range(1, max_dist + 1):
        layer = sorted((k for k in dist if dist[k] == level),
                       key=lambda k: comp_doc[k])
        for k in layer:
            preds = [p for p in c_edges[k] if dist.get(p) == level - 1]
            chosen = min(preds,
                         key=lambda p: [comp_doc[q] for q in best[p]])
            best[k] = [k] + best[chosen]

    # ---- 6) 组装影响行（每个 SCC 至多一行；组内影响已去重）----
    def group_node(k: int) -> dict:
        members = groups[k]
        node: dict = {
            "id": rep[k],
            "label": by_id[rep[k]].label,
            "heading": by_id[rep[k]].heading or "",
            "cycle": cyclic_group.get(k, False),
        }
        if len(members) > 1:
            node["group"] = True
            node["members"] = [_clause_brief(by_id[x]) for x in members]
        return node

    impacts: list[dict] = []
    for k in sorted(dist, key=lambda g: (dist[g], comp_doc[g])):
        cks = best[k]
        chain = [group_node(g) for g in cks]
        for tid in root_terminal.get(cks[-1], []):
            term_node = _clause_brief(by_id.get(tid))
            term_node["terminal"] = True
            chain.append(term_node)
        cyclic = any(cyclic_group.get(g, False) for g in cks)

        c = by_id[rep[k]]
        members = groups[k]
        dets_out = [{
            "kind": d["kind"],
            "message": d["message"],
            "clause_id": d.get("owner_id"),
            "target_id": d.get("target_id"),
        } for d in comp_details.get(k, [])]
        impacts.append({
            "clause_id": c.id,
            "label": c.label,
            "heading": c.heading or "",
            "distance": dist[k],
            "cycle": cyclic,
            "group": len(members) > 1,
            "members": [_clause_brief(by_id[x]) for x in members]
            if len(members) > 1 else [],
            "chain": chain,
            "details": dets_out,
        })

    # 前置说明中的直接影响（术语使用；不能成为引用边节点）
    if preamble_details:
        dets = sorted(preamble_details,
                      key=lambda d: (d["start"],
                                     _DETAIL_ORDER.get(d["kind"], 9)))
        chain = [{"id": "preamble", "label": "前置说明", "heading": ""}]
        terminals = sorted({d.get("target_id") for d in dets
                            if d.get("target_id")},
                           key=lambda t: doc.get(t, 10 ** 9))
        for tid in terminals:
            term_node = _clause_brief(by_id.get(tid))
            term_node["terminal"] = True
            chain.append(term_node)
        impacts.insert(0, {
            "clause_id": "preamble",
            "label": "前置说明",
            "heading": "",
            "distance": 0,
            "cycle": False,
            "group": False,
            "members": [],
            "chain": chain,
            "details": [{
                "kind": d["kind"],
                "message": d["message"],
                "clause_id": None,
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
