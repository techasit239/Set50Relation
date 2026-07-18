# -*- coding: utf-8 -*-
"""
Stream-parse the MemeTracker/Spinn3r quotes corpus (P/T/Q/L blocks) and aggregate
a domain-to-domain hyperlink network. Never loads the source file fully into memory.

Input:
    quotes_2009-04.txt.gz   (P=permalink, T=timestamp, Q=quote, L=link; blank line separates blocks)

Output (written next to this script):
    domain_edges_raw.csv    src_domain,tgt_domain,weight   (external links only, weight = raw link count)
    domain_nodes_raw.csv    domain,internal_link_count     (links where target domain == source domain)

Run:
    python build_domain_edges.py [path_to_quotes_gz]
"""
import gzip
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_INPUT = r"C:\Users\techasit\Downloads\quotes_2009-04.txt.gz"
OUT_DIR = Path(__file__).parent
EDGES_OUT = OUT_DIR / "domain_edges_raw.csv"
NODES_OUT = OUT_DIR / "domain_nodes_raw.csv"

PROGRESS_EVERY = 500_000  # blocks


def get_domain(url: str) -> str:
    try:
        netloc = urlparse(url.strip()).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""


def main() -> None:
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_INPUT)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    edge_weight: Counter = Counter()
    internal_link_count: Counter = Counter()

    n_blocks = 0
    n_external_links = 0
    n_internal_links = 0
    current_p_domain = None
    have_p = False

    start = time.time()
    with gzip.open(input_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "":
                if have_p:
                    n_blocks += 1
                    if n_blocks % PROGRESS_EVERY == 0:
                        elapsed = time.time() - start
                        print(
                            f"[{elapsed:8.1f}s] blocks={n_blocks:,} "
                            f"distinct_pairs={len(edge_weight):,} "
                            f"external_links={n_external_links:,} internal_links={n_internal_links:,}",
                            flush=True,
                        )
                current_p_domain = None
                have_p = False
                continue

            tag = line[:1]
            rest = line[2:] if len(line) > 1 and line[1] == "\t" else line[1:]

            if tag == "P":
                have_p = True
                current_p_domain = get_domain(rest)
            elif tag == "L":
                if not have_p or not current_p_domain:
                    continue
                d = get_domain(rest)
                if not d:
                    continue
                if d == current_p_domain:
                    internal_link_count[d] += 1
                    n_internal_links += 1
                else:
                    edge_weight[(current_p_domain, d)] += 1
                    n_external_links += 1
        # flush trailing block if file doesn't end with a blank line
        if have_p:
            n_blocks += 1

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s")
    print(f"total blocks (permalinks): {n_blocks:,}")
    print(f"total external links: {n_external_links:,}")
    print(f"total internal links: {n_internal_links:,}")
    print(f"distinct (src,tgt) domain pairs: {len(edge_weight):,}")

    with open(EDGES_OUT, "w", encoding="utf-8", newline="") as f:
        f.write("src_domain,tgt_domain,weight\n")
        for (src, tgt), w in edge_weight.items():
            f.write(f"{src},{tgt},{w}\n")
    print(f"wrote {EDGES_OUT}")

    # domains that only ever appear as a link target (never as P) still need a row
    all_domains = set(internal_link_count.keys())
    for src, tgt in edge_weight.keys():
        all_domains.add(src)
        all_domains.add(tgt)

    with open(NODES_OUT, "w", encoding="utf-8", newline="") as f:
        f.write("domain,internal_link_count\n")
        for d in all_domains:
            f.write(f"{d},{internal_link_count.get(d, 0)}\n")
    print(f"wrote {NODES_OUT}")


if __name__ == "__main__":
    main()
