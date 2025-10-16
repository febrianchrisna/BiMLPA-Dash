import networkx as nx
import numpy as np
from collections import Counter
from math import sqrt
from random import choice
import argparse
import generator  # untuk memuat graf dari file


class BiMLPA(object):
    def __init__(self, G, threshold, max_prop_label, max_MM_iter=100, max_MS_iter=100):
        self.G = G
        self.threshold = threshold
        self.max_prop_label = max_prop_label
        self.max_MM_iter = max_MM_iter
        self.max_MS_iter = max_MS_iter

    def _initialize(self):
        G = self.G
        top = {n for n, d in G.nodes(data=True) if d['bipartite'] == 0}
        bottom = set(G) - top
        if len(top) >= len(bottom):
            self.red = top
            self.blue = bottom
        else:
            self.red = bottom
            self.blue = top
        for i, v in enumerate(self.red):
            G.nodes[v]['label'] = {i+1: 1}
        for v in self.blue:
            G.nodes[v]['label'] = {}

    def _label_to_list(self, propagaters):
        # propagaterが持つラベル数をmax_prop_label以下に
        # ラベルの重みの降順でソート
        node2label  = dict(nx.get_node_attributes(self.G, 'label'))
        for node in propagaters:
            label = node2label[node]
            l = list(label.keys())
            r = list(label.values())
            if len(label) > self.max_prop_label:
                index = np.argsort(r)[::-1][:self.max_prop_label]
                new_label = [[l[i] for i in index], [r[j] for j in index]]
            else:
                new_label = [l, r]
            node2label[node] = new_label
        return node2label

    def _sum_label_ratio(self, label_freq, u, node2label):
        neighbor = self.G.neighbors(u)
        for v in neighbor:
            label_index, label_ratio = node2label[v]
            for i in range(len(label_index)):
                label_freq.update({label_index[i]: label_ratio[i]})

    def _propagate_multi_labels(self, receivers):
        G = self.G
        convergence = True
        propagaters = set(G) - set(receivers)
        node2label  = self._label_to_list(propagaters)

        # 各ノード、neighborからラベルを取得しthresholdを超えたラベルのみ取得
        for u in receivers:
            old_label = node2label[u]
            label_freq = Counter()
            self._sum_label_ratio(label_freq, u, node2label)

            freq_max = max(label_freq.values())
            new_labels = {label: freq for label, freq in label_freq.items() if freq/freq_max >= self.threshold}
            freq_sum = sum(new_labels.values())
            new_labels = {label: new_labels[label]/freq_sum for label in new_labels}
            G.nodes[u]['label'] = new_labels
            if convergence and (old_label.keys() != new_labels.keys()):
                convergence = False
        return convergence

    def _propagate_single_label(self, receivers):
        G = self.G
        convergence = True
        propagaters = set(G) - set(receivers)
        node2label  = self._label_to_list(propagaters)

        for u in receivers:
            old_label = node2label[u]
            label_freq = Counter()
            self._sum_label_ratio(label_freq, u, node2label)

            # jika node isolat: lewati
            if not label_freq:
                new_label = {}
            else:
                freq_max = max(label_freq.values())
                candidate = [label for label, freq in label_freq.items() if freq == freq_max]
                # Random tie-breaking: gunakan choice()
                new_label = {choice(candidate): 1}

            G.nodes[u]['label'] = new_label
            if convergence and old_label != new_label:
                convergence = False
        return convergence

    def _multi_multi_LP(self):
        # Multi Multi LP
        for _ in range(self.max_MM_iter):
            conv_blue = self._propagate_multi_labels(self.blue)
            conv_red  = self._propagate_multi_labels(self.red)
            if conv_blue and conv_red:
                break

    def _multi_single_LP(self):
        # Multi Single LP
        for _ in range(self.max_MS_iter):
            conv_blue = self._propagate_multi_labels(self.blue)
            conv_red  = self._propagate_single_label(self.red)
            if conv_blue and conv_red:
                break

    def start(self):
        self._initialize()
        self._multi_multi_LP()
        self._multi_single_LP()


class BiMLPA_SqrtDeg(BiMLPA):
    def __init__(self, G, threshold, max_prop_label, max_MM_iter=100, max_MS_iter=100):
        super().__init__(G, threshold, max_prop_label, max_MM_iter, max_MS_iter)
        self.node2degree = dict(G.degree())

    def _label_to_list(self, propagaters):
        node2label  = dict(nx.get_node_attributes(self.G, 'label'))
        for node in propagaters:
            d_sqrt = sqrt(self.node2degree[node])
            label = node2label[node]
            l = list(label.keys())
            r = list(label.values())
            if len(label) > self.max_prop_label:
                index = np.argsort(r)[::-1][:self.max_prop_label]
                new_label = [[l[i] for i in index], [r[j]/d_sqrt for j in index]]
            else:
                r = [ratio/d_sqrt for ratio in r]
                new_label = [l, r]
            node2label[node] = new_label
        return node2label


def _pick_single(label_dict: dict) -> int:
    # label dengan bobot terbesar; deterministik jika seri
    if not label_dict:
        return -1
    m = max(label_dict.values())
    cands = [k for k, v in label_dict.items() if v == m]
    return min(cands)

def _labels_to_partition(G: nx.Graph):
    node2lab = {}
    for n, data in G.nodes(data=True):
        ld = data.get('label', {})
        node2lab[n] = _pick_single(ld) if isinstance(ld, dict) else -1
    part = {}
    for n, lab in node2lab.items():
        part.setdefault(lab, set()).add(n)
    part.pop(-1, None)
    communities = list(part.values())
    return node2lab, communities

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BiMLPA (bipartite community detection) from terminal.")
    parser.add_argument("path", help="Path input (edge list 2 kolom)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--ints", action="store_true", help="Edge list 2 kolom integer (default)")
    mode.add_argument("--names", action="store_true", help="Edge list 2 kolom nama (dipisah TAB)")

    parser.add_argument("--variant", choices=["vanilla", "sqrtdeg"], default="vanilla")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-prop-label", type=int, default=3)
    parser.add_argument("--max-mm-iter", type=int, default=100)
    parser.add_argument("--max-ms-iter", type=int, default=100)

    parser.add_argument("--out-csv", help="Simpan mapping node→community ke CSV")
    parser.add_argument("--out-graph", help="Simpan graf dengan atribut 'community' (.gpickle/.graphml/.edgelist)")
    args = parser.parse_args()

    # Muat graf
    G = generator.generate_network_with_name(args.path) if args.names else generator.generate_network(args.path)

    # Backup label data ke 'display' jika ada, agar algoritma bebas pakai 'label'
    for n, data in G.nodes(data=True):
        if 'label' in data and not isinstance(data['label'], dict):
            data['display'] = data['label']
            del data['label']

    # Pilih varian
    if args.variant == "sqrtdeg":
        algo = BiMLPA_SqrtDeg(G, args.threshold, args.max_prop_label, args.max_mm_iter, args.max_ms_iter)
    else:
        algo = BiMLPA(G, args.threshold, args.max_prop_label, args.max_mm_iter, args.max_ms_iter)

    # Jalankan
    algo.start()

    # Ekstrak hasil
    node2lab, communities = _labels_to_partition(G)
    print(f"Communities: {len(communities)}")
    sizes = sorted((len(c) for c in communities), reverse=True)
    print(f"Sizes: {sizes}")

    # Simpan CSV
    if args.out_csv:
        import csv as _csv
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["node_id", "display", "side", "community"])
            for n in G.nodes():
                disp = G.nodes[n].get("display", n)
                side = "left" if G.nodes[n].get("bipartite") == 0 else "right"
                w.writerow([n, disp, side, node2lab.get(n, -1)])
        print(f"Wrote: {args.out_csv}")

    # Simpan graf (tambahkan atribut community)
    if args.out_graph:
        for n in G.nodes():
            G.nodes[n]["community"] = node2lab.get(n, -1)
        out = args.out_graph.lower()
        if out.endswith(".gpickle"):
            nx.write_gpickle(G, args.out_graph)
        elif out.endswith(".graphml"):
            nx.write_graphml(G, args.out_graph)
        elif out.endswith(".edgelist"):
            nx.write_edgelist(G, args.out_graph, data=["community"])
        else:
            raise SystemExit("Unknown graph extension. Use .gpickle/.graphml/.edgelist")
        print(f"Wrote: {args.out_graph}")
