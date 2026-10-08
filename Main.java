import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import java.io.*;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.URLDecoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.Executors;

public class Main {

    static final int DIMS = 16; // demo vectors

    // =====================================================================
    //  DATA TYPES
    // =====================================================================

    public static class VectorItem {
        public final int id;
        public final String metadata;
        public final String category;
        public final float[] emb;

        public VectorItem(int id, String metadata, String category, float[] emb) {
            this.id = id;
            this.metadata = metadata;
            this.category = category;
            this.emb = emb;
        }
    }

    @FunctionalInterface
    public interface DistFn {
        float apply(float[] a, float[] b);
    }

    // =====================================================================
    //  DISTANCE METRICS
    // =====================================================================

    public static float euclidean(float[] a, float[] b) {
        float s = 0;
        int len = Math.min(a.length, b.length);
        for (int i = 0; i < len; i++) {
            float d = a[i] - b[i];
            s += d * d;
        }
        return (float) Math.sqrt(s);
    }

    public static float cosine(float[] a, float[] b) {
        float dot = 0, na = 0, nb = 0;
        int len = Math.min(a.length, b.length);
        for (int i = 0; i < len; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        if (na < 1e-9f || nb < 1e-9f) return 1.0f;
        return 1.0f - (float) (dot / (Math.sqrt(na) * Math.sqrt(nb)));
    }

    public static float manhattan(float[] a, float[] b) {
        float s = 0;
        int len = Math.min(a.length, b.length);
        for (int i = 0; i < len; i++) {
            s += Math.abs(a[i] - b[i]);
        }
        return s;
    }

    public static DistFn getDistFn(String m) {
        if ("cosine".equalsIgnoreCase(m)) return Main::cosine;
        if ("manhattan".equalsIgnoreCase(m)) return Main::manhattan;
        return Main::euclidean;
    }

    public static class DistPair implements Comparable<DistPair> {
        public final float dist;
        public final int id;

        public DistPair(float dist, int id) {
            this.dist = dist;
            this.id = id;
        }

        @Override
        public int compareTo(DistPair o) {
            int cmp = Float.compare(this.dist, o.dist);
            if (cmp != 0) return cmp;
            return Integer.compare(this.id, o.id);
        }
    }

    // =====================================================================
    //  BRUTE FORCE
    // =====================================================================

    public static class BruteForce {
        public final List<VectorItem> items = new ArrayList<>();

        public synchronized void insert(VectorItem v) {
            items.add(v);
        }

        public synchronized List<DistPair> knn(float[] q, int k, DistFn dist) {
            List<DistPair> r = new ArrayList<>(items.size());
            for (VectorItem v : items) {
                r.add(new DistPair(dist.apply(q, v.emb), v.id));
            }
            Collections.sort(r);
            if (r.size() > k) {
                return new ArrayList<>(r.subList(0, k));
            }
            return r;
        }

        public synchronized void remove(int id) {
            items.removeIf(v -> v.id == id);
        }
    }

    // =====================================================================
    //  KD-TREE
    // =====================================================================

    public static class KDNode {
        public final VectorItem item;
        public KDNode left = null;
        public KDNode right = null;

        public KDNode(VectorItem item) {
            this.item = item;
        }
    }

    public static class KDTree {
        private KDNode root = null;
        private final int dims;

        public KDTree(int dims) {
            this.dims = dims;
        }

        private KDNode ins(KDNode n, VectorItem v, int d) {
            if (n == null) return new KDNode(v);
            int ax = d % dims;
            if (v.emb[ax] < n.item.emb[ax]) {
                n.left = ins(n.left, v, d + 1);
            } else {
                n.right = ins(n.right, v, d + 1);
            }
            return n;
        }

        private void knn(KDNode n, float[] q, int k, int d, DistFn dist,
                         PriorityQueue<DistPair> heap) {
            if (n == null) return;
            float dn = dist.apply(q, n.item.emb);
            if (heap.size() < k || dn < heap.peek().dist) {
                heap.offer(new DistPair(dn, n.item.id));
                if (heap.size() > k) heap.poll();
            }
            int ax = d % dims;
            float diff = q[ax] - n.item.emb[ax];
            KDNode closer = diff < 0 ? n.left : n.right;
            KDNode farther = diff < 0 ? n.right : n.left;

            knn(closer, q, k, d + 1, dist, heap);
            if (heap.size() < k || Math.abs(diff) < heap.peek().dist) {
                knn(farther, q, k, d + 1, dist, heap);
            }
        }

        public synchronized void insert(VectorItem v) {
            root = ins(root, v, 0);
        }

        public synchronized List<DistPair> knn(float[] q, int k, DistFn dist) {
            // Max-heap so peek() is the largest distance candidate in the top-k
            PriorityQueue<DistPair> heap = new PriorityQueue<>(k + 1, (a, b) -> Float.compare(b.dist, a.dist));
            knn(root, q, k, 0, dist, heap);
            List<DistPair> r = new ArrayList<>(heap);
            Collections.sort(r);
            return r;
        }

        public synchronized void rebuild(List<VectorItem> items) {
            root = null;
            for (VectorItem v : items) {
                insert(v);
            }
        }
    }

    // =====================================================================
    //  HNSW — Hierarchical Navigable Small World
    // =====================================================================

    public static class HNSW {
        public static class Node {
            public final VectorItem item;
            public final int maxLyr;
            public final List<List<Integer>> nbrs;

            public Node(VectorItem item, int maxLyr) {
                this.item = item;
                this.maxLyr = maxLyr;
                this.nbrs = new ArrayList<>(maxLyr + 1);
                for (int i = 0; i <= maxLyr; i++) {
                    this.nbrs.add(new ArrayList<>());
                }
            }
        }

        private final Map<Integer, Node> G = new HashMap<>();
        private final int M;
        private final int M0;
        private final int efBuild;
        private final float mL;
        private int topLayer = -1;
        private int entryPt = -1;
        private final Random rng = new Random(42);

        public HNSW(int m, int efBuild) {
            this.M = m;
            this.M0 = 2 * m;
            this.efBuild = efBuild;
            this.mL = 1.0f / (float) Math.log(m);
        }

        private int randLevel() {
            float u = rng.nextFloat();
            if (u <= 0.0f) u = 1e-7f;
            return (int) Math.floor(-Math.log(u) * mL);
        }

        private List<DistPair> searchLayer(float[] q, int ep, int ef, int lyr, DistFn dist) {
            Set<Integer> vis = new HashSet<>();
            // Min-heap for candidate queue
            PriorityQueue<DistPair> cands = new PriorityQueue<>();
            // Max-heap for found results
            PriorityQueue<DistPair> found = new PriorityQueue<>(ef + 1, (a, b) -> Float.compare(b.dist, a.dist));

            Node epNode = G.get(ep);
            if (epNode == null) return Collections.emptyList();

            float d0 = dist.apply(q, epNode.item.emb);
            vis.add(ep);
            DistPair start = new DistPair(d0, ep);
            cands.offer(start);
            found.offer(start);

            while (!cands.isEmpty()) {
                DistPair cand = cands.poll();
                if (found.size() >= ef && cand.dist > found.peek().dist) break;

                Node currNode = G.get(cand.id);
                if (currNode == null || lyr >= currNode.nbrs.size()) continue;

                List<Integer> layerNbrs = currNode.nbrs.get(lyr);
                for (int nid : layerNbrs) {
                    if (vis.contains(nid) || !G.containsKey(nid)) continue;
                    vis.add(nid);

                    Node nNode = G.get(nid);
                    float nd = dist.apply(q, nNode.item.emb);
                    if (found.size() < ef || nd < found.peek().dist) {
                        DistPair np = new DistPair(nd, nid);
                        cands.offer(np);
                        found.offer(np);
                        if (found.size() > ef) found.poll();
                    }
                }
            }

            List<DistPair> res = new ArrayList<>(found);
            Collections.sort(res);
            return res;
        }

        private List<Integer> selectNbrs(List<DistPair> cands, int maxM) {
            List<Integer> r = new ArrayList<>(Math.min(cands.size(), maxM));
            for (int i = 0; i < Math.min(cands.size(), maxM); i++) {
                r.add(cands.get(i).id);
            }
            return r;
        }

        public synchronized void insert(VectorItem item, DistFn dist) {
            int id = item.id;
            int lvl = randLevel();
            Node node = new Node(item, lvl);
            G.put(id, node);

            if (entryPt == -1) {
                entryPt = id;
                topLayer = lvl;
                return;
            }

            int ep = entryPt;
            for (int lc = topLayer; lc > lvl; lc--) {
                Node epNode = G.get(ep);
                if (epNode != null && lc < epNode.nbrs.size()) {
                    List<DistPair> W = searchLayer(item.emb, ep, 1, lc, dist);
                    if (!W.isEmpty()) ep = W.get(0).id;
                }
            }

            for (int lc = Math.min(topLayer, lvl); lc >= 0; lc--) {
                List<DistPair> W = searchLayer(item.emb, ep, efBuild, lc, dist);
                int maxM = (lc == 0) ? M0 : M;
                List<Integer> sel = selectNbrs(W, maxM);
                node.nbrs.set(lc, new ArrayList<>(sel));

                for (int nid : sel) {
                    Node nNode = G.get(nid);
                    if (nNode == null) continue;
                    while (nNode.nbrs.size() <= lc) {
                        nNode.nbrs.add(new ArrayList<>());
                    }
                    List<Integer> conn = nNode.nbrs.get(lc);
                    conn.add(id);
                    if (conn.size() > maxM) {
                        List<DistPair> ds = new ArrayList<>();
                        for (int c : conn) {
                            Node cNode = G.get(c);
                            if (cNode != null) {
                                ds.add(new DistPair(dist.apply(nNode.item.emb, cNode.item.emb), c));
                            }
                        }
                        Collections.sort(ds);
                        conn.clear();
                        for (int i = 0; i < maxM && i < ds.size(); i++) {
                            conn.add(ds.get(i).id);
                        }
                    }
                }
                if (!W.isEmpty()) ep = W.get(0).id;
            }

            if (lvl > topLayer) {
                topLayer = lvl;
                entryPt = id;
            }
        }

        public synchronized List<DistPair> knn(float[] q, int k, int ef, DistFn dist) {
            if (entryPt == -1) return Collections.emptyList();
            int ep = entryPt;
            for (int lc = topLayer; lc > 0; lc--) {
                Node epNode = G.get(ep);
                if (epNode != null && lc < epNode.nbrs.size()) {
                    List<DistPair> W = searchLayer(q, ep, 1, lc, dist);
                    if (!W.isEmpty()) ep = W.get(0).id;
                }
            }
            List<DistPair> W = searchLayer(q, ep, Math.max(ef, k), 0, dist);
            if (W.size() > k) {
                return new ArrayList<>(W.subList(0, k));
            }
            return W;
        }

        public synchronized void remove(int id) {
            if (!G.containsKey(id)) return;
            for (Node nd : G.values()) {
                for (List<Integer> layer : nd.nbrs) {
                    layer.removeIf(val -> val == id);
                }
            }
            if (entryPt == id) {
                entryPt = -1;
                for (int nid : G.keySet()) {
                    if (nid != id) {
                        entryPt = nid;
                        break;
                    }
                }
            }
            G.remove(id);
        }

        public static class NodeView {
            public final int id;
            public final String metadata;
            public final String category;
            public final int maxLyr;

            public NodeView(int id, String metadata, String category, int maxLyr) {
                this.id = id;
                this.metadata = metadata;
                this.category = category;
                this.maxLyr = maxLyr;
            }
        }

        public static class EdgeView {
            public final int src;
            public final int dst;
            public final int lyr;

            public EdgeView(int src, int dst, int lyr) {
                this.src = src;
                this.dst = dst;
                this.lyr = lyr;
            }
        }

        public static class GraphInfo {
            public int topLayer;
            public int nodeCount;
            public final List<Integer> nodesPerLayer = new ArrayList<>();
            public final List<Integer> edgesPerLayer = new ArrayList<>();
            public final List<NodeView> nodes = new ArrayList<>();
            public final List<EdgeView> edges = new ArrayList<>();
        }

        public synchronized GraphInfo getInfo() {
            GraphInfo gi = new GraphInfo();
            gi.topLayer = topLayer;
            gi.nodeCount = G.size();
            int maxL = Math.max(topLayer + 1, 1);
            for (int i = 0; i < maxL; i++) {
                gi.nodesPerLayer.add(0);
                gi.edgesPerLayer.add(0);
            }
            for (Map.Entry<Integer, Node> entry : G.entrySet()) {
                int id = entry.getKey();
                Node nd = entry.getValue();
                gi.nodes.add(new NodeView(id, nd.item.metadata, nd.item.category, nd.maxLyr));
                for (int lc = 0; lc <= nd.maxLyr && lc < maxL; lc++) {
                    gi.nodesPerLayer.set(lc, gi.nodesPerLayer.get(lc) + 1);
                    if (lc < nd.nbrs.size()) {
                        for (int nid : nd.nbrs.get(lc)) {
                            if (id < nid) {
                                gi.edgesPerLayer.set(lc, gi.edgesPerLayer.get(lc) + 1);
                                gi.edges.add(new EdgeView(id, nid, lc));
                            }
                        }
                    }
                }
            }
            return gi;
        }

        public synchronized int size() {
            return G.size();
        }
    }

    // =====================================================================
    //  VECTOR DATABASE (demo 16D index)
    // =====================================================================

    public static class VectorDB {
        private final Map<Integer, VectorItem> store = new HashMap<>();
        private final BruteForce bf = new BruteForce();
        private final KDTree kdt;
        private final HNSW hnsw = new HNSW(16, 200);
        public final int dims;
        private int nextId = 1;

        public VectorDB(int dims) {
            this.dims = dims;
            this.kdt = new KDTree(dims);
        }

        public synchronized int insert(String meta, String cat, float[] emb, DistFn dist) {
            VectorItem v = new VectorItem(nextId++, meta, cat, emb);
            store.put(v.id, v);
            bf.insert(v);
            kdt.insert(v);
            hnsw.insert(v, dist);
            return v.id;
        }

        public synchronized boolean remove(int id) {
            if (!store.containsKey(id)) return false;
            store.remove(id);
            bf.remove(id);
            hnsw.remove(id);
            kdt.rebuild(new ArrayList<>(store.values()));
            return true;
        }

        public static class Hit {
            public final int id;
            public final String meta;
            public final String cat;
            public final float[] emb;
            public final float dist;

            public Hit(int id, String meta, String cat, float[] emb, float dist) {
                this.id = id;
                this.meta = meta;
                this.cat = cat;
                this.emb = emb;
                this.dist = dist;
            }
        }

        public static class SearchOut {
            public final List<Hit> hits = new ArrayList<>();
            public long us;
            public String algo;
            public String metric;
        }

        public synchronized SearchOut search(float[] q, int k, String metric, String algo) {
            DistFn dfn = getDistFn(metric);
            long t0 = System.nanoTime();

            List<DistPair> raw;
            if ("bruteforce".equalsIgnoreCase(algo)) {
                raw = bf.knn(q, k, dfn);
            } else if ("kdtree".equalsIgnoreCase(algo)) {
                raw = kdt.knn(q, k, dfn);
            } else {
                raw = hnsw.knn(q, k, 50, dfn);
            }

            long us = (System.nanoTime() - t0) / 1000;
            SearchOut out = new SearchOut();
            out.us = us;
            out.algo = algo;
            out.metric = metric;
            for (DistPair dp : raw) {
                VectorItem item = store.get(dp.id);
                if (item != null) {
                    out.hits.add(new Hit(dp.id, item.metadata, item.category, item.emb, dp.dist));
                }
            }
            return out;
        }

        public static class BenchOut {
            public long bfUs, kdUs, hnswUs;
            public int n;
        }

        public synchronized BenchOut benchmark(float[] q, int k, String metric) {
            DistFn dfn = getDistFn(metric);
            BenchOut bo = new BenchOut();
            bo.n = store.size();

            long t0 = System.nanoTime();
            bf.knn(q, k, dfn);
            bo.bfUs = (System.nanoTime() - t0) / 1000;

            long t1 = System.nanoTime();
            kdt.knn(q, k, dfn);
            bo.kdUs = (System.nanoTime() - t1) / 1000;

            long t2 = System.nanoTime();
            hnsw.knn(q, k, 50, dfn);
            bo.hnswUs = (System.nanoTime() - t2) / 1000;

            return bo;
        }

        public synchronized List<VectorItem> all() {
            return new ArrayList<>(store.values());
        }

        public synchronized HNSW.GraphInfo hnswInfo() {
            return hnsw.getInfo();
        }

        public synchronized int size() {
            return store.size();
        }
    }

    // =====================================================================
    //  JSON HELPERS
    // =====================================================================

    public static String jS(String s) {
        if (s == null) return "\"\"";
        StringBuilder sb = new StringBuilder();
        sb.append('"');
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"' -> sb.append("\\\"");
                case '\\' -> sb.append("\\\\");
                case '\n' -> sb.append("\\n");
                case '\r' -> sb.append("\\r");
                case '\t' -> sb.append("\\t");
                default -> sb.append(c);
            }
        }
        sb.append('"');
        return sb.toString();
    }

    public static String jVec(float[] v) {
        if (v == null) return "[]";
        StringBuilder sb = new StringBuilder();
        sb.append('[');
        for (int i = 0; i < v.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(String.format(Locale.US, "%.4f", v[i]));
        }
        sb.append(']');
        return sb.toString();
    }

    public static float[] parseVec(String s) {
        if (s == null || s.isEmpty()) return new float[0];
        String[] parts = s.split(",");
        List<Float> list = new ArrayList<>();
        for (String p : parts) {
            try {
                list.add(Float.parseFloat(p.trim()));
            } catch (Exception ignored) {}
        }
        float[] res = new float[list.size()];
        for (int i = 0; i < list.size(); i++) res[i] = list.get(i);
        return res;
    }

    public static String extractStr(String body, String key) {
        if (body == null) return "";
        String pattern = "\"" + key + "\"";
        int p = body.indexOf(pattern);
        if (p == -1) return "";
        p = body.indexOf(':', p);
        if (p == -1) return "";
        p++;
        while (p < body.length() && (body.charAt(p) == ' ' || body.charAt(p) == '\t' || body.charAt(p) == '\n' || body.charAt(p) == '\r')) {
            p++;
        }
        if (p >= body.length() || body.charAt(p) != '"') return "";
        p++;
        StringBuilder result = new StringBuilder();
        while (p < body.length()) {
            char c = body.charAt(p);
            if (c == '"') break;
            if (c == '\\' && p + 1 < body.length()) {
                p++;
                char next = body.charAt(p);
                switch (next) {
                    case '"' -> result.append('"');
                    case '\\' -> result.append('\\');
                    case 'n' -> result.append('\n');
                    case 'r' -> result.append('\r');
                    case 't' -> result.append('\t');
                    default -> result.append(next);
                }
            } else {
                result.append(c);
            }
            p++;
        }
        return result.toString();
    }

    public static int extractInt(String body, String key, int def) {
        if (body == null) return def;
        String pattern = "\"" + key + "\"";
        int p = body.indexOf(pattern);
        if (p == -1) return def;
        p = body.indexOf(':', p);
        if (p == -1) return def;
        p++;
        while (p < body.length() && (body.charAt(p) == ' ' || body.charAt(p) == '\t')) p++;
        int end = p;
        while (end < body.length() && (Character.isDigit(body.charAt(end)) || body.charAt(end) == '-')) end++;
        try {
            return Integer.parseInt(body.substring(p, end).trim());
        } catch (Exception e) {
            return def;
        }
    }

    public static float[] extractArr(String body, String key) {
        if (body == null) return new float[0];
        String pattern = "\"" + key + "\"";
        int p = body.indexOf(pattern);
        if (p == -1) return new float[0];
        p = body.indexOf('[', p);
        if (p == -1) return new float[0];
        int e = body.indexOf(']', p);
        if (e == -1) return new float[0];
        return parseVec(body.substring(p + 1, e));
    }

    // =====================================================================
    //  TEXT CHUNKER
    // =====================================================================

    public static List<String> chunkText(String text, int chunkWords, int overlapWords) {
        if (text == null || text.trim().isEmpty()) return Collections.emptyList();
        String[] words = text.trim().split("\\s+");
        if (words.length <= chunkWords) return List.of(text.trim());

        List<String> chunks = new ArrayList<>();
        int step = Math.max(1, chunkWords - overlapWords);
        for (int i = 0; i < words.length; i += step) {
            int end = Math.min(i + chunkWords, words.length);
            StringBuilder sb = new StringBuilder();
            for (int j = i; j < end; j++) {
                if (j > i) sb.append(' ');
                sb.append(words[j]);
            }
            chunks.add(sb.toString());
            if (end == words.length) break;
        }
        return chunks;
    }

    // =====================================================================
    //  OLLAMA CLIENT — wraps local Ollama REST API
    // =====================================================================

    public static class OllamaClient {
        private final String host;
        private final int port;
        private final HttpClient client;
        public String embedModel = "nomic-embed-text";
        public String genModel = "llama3.2";

        public OllamaClient(String host, int port) {
            this.host = host;
            this.port = port;
            this.client = HttpClient.newBuilder()
                    .connectTimeout(Duration.ofSeconds(3))
                    .build();
        }

        private static String esc(String s) {
            if (s == null) return "";
            StringBuilder o = new StringBuilder();
            for (int i = 0; i < s.length(); i++) {
                char c = s.charAt(i);
                switch (c) {
                    case '"' -> o.append("\\\"");
                    case '\\' -> o.append("\\\\");
                    case '\n' -> o.append("\\n");
                    case '\r' -> o.append("\\r");
                    case '\t' -> o.append("\\t");
                    default -> o.append(c);
                }
            }
            return o.toString();
        }

        public boolean isAvailable() {
            try {
                HttpRequest req = HttpRequest.newBuilder()
                        .uri(URI.create("http://" + host + ":" + port + "/api/tags"))
                        .timeout(Duration.ofSeconds(2))
                        .GET()
                        .build();
                HttpResponse<String> resp = client.send(req, HttpResponse.BodyHandlers.ofString());
                return resp.statusCode() == 200;
            } catch (Exception e) {
                return false;
            }
        }

        public float[] embed(String text) {
            try {
                String body = "{\"model\":\"" + embedModel + "\",\"prompt\":\"" + esc(text) + "\"}";
                HttpRequest req = HttpRequest.newBuilder()
                        .uri(URI.create("http://" + host + ":" + port + "/api/embeddings"))
                        .timeout(Duration.ofSeconds(120))
                        .header("Content-Type", "application/json")
                        .POST(HttpRequest.BodyPublishers.ofString(body, StandardCharsets.UTF_8))
                        .build();
                HttpResponse<String> resp = client.send(req, HttpResponse.BodyHandlers.ofString());
                if (resp.statusCode() != 200) return new float[0];

                String respBody = resp.body();
                int p = respBody.indexOf("\"embedding\"");
                if (p == -1) return new float[0];
                p = respBody.indexOf('[', p);
                if (p == -1) return new float[0];
                int e = p + 1;
                int depth = 1;
                while (e < respBody.length() && depth > 0) {
                    if (respBody.charAt(e) == '[') depth++;
                    else if (respBody.charAt(e) == ']') depth--;
                    e++;
                }
                return parseVec(respBody.substring(p + 1, e - 1));
            } catch (Exception e) {
                return new float[0];
            }
        }

        public String generate(String prompt) {
            try {
                String body = "{\"model\":\"" + genModel + "\"," +
                        "\"prompt\":\"" + esc(prompt) + "\"," +
                        "\"stream\":false}";
                HttpRequest req = HttpRequest.newBuilder()
                        .uri(URI.create("http://" + host + ":" + port + "/api/generate"))
                        .timeout(Duration.ofSeconds(180))
                        .header("Content-Type", "application/json")
                        .POST(HttpRequest.BodyPublishers.ofString(body, StandardCharsets.UTF_8))
                        .build();
                HttpResponse<String> resp = client.send(req, HttpResponse.BodyHandlers.ofString());
                if (resp.statusCode() != 200) {
                    return "ERROR: Ollama unavailable. Run: ollama serve";
                }
                return extractStr(resp.body(), "response");
            } catch (Exception e) {
                return "ERROR: Ollama unavailable. Run: ollama serve";
            }
        }
    }

    // =====================================================================
    //  DOCUMENT DATABASE — HNSW over real Ollama embeddings
    // =====================================================================

    public static class DocItem {
        public final int id;
        public final String title;
        public final String text;
        public final float[] emb;

        public DocItem(int id, String title, String text, float[] emb) {
            this.id = id;
            this.title = title;
            this.text = text;
            this.emb = emb;
        }
    }

    public static class DocSearchResult {
        public final float dist;
        public final DocItem item;

        public DocSearchResult(float dist, DocItem item) {
            this.dist = dist;
            this.item = item;
        }
    }

    public static class DocumentDB {
        private final Map<Integer, DocItem> store = new HashMap<>();
        private final HNSW hnsw = new HNSW(16, 200);
        private final BruteForce bf = new BruteForce();
        private int nextId = 1;
        private int dims = 0;

        public synchronized int insert(String title, String text, float[] emb) {
            if (dims == 0) dims = emb.length;
            DocItem item = new DocItem(nextId++, title, text, emb);
            store.put(item.id, item);
            VectorItem vi = new VectorItem(item.id, title, "doc", emb);
            hnsw.insert(vi, Main::cosine);
            bf.insert(vi);
            return item.id;
        }

        public synchronized List<DocSearchResult> search(float[] q, int k, float maxDist) {
            if (store.isEmpty()) return Collections.emptyList();
            List<DistPair> raw = (store.size() < 10)
                    ? bf.knn(q, k, Main::cosine)
                    : hnsw.knn(q, k, 50, Main::cosine);
            List<DocSearchResult> out = new ArrayList<>();
            for (DistPair dp : raw) {
                DocItem di = store.get(dp.id);
                if (di != null && dp.dist <= maxDist) {
                    out.add(new DocSearchResult(dp.dist, di));
                }
            }
            return out;
        }

        public synchronized boolean remove(int id) {
            if (!store.containsKey(id)) return false;
            store.remove(id);
            hnsw.remove(id);
            bf.remove(id);
            return true;
        }

        public synchronized List<DocItem> all() {
            return new ArrayList<>(store.values());
        }

        public synchronized int size() {
            return store.size();
        }

        public synchronized int getDims() {
            return dims;
        }
    }

    // =====================================================================
    //  DEMO DATA (16D categorical vectors)
    // =====================================================================

    public static void loadDemo(VectorDB db) {
        DistFn dist = getDistFn("cosine");
        db.insert("Linked List: nodes connected by pointers", "cs",
                new float[]{0.90f,0.85f,0.72f,0.68f,0.12f,0.08f,0.15f,0.10f,0.05f,0.08f,0.06f,0.09f,0.07f,0.11f,0.08f,0.06f}, dist);
        db.insert("Binary Search Tree: O(log n) search and insert", "cs",
                new float[]{0.88f,0.82f,0.78f,0.74f,0.15f,0.10f,0.08f,0.12f,0.06f,0.07f,0.08f,0.05f,0.09f,0.06f,0.07f,0.10f}, dist);
        db.insert("Dynamic Programming: memoization overlapping subproblems", "cs",
                new float[]{0.82f,0.76f,0.88f,0.80f,0.20f,0.18f,0.12f,0.09f,0.07f,0.06f,0.08f,0.07f,0.08f,0.09f,0.06f,0.07f}, dist);
        db.insert("Graph BFS and DFS: breadth and depth first traversal", "cs",
                new float[]{0.85f,0.80f,0.75f,0.82f,0.18f,0.14f,0.10f,0.08f,0.06f,0.09f,0.07f,0.06f,0.10f,0.08f,0.09f,0.07f}, dist);
        db.insert("Hash Table: O(1) lookup with collision chaining", "cs",
                new float[]{0.87f,0.78f,0.70f,0.76f,0.13f,0.11f,0.09f,0.14f,0.08f,0.07f,0.06f,0.08f,0.07f,0.10f,0.08f,0.09f}, dist);
        db.insert("Calculus: derivatives integrals and limits", "math",
                new float[]{0.12f,0.15f,0.18f,0.10f,0.91f,0.86f,0.78f,0.72f,0.08f,0.06f,0.07f,0.09f,0.07f,0.08f,0.06f,0.10f}, dist);
        db.insert("Linear Algebra: matrices eigenvalues eigenvectors", "math",
                new float[]{0.20f,0.18f,0.15f,0.12f,0.88f,0.90f,0.82f,0.76f,0.09f,0.07f,0.08f,0.06f,0.10f,0.07f,0.08f,0.09f}, dist);
        db.insert("Probability: distributions random variables Bayes theorem", "math",
                new float[]{0.15f,0.12f,0.20f,0.18f,0.84f,0.80f,0.88f,0.82f,0.07f,0.08f,0.06f,0.10f,0.09f,0.06f,0.09f,0.08f}, dist);
        db.insert("Number Theory: primes modular arithmetic RSA cryptography", "math",
                new float[]{0.22f,0.16f,0.14f,0.20f,0.80f,0.85f,0.76f,0.90f,0.08f,0.09f,0.07f,0.06f,0.08f,0.10f,0.07f,0.06f}, dist);
        db.insert("Combinatorics: permutations combinations generating functions", "math",
                new float[]{0.18f,0.20f,0.16f,0.14f,0.86f,0.78f,0.84f,0.80f,0.06f,0.07f,0.09f,0.08f,0.06f,0.09f,0.10f,0.07f}, dist);
        db.insert("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
                new float[]{0.08f,0.06f,0.09f,0.07f,0.07f,0.08f,0.06f,0.09f,0.90f,0.86f,0.78f,0.72f,0.08f,0.06f,0.09f,0.07f}, dist);
        db.insert("Sushi: vinegared rice raw fish and nori rolls", "food",
                new float[]{0.06f,0.08f,0.07f,0.09f,0.09f,0.06f,0.08f,0.07f,0.86f,0.90f,0.82f,0.76f,0.07f,0.09f,0.06f,0.08f}, dist);
        db.insert("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
                new float[]{0.09f,0.07f,0.06f,0.08f,0.08f,0.09f,0.07f,0.06f,0.82f,0.78f,0.90f,0.84f,0.09f,0.07f,0.08f,0.06f}, dist);
        db.insert("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
                new float[]{0.07f,0.09f,0.08f,0.06f,0.06f,0.07f,0.09f,0.08f,0.78f,0.82f,0.86f,0.90f,0.06f,0.08f,0.07f,0.09f}, dist);
        db.insert("Croissant: laminated pastry with buttery flaky layers", "food",
                new float[]{0.06f,0.07f,0.10f,0.09f,0.10f,0.06f,0.07f,0.10f,0.85f,0.80f,0.76f,0.82f,0.09f,0.07f,0.10f,0.06f}, dist);
        db.insert("Basketball: fast-paced shooting dribbling slam dunks", "sports",
                new float[]{0.09f,0.07f,0.08f,0.10f,0.08f,0.09f,0.07f,0.06f,0.08f,0.07f,0.09f,0.06f,0.91f,0.85f,0.78f,0.72f}, dist);
        db.insert("Football: tackles touchdowns field goals and strategy", "sports",
                new float[]{0.07f,0.09f,0.06f,0.08f,0.09f,0.07f,0.10f,0.08f,0.07f,0.09f,0.08f,0.07f,0.87f,0.89f,0.82f,0.76f}, dist);
        db.insert("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
                new float[]{0.08f,0.06f,0.09f,0.07f,0.07f,0.08f,0.06f,0.09f,0.09f,0.06f,0.07f,0.08f,0.83f,0.80f,0.88f,0.82f}, dist);
        db.insert("Chess: openings endgames tactics strategic board game", "sports",
                new float[]{0.25f,0.20f,0.22f,0.18f,0.22f,0.18f,0.20f,0.15f,0.06f,0.08f,0.07f,0.09f,0.80f,0.84f,0.78f,0.90f}, dist);
        db.insert("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
                new float[]{0.06f,0.08f,0.07f,0.09f,0.08f,0.06f,0.09f,0.07f,0.10f,0.08f,0.06f,0.07f,0.85f,0.82f,0.86f,0.80f}, dist);
    }

    // =====================================================================
    //  HTTP SERVER & ROUTER
    // =====================================================================

    private static Map<String, String> parseQueryParams(URI uri) {
        Map<String, String> map = new HashMap<>();
        String query = uri.getRawQuery();
        if (query == null || query.isEmpty()) return map;
        String[] pairs = query.split("&");
        for (String pair : pairs) {
            int idx = pair.indexOf("=");
            if (idx > 0) {
                String k = URLDecoder.decode(pair.substring(0, idx), StandardCharsets.UTF_8);
                String v = URLDecoder.decode(pair.substring(idx + 1), StandardCharsets.UTF_8);
                map.put(k, v);
            } else {
                map.put(URLDecoder.decode(pair, StandardCharsets.UTF_8), "");
            }
        }
        return map;
    }

    private static void sendResponse(HttpExchange ex, int code, String contentType, byte[] data) throws IOException {
        ex.getResponseHeaders().set("Access-Control-Allow-Origin", "*");
        ex.getResponseHeaders().set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
        ex.getResponseHeaders().set("Access-Control-Allow-Headers", "Content-Type");
        if (contentType != null) {
            ex.getResponseHeaders().set("Content-Type", contentType);
        }
        ex.sendResponseHeaders(code, data.length);
        try (OutputStream os = ex.getResponseBody()) {
            os.write(data);
        }
    }

    private static void sendJson(HttpExchange ex, int code, String json) throws IOException {
        sendResponse(ex, code, "application/json", json.getBytes(StandardCharsets.UTF_8));
    }

    public static void main(String[] args) throws IOException {
        VectorDB db = new VectorDB(DIMS);
        DocumentDB docDB = new DocumentDB();
        OllamaClient ollama = new OllamaClient("127.0.0.1", 11434);

        loadDemo(db);

        boolean ollamaUp = ollama.isAvailable();
        System.out.println("=== VectorDB Engine (Java) ===");
        System.out.println("http://localhost:8080");
        System.out.println(db.size() + " demo vectors | " + DIMS + " dims | HNSW+KD-Tree+BruteForce");
        System.out.println("Ollama: " + (ollamaUp ? "ONLINE" : "OFFLINE (install from ollama.com)"));
        if (ollamaUp) {
            System.out.println("  embed model: " + ollama.embedModel + "  gen model: " + ollama.genModel);
        }

        HttpServer server = HttpServer.create(new InetSocketAddress(8080), 0);
        server.setExecutor(Executors.newCachedThreadPool());

        server.createContext("/", exchange -> {
            try {
                String method = exchange.getRequestMethod().toUpperCase();
                URI uri = exchange.getRequestURI();
                String path = uri.getPath();

                // Handle CORS preflight
                if ("OPTIONS".equals(method)) {
                    sendResponse(exchange, 204, null, new byte[0]);
                    return;
                }

                // Serve index.html
                if ("/".equals(path) && "GET".equals(method)) {
                    File html = new File("index.html");
                    if (html.exists()) {
                        byte[] bytes = Files.readAllBytes(html.toPath());
                        sendResponse(exchange, 200, "text/html; charset=utf-8", bytes);
                    } else {
                        sendResponse(exchange, 404, "text/plain", "index.html not found".getBytes(StandardCharsets.UTF_8));
                    }
                    return;
                }

                // GET /search?v=...&k=...&metric=...&algo=...
                if ("/search".equals(path) && "GET".equals(method)) {
                    Map<String, String> qp = parseQueryParams(uri);
                    float[] q = parseVec(qp.get("v"));
                    if (q.length != DIMS) {
                        sendJson(exchange, 200, "{\"error\":\"need " + DIMS + "D vector\"}");
                        return;
                    }
                    int k = 5;
                    try { k = Integer.parseInt(qp.getOrDefault("k", "5")); } catch (Exception ignored) {}
                    String metric = qp.getOrDefault("metric", "cosine");
                    String algo = qp.getOrDefault("algo", "hnsw");

                    VectorDB.SearchOut out = db.search(q, k, metric, algo);
                    StringBuilder ss = new StringBuilder();
                    ss.append("{\"results\":[");
                    for (int i = 0; i < out.hits.size(); i++) {
                        if (i > 0) ss.append(',');
                        VectorDB.Hit h = out.hits.get(i);
                        ss.append("{\"id\":").append(h.id)
                          .append(",\"metadata\":").append(jS(h.meta))
                          .append(",\"category\":").append(jS(h.cat))
                          .append(String.format(Locale.US, ",\"distance\":%.6f", h.dist))
                          .append(",\"embedding\":").append(jVec(h.emb))
                          .append('}');
                    }
                    ss.append("],\"latencyUs\":").append(out.us)
                      .append(",\"algo\":").append(jS(out.algo))
                      .append(",\"metric\":").append(jS(out.metric))
                      .append('}');
                    sendJson(exchange, 200, ss.toString());
                    return;
                }

                // POST /insert
                if ("/insert".equals(path) && "POST".equals(method)) {
                    String body = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
                    String meta = extractStr(body, "metadata");
                    String cat = extractStr(body, "category");
                    float[] emb = extractArr(body, "embedding");

                    if (meta.isEmpty() || emb.length != DIMS) {
                        sendJson(exchange, 200, "{\"error\":\"invalid body\"}");
                        return;
                    }
                    int id = db.insert(meta, cat, emb, getDistFn("cosine"));
                    sendJson(exchange, 200, "{\"id\":" + id + "}");
                    return;
                }

                // DELETE /delete/{id}
                if (path.startsWith("/delete/") && "DELETE".equals(method)) {
                    String idStr = path.substring("/delete/".length());
                    try {
                        int id = Integer.parseInt(idStr);
                        boolean ok = db.remove(id);
                        sendJson(exchange, 200, "{\"ok\":" + ok + "}");
                    } catch (Exception e) {
                        sendJson(exchange, 200, "{\"ok\":false}");
                    }
                    return;
                }

                // GET /items
                if ("/items".equals(path) && "GET".equals(method)) {
                    List<VectorItem> items = db.all();
                    StringBuilder ss = new StringBuilder("[");
                    for (int i = 0; i < items.size(); i++) {
                        if (i > 0) ss.append(',');
                        VectorItem v = items.get(i);
                        ss.append("{\"id\":").append(v.id)
                          .append(",\"metadata\":").append(jS(v.metadata))
                          .append(",\"category\":").append(jS(v.category))
                          .append(",\"embedding\":").append(jVec(v.emb))
                          .append('}');
                    }
                    ss.append(']');
                    sendJson(exchange, 200, ss.toString());
                    return;
                }

                // GET /benchmark
                if ("/benchmark".equals(path) && "GET".equals(method)) {
                    Map<String, String> qp = parseQueryParams(uri);
                    float[] q = parseVec(qp.get("v"));
                    if (q.length != DIMS) {
                        sendJson(exchange, 200, "{\"error\":\"need " + DIMS + "D vector\"}");
                        return;
                    }
                    int k = 5;
                    try { k = Integer.parseInt(qp.getOrDefault("k", "5")); } catch (Exception ignored) {}
                    String metric = qp.getOrDefault("metric", "cosine");
                    VectorDB.BenchOut b = db.benchmark(q, k, metric);

                    String res = "{\"bruteforceUs\":" + b.bfUs +
                            ",\"kdtreeUs\":" + b.kdUs +
                            ",\"hnswUs\":" + b.hnswUs +
                            ",\"itemCount\":" + b.n + "}";
                    sendJson(exchange, 200, res);
                    return;
                }

                // GET /hnsw-info
                if ("/hnsw-info".equals(path) && "GET".equals(method)) {
                    HNSW.GraphInfo gi = db.hnswInfo();
                    StringBuilder ss = new StringBuilder();
                    ss.append("{\"topLayer\":").append(gi.topLayer)
                      .append(",\"nodeCount\":").append(gi.nodeCount)
                      .append(",\"nodesPerLayer\":[");
                    for (int i = 0; i < gi.nodesPerLayer.size(); i++) {
                        if (i > 0) ss.append(',');
                        ss.append(gi.nodesPerLayer.get(i));
                    }
                    ss.append("],\"edgesPerLayer\":[");
                    for (int i = 0; i < gi.edgesPerLayer.size(); i++) {
                        if (i > 0) ss.append(',');
                        ss.append(gi.edgesPerLayer.get(i));
                    }
                    ss.append("],\"nodes\":[");
                    for (int i = 0; i < gi.nodes.size(); i++) {
                        if (i > 0) ss.append(',');
                        HNSW.NodeView n = gi.nodes.get(i);
                        ss.append("{\"id\":").append(n.id)
                          .append(",\"metadata\":").append(jS(n.metadata))
                          .append(",\"category\":").append(jS(n.category))
                          .append(",\"maxLyr\":").append(n.maxLyr)
                          .append('}');
                    }
                    ss.append("],\"edges\":[");
                    for (int i = 0; i < gi.edges.size(); i++) {
                        if (i > 0) ss.append(',');
                        HNSW.EdgeView e = gi.edges.get(i);
                        ss.append("{\"src\":").append(e.src)
                          .append(",\"dst\":").append(e.dst)
                          .append(",\"lyr\":").append(e.lyr)
                          .append('}');
                    }
                    ss.append("]}");
                    sendJson(exchange, 200, ss.toString());
                    return;
                }

                // POST /doc/insert
                if ("/doc/insert".equals(path) && "POST".equals(method)) {
                    String body = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
                    String title = extractStr(body, "title");
                    String text = extractStr(body, "text");
                    if (title.isEmpty() || text.isEmpty()) {
                        sendJson(exchange, 200, "{\"error\":\"need title and text\"}");
                        return;
                    }

                    List<String> chunks = chunkText(text, 250, 30);
                    List<Integer> ids = new ArrayList<>();

                    for (int i = 0; i < chunks.size(); i++) {
                        float[] emb = ollama.embed(chunks.get(i));
                        if (emb.length == 0) {
                            sendJson(exchange, 200, "{\"error\":\"Ollama unavailable. " +
                                    "Install from https://ollama.com then run: " +
                                    "ollama pull nomic-embed-text && ollama pull llama3.2\"}");
                            return;
                        }
                        String chunkTitle = (chunks.size() > 1)
                                ? title + " [" + (i + 1) + "/" + chunks.size() + "]"
                                : title;
                        ids.add(docDB.insert(chunkTitle, chunks.get(i), emb));
                    }

                    StringBuilder ss = new StringBuilder();
                    ss.append("{\"ids\":[");
                    for (int i = 0; i < ids.size(); i++) {
                        if (i > 0) ss.append(',');
                        ss.append(ids.get(i));
                    }
                    ss.append("],\"chunks\":").append(chunks.size())
                      .append(",\"dims\":").append(docDB.getDims())
                      .append('}');
                    sendJson(exchange, 200, ss.toString());
                    return;
                }

                // DELETE /doc/delete/{id}
                if (path.startsWith("/doc/delete/") && "DELETE".equals(method)) {
                    String idStr = path.substring("/doc/delete/".length());
                    try {
                        int id = Integer.parseInt(idStr);
                        boolean ok = docDB.remove(id);
                        sendJson(exchange, 200, "{\"ok\":" + ok + "}");
                    } catch (Exception e) {
                        sendJson(exchange, 200, "{\"ok\":false}");
                    }
                    return;
                }

                // GET /doc/list
                if ("/doc/list".equals(path) && "GET".equals(method)) {
                    List<DocItem> docs = docDB.all();
                    StringBuilder ss = new StringBuilder("[");
                    for (int i = 0; i < docs.size(); i++) {
                        if (i > 0) ss.append(',');
                        DocItem d = docs.get(i);
                        String preview = d.text.length() > 120 ? d.text.substring(0, 120) + "…" : d.text;
                        int words = d.text.trim().isEmpty() ? 0 : d.text.trim().split("\\s+").length;
                        ss.append("{\"id\":").append(d.id)
                          .append(",\"title\":").append(jS(d.title))
                          .append(",\"preview\":").append(jS(preview))
                          .append(",\"words\":").append(words)
                          .append('}');
                    }
                    ss.append(']');
                    sendJson(exchange, 200, ss.toString());
                    return;
                }

                // POST /doc/search
                if ("/doc/search".equals(path) && "POST".equals(method)) {
                    String body = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
                    String question = extractStr(body, "question");
                    int k = extractInt(body, "k", 3);
                    if (question.isEmpty()) {
                        sendJson(exchange, 200, "{\"error\":\"need question\"}");
                        return;
                    }

                    float[] qEmb = ollama.embed(question);
                    if (qEmb.length == 0) {
                        sendJson(exchange, 200, "{\"error\":\"Ollama unavailable\"}");
                        return;
                    }

                    List<DocSearchResult> hits = docDB.search(qEmb, k, 0.7f);
                    StringBuilder ss = new StringBuilder("{\"contexts\":[");
                    for (int i = 0; i < hits.size(); i++) {
                        if (i > 0) ss.append(',');
                        DocSearchResult ds = hits.get(i);
                        ss.append("{\"id\":").append(ds.item.id)
                          .append(",\"title\":").append(jS(ds.item.title))
                          .append(String.format(Locale.US, ",\"distance\":%.4f", ds.dist))
                          .append('}');
                    }
                    ss.append("]}");
                    sendJson(exchange, 200, ss.toString());
                    return;
                }

                // POST /doc/ask
                if ("/doc/ask".equals(path) && "POST".equals(method)) {
                    String body = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
                    String question = extractStr(body, "question");
                    int k = extractInt(body, "k", 3);
                    if (question.isEmpty()) {
                        sendJson(exchange, 200, "{\"error\":\"need question\"}");
                        return;
                    }

                    float[] qEmb = ollama.embed(question);
                    if (qEmb.length == 0) {
                        sendJson(exchange, 200, "{\"error\":\"Ollama unavailable\"}");
                        return;
                    }

                    List<DocSearchResult> hits = docDB.search(qEmb, k, 0.7f);

                    StringBuilder ctx = new StringBuilder();
                    for (int i = 0; i < hits.size(); i++) {
                        DocSearchResult ds = hits.get(i);
                        ctx.append("[").append(i + 1).append("] ")
                           .append(ds.item.title).append(":\n")
                           .append(ds.item.text).append("\n\n");
                    }

                    String prompt = "You are a helpful assistant. Answer the user's question directly. "
                            + "Use the provided context if it contains relevant information. "
                            + "If it doesn't, just use your own general knowledge. "
                            + "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like 'the context doesn't mention'. "
                            + "Just answer the question naturally.\n\n"
                            + "Context:\n" + ctx
                            + "Question: " + question + "\n\n"
                            + "Answer:";

                    String answer = ollama.generate(prompt);

                    StringBuilder ss = new StringBuilder();
                    ss.append("{\"answer\":").append(jS(answer))
                      .append(",\"model\":").append(jS(ollama.genModel))
                      .append(",\"contexts\":[");
                    for (int i = 0; i < hits.size(); i++) {
                        if (i > 0) ss.append(',');
                        DocSearchResult ds = hits.get(i);
                        ss.append("{\"id\":").append(ds.item.id)
                          .append(",\"title\":").append(jS(ds.item.title))
                          .append(",\"text\":").append(jS(ds.item.text))
                          .append(String.format(Locale.US, ",\"distance\":%.4f", ds.dist))
                          .append('}');
                    }
                    ss.append("],\"docCount\":").append(docDB.size()).append('}');
                    sendJson(exchange, 200, ss.toString());
                    return;
                }

                // GET /status
                if ("/status".equals(path) && "GET".equals(method)) {
                    boolean up = ollama.isAvailable();
                    String ss = "{\"ollamaAvailable\":" + (up ? "true" : "false") +
                            ",\"embedModel\":" + jS(ollama.embedModel) +
                            ",\"genModel\":" + jS(ollama.genModel) +
                            ",\"docCount\":" + docDB.size() +
                            ",\"docDims\":" + docDB.getDims() +
                            ",\"demoDims\":" + DIMS +
                            ",\"demoCount\":" + db.size() + "}";
                    sendJson(exchange, 200, ss);
                    return;
                }

                // GET /stats
                if ("/stats".equals(path) && "GET".equals(method)) {
                    String ss = "{\"count\":" + db.size() +
                            ",\"dims\":" + DIMS +
                            ",\"algorithms\":[\"bruteforce\",\"kdtree\",\"hnsw\"]" +
                            ",\"metrics\":[\"euclidean\",\"cosine\",\"manhattan\"]}";
                    sendJson(exchange, 200, ss);
                    return;
                }

                sendResponse(exchange, 404, "application/json", "{\"error\":\"not found\"}".getBytes(StandardCharsets.UTF_8));
            } catch (Exception ex) {
                ex.printStackTrace();
                sendResponse(exchange, 500, "application/json", ("{\"error\":" + jS(ex.getMessage()) + "}").getBytes(StandardCharsets.UTF_8));
            }
        });

        server.start();
        System.out.println("Server listening on port 8080...");
    }
}
