"""
CSN 检索用「常见伪代码」查询增强：纯本地规则，无 API。
将 Javadoc 式英文查询中与 Java 常见 API 相关的提示拼成短代码骨架，再与原文一并送入 UniXcoder 编码。
同时提供 clean_java_query()：清洗 Javadoc 标签/HTML/句末冗余，提升 bi-encoder 查询质量。

改进（v2）：
- 规则表从 24 条扩至 ~60 条，补全集合/并发/IO/字符串/日期/数字等高频 Java 场景
- 新增动词签名启发：从 NL 动词猜测 Java 方法签名骨架，追加 `// method: xxx` 注释
- 新增 Java 类名检测：直接提取 NL 中 CamelCase 标识符作为 import 提示
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

# ---- Javadoc / HTML 清洗 ----

_JAVADOC_TAG = re.compile(
    r"@(?:param|return|returns|throws|exception|see|since|deprecated|author|version|serial"
    r"|serialField|serialData|link|linkplain|inheritDoc|value|code|docRoot|literal|"
    r"Override|SuppressWarnings|Nullable|NonNull)\b[^\n]*",
    re.IGNORECASE,
)
_HTML_TAG = re.compile(r"<[^>]{1,120}>")
_MULTI_WS = re.compile(r"\s{2,}")


def clean_java_query(text: str, max_chars: int = 300) -> str:
    """
    去除 Javadoc @tag、HTML 标签、多余空白；截到 max_chars 保留核心语义句。
    不改变 NL 词序，只做噪声剥离，适合在送入 bi-encoder 前调用。
    """
    s = (text or "").strip()
    s = _JAVADOC_TAG.sub(" ", s)
    s = _HTML_TAG.sub(" ", s)
    dot = s.find(". ")
    if dot > 40:
        s = s[: dot + 1]
    s = _MULTI_WS.sub(" ", s).strip()
    if len(s) > max_chars:
        s = s[:max_chars].rstrip()
    return s if s else (text or "").strip()


# ---- 规则表：(关键词子串组, 伪代码骨架) ----
# 全部关键词均需出现在规范化查询中才附加，降低乱匹配率。

_RULES: List[Tuple[Tuple[str, ...], str]] = [
    # --- I/O ---
    (
        ("file", "read"),
        "BufferedReader br = new BufferedReader(new FileReader(path)); String line;",
    ),
    (("file", "write"), "Files.write(path, bytes, StandardOpenOption.CREATE);"),
    (("file", "copy"), "Files.copy(src, dst, StandardCopyOption.REPLACE_EXISTING);"),
    (("file", "delete"), "Files.deleteIfExists(path);"),
    (("file", "exist"), "Files.exists(path);"),
    (("file", "list"), "try (Stream<Path> s = Files.list(dir)) { s.forEach(p -> {}); }"),
    (("path", "walk"), "Files.walk(path).filter(Files::isRegularFile).forEach(p -> {});"),
    (("directory", "creat"), "Files.createDirectories(path);"),
    (("stream", "read"), "InputStream in = ...; byte[] buf = new byte[8192]; int n;"),
    (("stream", "write"), "OutputStream out = ...; out.write(bytes, 0, len);"),
    (("stream", "close"), "try (InputStream in = ...) { }  // auto-close"),
    (("reader", "line"), "BufferedReader br = ...; String line; while ((line = br.readLine()) != null) {}"),
    (("writer", "flush"), "BufferedWriter bw = new BufferedWriter(new FileWriter(f)); bw.write(s); bw.flush();"),
    (("input", "stream", "byte"), "DataInputStream dis = new DataInputStream(in); int val = dis.readInt();"),
    (("output", "stream", "byte"), "DataOutputStream dos = new DataOutputStream(out); dos.writeInt(val);"),
    (("stream", "filter"), ".stream().filter(x ->).map(x ->).collect(Collectors.toList());"),
    (("channel", "buffer"), "FileChannel fc = FileChannel.open(path, READ); ByteBuffer bb = ByteBuffer.allocate(n);"),
    (("zip", "compress"), "ZipOutputStream zos = new ZipOutputStream(out); zos.putNextEntry(new ZipEntry(name));"),
    (("zip", "extract"), "ZipInputStream zis = new ZipInputStream(in); ZipEntry entry;"),
    # --- 网络 ---
    (("socket", "connect"), "Socket s = new Socket(host, port); InputStream is = s.getInputStream();"),
    (("url", "http"), "HttpURLConnection c = (HttpURLConnection) new URL(url).openConnection();"),
    (("http", "get"), "HttpClient client = HttpClient.newHttpClient(); HttpRequest req = HttpRequest.newBuilder().uri(URI.create(url)).GET().build();"),
    (("http", "post"), "HttpRequest req = HttpRequest.newBuilder().uri(uri).POST(HttpRequest.BodyPublishers.ofString(body)).build();"),
    (("http", "response"), "HttpResponse<String> resp = client.send(req, HttpResponse.BodyHandlers.ofString());"),
    (("server", "socket"), "ServerSocket ss = new ServerSocket(port); Socket client = ss.accept();"),
    # --- 集合 ---
    (("list", "element"), "List<T> list = new ArrayList<>(); list.add(e); for (T x : list) {}"),
    (("list", "sort"), "Collections.sort(list); // or list.sort(Comparator.naturalOrder())"),
    (("list", "sublist"), "List<T> sub = list.subList(fromIndex, toIndex);"),
    (("map", "key"), "Map<K,V> m = new HashMap<>(); m.put(k,v); m.get(k);"),
    (("map", "entry"), "for (Map.Entry<K,V> e : map.entrySet()) { e.getKey(); e.getValue(); }"),
    (("map", "merge"), "map.merge(key, val, (old, v) -> old + v);"),
    (("set", "element"), "Set<T> s = new HashSet<>(); s.add(x); s.contains(x);"),
    (("queue", "offer"), "Queue<T> q = new LinkedList<>(); q.offer(e); T head = q.poll();"),
    (("deque", "push"), "Deque<T> dq = new ArrayDeque<>(); dq.push(e); dq.pop();"),
    (("stack", "push"), "Deque<T> stack = new ArrayDeque<>(); stack.push(e); T top = stack.peek();"),
    (("priority", "queue"), "PriorityQueue<T> pq = new PriorityQueue<>(Comparator.naturalOrder()); pq.offer(e); pq.poll();"),
    (("iterator", "next"), "Iterator<T> it = c.iterator(); while (it.hasNext()) { it.next(); }"),
    (("sort", "compar"), "Collections.sort(list, Comparator.comparing(T::getField));"),
    (("array", "index"), "T[] a = ...; for (int i = 0; i < a.length; i++) {}"),
    (("array", "copy"), "T[] copy = Arrays.copyOf(src, newLength);"),
    (("array", "sort"), "Arrays.sort(arr); // or Arrays.sort(arr, from, to)"),
    (("stream", "collect"), ".stream().filter(...).map(...).collect(Collectors.toList());"),
    (("stream", "group"), ".stream().collect(Collectors.groupingBy(T::getKey));"),
    (("optional", "present"), "Optional<T> opt = Optional.ofNullable(val); opt.ifPresent(v -> {}); opt.orElse(def);"),
    # --- 字符串 ---
    (("string", "split"), "String[] parts = str.split(regex); // or str.split(delim, limit)"),
    (("string", "join"), "String s = String.join(\", \", list); // or Collectors.joining(\", \")"),
    (("string", "format"), "String s = String.format(\"%s=%d\", key, val);"),
    (("string", "replac"), "String s = str.replace(old, newStr); // or replaceAll(regex, repl)"),
    (("string", "contains"), "str.contains(sub); str.startsWith(prefix); str.endsWith(suffix);"),
    (("string", "trim"), "str.trim(); // or str.strip() (Unicode-aware, Java 11+)"),
    (("string", "convert"), "String.valueOf(n); Integer.parseInt(s); Double.parseDouble(s);"),
    (("string", "upper"), "str.toUpperCase(Locale.ROOT); str.toLowerCase(Locale.ROOT);"),
    (("charset", "encod"), "StandardCharsets.UTF_8; new String(bytes, UTF_8); str.getBytes(UTF_8);"),
    (("base64", "encod"), "Base64.getEncoder().encodeToString(bytes); Base64.getDecoder().decode(s);"),
    # --- 数字 / 数学 ---
    (("number", "format"), "NumberFormat nf = NumberFormat.getInstance(); nf.format(num);"),
    (("integer", "parse"), "int n = Integer.parseInt(s); long l = Long.parseLong(s);"),
    (("double", "parse"), "double d = Double.parseDouble(s);"),
    (("math", "random"), "double r = Math.random(); // or ThreadLocalRandom.current().nextInt(n)"),
    (("big", "decimal"), "BigDecimal bd = new BigDecimal(\"1.23\"); bd.add(other).setScale(2, HALF_UP);"),
    # --- 日期时间 ---
    (("date", "format"), "DateTimeFormatter fmt = DateTimeFormatter.ofPattern(\"yyyy-MM-dd\"); LocalDate.now().format(fmt);"),
    (("date", "parse"), "LocalDate d = LocalDate.parse(str, fmt); Instant i = Instant.parse(iso);"),
    (("timestamp", "current"), "Instant now = Instant.now(); long ms = System.currentTimeMillis();"),
    (("date", "compar"), "d1.isBefore(d2); d1.isAfter(d2); Duration.between(t1, t2).toMillis();"),
    # --- 并发 ---
    (("thread", "run"), "ExecutorService ex = Executors.newFixedThreadPool(n); ex.submit(() -> {});"),
    (("synchron", "lock"), "synchronized (lock) {} // or ReentrantLock lock = new ReentrantLock();"),
    (("concurrent", "map"), "ConcurrentHashMap<K,V> m = new ConcurrentHashMap<>(); m.computeIfAbsent(k, k2 -> new V());"),
    (("future", "async"), "CompletableFuture<T> f = CompletableFuture.supplyAsync(() -> compute()); f.thenApply(r -> {});"),
    (("atomic", "counter"), "AtomicInteger cnt = new AtomicInteger(0); cnt.incrementAndGet();"),
    (("countdown", "latch"), "CountDownLatch latch = new CountDownLatch(n); latch.countDown(); latch.await();"),
    (("semaphore", "acquir"), "Semaphore sem = new Semaphore(permits); sem.acquire(); sem.release();"),
    (("thread", "sleep"), "Thread.sleep(millis); // or TimeUnit.SECONDS.sleep(n)"),
    (("volatile", "field"), "private volatile boolean running = true;"),
    # --- 序列化 / JSON / XML ---
    (("json", "parse"), "JsonParser.parseString(s).getAsJsonObject(); // or new JSONObject(s)"),
    (("json", "object"), "JsonObject obj = new JsonObject(); obj.addProperty(\"key\", val);"),
    (("xml", "parse"), "DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(in);"),
    (("xml", "element"), "Element e = doc.createElement(tag); e.setAttribute(attr, val); parent.appendChild(e);"),
    (("serial", "object"), "ObjectOutputStream oos = new ObjectOutputStream(out); oos.writeObject(obj);"),
    (("deserializ", "object"), "ObjectInputStream ois = new ObjectInputStream(in); T obj = (T) ois.readObject();"),
    # --- 数据库 ---
    (("sql", "query"), "PreparedStatement ps = conn.prepareStatement(sql); ResultSet rs = ps.executeQuery();"),
    (("sql", "update"), "PreparedStatement ps = conn.prepareStatement(sql); ps.setString(1, val); ps.executeUpdate();"),
    (("transaction", "commit"), "conn.setAutoCommit(false); try { ...; conn.commit(); } catch (Exception e) { conn.rollback(); }"),
    (("result", "set"), "while (rs.next()) { String v = rs.getString(\"col\"); int n = rs.getInt(\"id\"); }"),
    # --- 反射 ---
    (("reflect", "class"), "Class<?> c = Class.forName(name); Method m = c.getDeclaredMethod(methodName, paramTypes);"),
    (("reflect", "field"), "Field f = clazz.getDeclaredField(name); f.setAccessible(true); Object val = f.get(obj);"),
    (("reflect", "invoke"), "Method m = clazz.getMethod(name, paramTypes); Object result = m.invoke(instance, args);"),
    # --- 加密 / 哈希 ---
    (("digest", "hash"), "MessageDigest md = MessageDigest.getInstance(\"SHA-256\"); byte[] h = md.digest(bytes);"),
    (("cipher", "encrypt"), "Cipher c = Cipher.getInstance(\"AES/CBC/PKCS5Padding\"); c.init(Cipher.ENCRYPT_MODE, key, iv); c.doFinal(data);"),
    (("hmac", "sign"), "Mac mac = Mac.getInstance(\"HmacSHA256\"); mac.init(secretKey); byte[] sig = mac.doFinal(data);"),
    # --- 正则 ---
    (("regex", "match"), "Pattern p = Pattern.compile(regex); Matcher m = p.matcher(text); m.find();"),
    (("regex", "replac"), "String s = text.replaceAll(regex, replacement);"),
    (("regex", "group"), "Matcher m = Pattern.compile(regex).matcher(s); if (m.matches()) { m.group(1); }"),
    # --- 克隆 / 比较 ---
    (("clone", "object"), "protected Object clone() throws CloneNotSupportedException { return super.clone(); }"),
    (("comparable", "compar"), "public int compareTo(T other) { return Integer.compare(this.val, other.val); }"),
    (("comparator", "sort"), "list.sort(Comparator.comparing(T::getField).thenComparing(T::getOther));"),
    # --- 缓冲 ---
    (("buffer", "byte"), "ByteBuffer buf = ByteBuffer.allocate(n); buf.put(b); buf.flip(); buf.get(arr);"),
    (("string", "builder"), "StringBuilder sb = new StringBuilder(); sb.append(s); sb.toString();"),
    (("string", "buffer"), "StringBuffer sb = new StringBuffer(); sb.append(s); // thread-safe"),
]


# ---- 动词签名启发：从常见 Java 动词推断方法骨架 ----

_VERB_HINTS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(parse|deserializ)\b"), "// method: public T parse(String input)"),
    (re.compile(r"\b(serializ|marshal|encode)\b"), "// method: public String serialize(T obj)"),
    (re.compile(r"\b(convert|transform|map)\b"), "// method: public R convert(S source)"),
    (re.compile(r"\b(create|build|construct|instantiat)\b"), "// method: public T create(...)"),
    (re.compile(r"\b(get|retrieve|fetch|obtain|find|look.?up)\b"), "// method: public T get(K key)"),
    (re.compile(r"\b(set|put|store|save|persist)\b"), "// method: public void set(K key, V value)"),
    (re.compile(r"\b(add|append|insert|push|enqueue)\b"), "// method: public void add(T element)"),
    (re.compile(r"\b(remove|delete|pop|dequeue|clear)\b"), "// method: public void remove(T element)"),
    (re.compile(r"\b(check|validat|verify|test)\b"), "// method: public boolean isValid(T input)"),
    (re.compile(r"\b(list|collect|gather|enumerate)\b"), "// method: public List<T> list()"),
    (re.compile(r"\b(sort|order|rank)\b"), "// method: public List<T> sort(List<T> list)"),
    (re.compile(r"\b(calculat|comput|evaluat|count|sum)\b"), "// method: public R calculate(...)"),
    (re.compile(r"\b(send|publish|emit|dispatch)\b"), "// method: public void send(T message)"),
    (re.compile(r"\b(receiv|consum|accept|listen)\b"), "// method: public T receive()"),
    (re.compile(r"\b(close|shutdown|destroy|releas|stop)\b"), "// method: public void close() throws IOException"),
    (re.compile(r"\b(open|init|start|connect)\b"), "// method: public void open(...)"),
    (re.compile(r"\b(format|render|print|display)\b"), "// method: public String format(T value)"),
    (re.compile(r"\b(split|tokeniz|divid|partition)\b"), "// method: public String[] split(String s, String delim)"),
    (re.compile(r"\b(join|merg|concat|combin)\b"), "// method: public String join(Collection<String> parts, String sep)"),
    (re.compile(r"\b(compress|zip|pack)\b"), "// method: public byte[] compress(byte[] data)"),
    (re.compile(r"\b(decompress|unzip|unpack|extract)\b"), "// method: public byte[] decompress(byte[] data)"),
    (re.compile(r"\b(hash|digest|checksum)\b"), "// method: public byte[] hash(byte[] input)"),
    (re.compile(r"\b(encrypt|encipher)\b"), "// method: public byte[] encrypt(byte[] plain, Key key)"),
    (re.compile(r"\b(decrypt|decipher)\b"), "// method: public byte[] decrypt(byte[] cipher, Key key)"),
    (re.compile(r"\b(read|load|import)\b"), "// method: public T read(InputStream in)"),
    (re.compile(r"\b(write|dump|export)\b"), "// method: public void write(T data, OutputStream out)"),
    (re.compile(r"\b(copy|clone|duplicate)\b"), "// method: public T copy(T source)"),
    (re.compile(r"\b(compil|execut|evaluat|run)\b"), "// method: public R execute(String code)"),
    (re.compile(r"\b(register|subscrib|attach|bind)\b"), "// method: public void register(Listener l)"),
    (re.compile(r"\b(unregister|unsubscrib|detach|unbind)\b"), "// method: public void unregister(Listener l)"),
]

# Java CamelCase 类名检测：匹配 NL 中形如 InputStream / BufferedReader / HttpClient 的词
_CAMEL_CLASS = re.compile(r"\b([A-Z][a-z]+(?:[A-Z][a-z]*)+)\b")

# 已知 Java 关键字（过滤误匹配）
_NOT_CLASS = frozenset({
    "This", "The", "In", "If", "It", "For", "By", "An", "On",
    "As", "At", "Of", "To", "Or", "And", "With", "From", "When",
    "True", "False", "Null", "New", "Get", "Set", "Is",
})


def _extract_java_classes(nl: str) -> List[str]:
    """从 NL 中提取 CamelCase Java 类/接口名（过滤普通英文首字母大写词）。"""
    found = []
    for m in _CAMEL_CLASS.finditer(nl):
        name = m.group(1)
        if name not in _NOT_CLASS:
            found.append(name)
    return list(dict.fromkeys(found))  # 去重保序


def _normalize(q: str) -> str:
    s = q.lower()
    s = re.sub(r"\s+", " ", s)
    return s


def build_pseudo_augmented_query(nl_query: str, max_extra_chars: int = 380) -> str:
    """
    在原始 NL 后依次附加：
    1. 规则命中的 Java API 伪代码骨架（_RULES）
    2. 动词签名启发注释（_VERB_HINTS，最多 1 条避免冗余）
    3. NL 中检测到的 Java 类名（// classes: Xxx, Yyy）
    若无任何内容命中，返回原文。总额外字符不超过 max_extra_chars。
    """
    raw = (nl_query or "").strip()
    if not raw:
        return raw

    norm = _normalize(raw)
    seen: set[str] = set()
    chunks: List[str] = []

    # --- 规则伪代码 ---
    for keys, stub in _RULES:
        if all(k in norm for k in keys):
            if stub not in seen:
                seen.add(stub)
                chunks.append(stub)

    # --- 动词签名（第一个匹配即止，避免堆叠） ---
    verb_hint: Optional[str] = None
    for pattern, hint in _VERB_HINTS:
        if pattern.search(norm):
            verb_hint = hint
            break
    if verb_hint:
        chunks.append(verb_hint)

    # --- Java 类名提示 ---
    java_classes = _extract_java_classes(raw)
    if java_classes:
        chunks.append("// classes: " + ", ".join(java_classes[:5]))

    if not chunks:
        return raw

    extra = " \n// pseudo-stubs:\n" + "\n".join(chunks)
    if len(extra) > max_extra_chars:
        extra = extra[: max_extra_chars - 3] + "..."

    return raw + extra


def augment_if_enabled(nl_query: str, enabled: bool, max_extra_chars: int = 380) -> str:
    if not enabled:
        return (nl_query or "").strip()
    return build_pseudo_augmented_query(nl_query, max_extra_chars=max_extra_chars)
