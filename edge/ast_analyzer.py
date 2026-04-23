"""AST-based code analysis and hotspot detection."""

import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

try:
    from tree_sitter import Language, Parser, Node
except ImportError:
    Language = Parser = Node = None

from shared.schemas import CodeFragment, CodeLanguage


@dataclass
class Hotspot:
    """Represents a code hotspot worthy of deeper analysis."""
    
    fragment: CodeFragment
    hotspot_score: float  # 0.0-1.0, higher = more suspicious
    complexity_factors: Dict[str, float]
    reason: str
    
    def __lt__(self, other):
        """For sorting by hotspot score."""
        return self.hotspot_score < other.hotspot_score


class ASTAnalyzer:
    """Multi-language AST analyzer and hotspot detector."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize AST analyzer.
        
        Args:
            config: Configuration dict with thresholds and settings
        """
        self.config = config or {}
        self.thresholds = self.config.get('complexity', {})
        self.weights = self.config.get('hotspot_weights', {
            'cyclomatic_complexity': 0.3,
            'nesting_depth': 0.2,
            'function_length': 0.15,
            'exception_handling_missing': 0.15,
            'null_checks_missing': 0.1,
            'resource_management': 0.1
        })
        
        # Tree-sitter parsers (lazy loaded)
        self._parsers: Dict[str, Parser] = {}
    
    def _get_parser(self, language: CodeLanguage) -> Optional[Parser]:
        """Get or create tree-sitter parser for language.
        
        Note: In production, you need to build tree-sitter libraries.
        This is a simplified version that falls back to regex-based analysis.
        """
        if language.value in self._parsers:
            return self._parsers[language.value]
        
        if Language is None or Parser is None:
            # Tree-sitter not available, return None for fallback
            return None
        
        try:
            # In real implementation, load compiled language library
            # Language.build_library('build/languages.so', [...])
            # For now, return None to trigger fallback
            return None
        except Exception:
            return None
    
    def analyze_file(
        self,
        file_path: str,
        language: Optional[CodeLanguage] = None
    ) -> List[Hotspot]:
        """Analyze a file and identify hotspots.
        
        Args:
            file_path: Path to source file
            language: Programming language (auto-detected if None)
        
        Returns:
            List of hotspots sorted by score (highest first)
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        content = path.read_text(encoding='utf-8', errors='ignore')
        
        if language is None:
            language = self._detect_language(file_path)
        
        return self.analyze_code(content, file_path, language)
    
    def analyze_code(
        self,
        code: str,
        file_path: str,
        language: CodeLanguage
    ) -> List[Hotspot]:
        """Analyze code content and identify hotspots.
        
        Args:
            code: Source code content
            file_path: File path for reference
            language: Programming language
        
        Returns:
            List of hotspots sorted by score (highest first)
        """
        parser = self._get_parser(language)
        
        if parser is not None:
            # Use tree-sitter for precise analysis
            hotspots = self._analyze_with_tree_sitter(
                code, file_path, language, parser
            )
        else:
            # Fallback to regex-based analysis
            hotspots = self._analyze_with_regex(
                code, file_path, language
            )
        
        # Sort by hotspot score (highest first)
        hotspots.sort(reverse=True)
        
        return hotspots
    
    def _analyze_with_tree_sitter(
        self,
        code: str,
        file_path: str,
        language: CodeLanguage,
        parser: Parser
    ) -> List[Hotspot]:
        """Precise analysis using tree-sitter (stub for now)."""
        # This would use tree-sitter to parse and analyze
        # For brevity, falling back to regex
        return self._analyze_with_regex(code, file_path, language)
    
    def _analyze_with_regex(
        self,
        code: str,
        file_path: str,
        language: CodeLanguage
    ) -> List[Hotspot]:
        """Regex-based analysis (fallback when tree-sitter unavailable)."""
        hotspots = []
        
        # Split into functions/methods
        functions = self._extract_functions_regex(code, language)
        
        for func_info in functions:
            fragment = CodeFragment(
                file_path=file_path,
                start_line=func_info['start_line'],
                end_line=func_info['end_line'],
                content=func_info['content'],
                language=language,
                function_name=func_info['name']
            )
            
            # Calculate complexity factors
            factors = self._calculate_complexity_factors(
                func_info['content'],
                language
            )
            
            # Calculate hotspot score
            score = self._calculate_hotspot_score(factors)
            
            # Generate reason
            reason = self._generate_hotspot_reason(factors)
            
            # Only include if score is significant
            if score > 0.3:
                hotspots.append(Hotspot(
                    fragment=fragment,
                    hotspot_score=score,
                    complexity_factors=factors,
                    reason=reason
                ))
        
        return hotspots
    
    def _extract_functions_regex(
        self,
        code: str,
        language: CodeLanguage
    ) -> List[Dict[str, Any]]:
        """Extract functions using regex patterns."""
        functions = []
        lines = code.split('\n')
        
        if language == CodeLanguage.PYTHON:
            pattern = r'^\s*def\s+(\w+)\s*\('
        elif language == CodeLanguage.JAVA:
            pattern = r'^\s*(public|private|protected)?\s*\w+\s+(\w+)\s*\('
        elif language in (CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT):
            pattern = r'^\s*function\s+(\w+)\s*\(|^\s*(\w+)\s*=\s*\([^)]*\)\s*=>|^\s*(\w+)\s*\([^)]*\)\s*\{'
        elif language in (CodeLanguage.C, CodeLanguage.CPP):
            pattern = r'^\s*\w+\s+(\w+)\s*\([^)]*\)\s*\{'
        else:
            pattern = r'^\s*function\s+(\w+)'
        
        i = 0
        while i < len(lines):
            match = re.match(pattern, lines[i])
            if match:
                func_name = next((g for g in match.groups() if g), 'unknown')
                start_line = i + 1
                
                # Find end of function (simplified)
                end_line = self._find_function_end(lines, i, language)
                
                if end_line > start_line:
                    content = '\n'.join(lines[i:end_line])
                    functions.append({
                        'name': func_name,
                        'start_line': start_line,
                        'end_line': end_line,
                        'content': content
                    })
                    i = end_line
                else:
                    i += 1
            else:
                i += 1
        
        return functions
    
    def _find_function_end(
        self,
        lines: List[str],
        start: int,
        language: CodeLanguage
    ) -> int:
        """Find the end line of a function (simplified heuristic)."""
        if language == CodeLanguage.PYTHON:
            # Python: ends when indentation returns to original or less
            base_indent = len(lines[start]) - len(lines[start].lstrip())
            for i in range(start + 1, len(lines)):
                line = lines[i]
                if line.strip() and not line.startswith(' ' * (base_indent + 1)):
                    return i
            return len(lines)
        else:
            # C-like: count braces
            brace_count = 0
            for i in range(start, len(lines)):
                brace_count += lines[i].count('{') - lines[i].count('}')
                if brace_count == 0 and i > start and '{' in lines[start]:
                    return i + 1
            return min(start + 50, len(lines))  # Max 50 lines
    
    def _calculate_complexity_factors(
        self,
        code: str,
        language: CodeLanguage
    ) -> Dict[str, float]:
        """Calculate various complexity factors - enhanced for clone detection."""
        factors = {}
        code_lower = code.lower()
        
        # Cyclomatic complexity (improved)
        decision_keywords = ['if', 'else', 'elif', 'for', 'while', 'case', 'catch', 'switch', 
                           '&&', '||', '?', 'try', 'catch', 'throw']
        decision_count = sum(code_lower.count(kw) for kw in decision_keywords)
        factors['cyclomatic_complexity'] = min(decision_count / 12.0, 1.0)
        
        # Nesting depth
        max_nesting = self._calculate_max_nesting(code)
        factors['nesting_depth'] = min(max_nesting / 7.0, 1.0)
        
        # Function length
        line_count = len([l for l in code.split('\n') if l.strip()])
        factors['function_length'] = min(line_count / 80.0, 1.0)
        
        # Method calls (important for clone detection)
        method_call_count = len(re.findall(r'\.\w+\s*\(', code))
        factors['method_calls'] = min(method_call_count / 8.0, 1.0)
        
        # Exception handling
        has_try = 'try' in code_lower or 'catch' in code_lower
        has_risky_ops = any(op in code_lower for op in ['open(', 'read(', 'write(', 'new ', 'malloc', 'divide', '/ 0'])
        factors['exception_handling_missing'] = 1.0 if (has_risky_ops and not has_try) else 0.0
        
        # Null/empty checks
        has_null_check = any(k in code_lower for k in ['== null', '!= null', '== none', '!= none', 'is null', 'not null'])
        has_access = any(op in code for op in ['.', '->', '[', 'get', 'set'])
        factors['null_checks_missing'] = 0.8 if (has_access and not has_null_check) else 0.0
        
        # Resource management
        has_open = any(op in code_lower for op in ['open(', 'connect(', 'new file', 'new socket', 'new thread'])
        has_close = any(op in code_lower for op in ['close()', 'with ', 'try-with-resources', 'finally'])
        factors['resource_management'] = 0.9 if (has_open and not has_close) else 0.0
        
        # Duplicate code patterns (useful for clone detection)
        duplicate_patterns = len(re.findall(r'(\b\w+\b).*\1', code_lower))
        factors['duplicate_patterns'] = min(duplicate_patterns / 5.0, 1.0)
        
        return factors
    
    def _calculate_max_nesting(self, code: str) -> int:
        """Calculate maximum nesting depth."""
        max_depth = 0
        current_depth = 0
        
        for char in code:
            if char == '{':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == '}':
                current_depth = max(0, current_depth - 1)
        
        # Also count Python indentation
        for line in code.split('\n'):
            if line.strip():
                indent = (len(line) - len(line.lstrip())) // 4
                max_depth = max(max_depth, indent)
        
        return max_depth
    
    def _calculate_hotspot_score(self, factors: Dict[str, float]) -> float:
        """Calculate overall hotspot score from factors - tuned for clone detection."""
        score = 0.0
        weights = {
            'cyclomatic_complexity': 0.25,
            'nesting_depth': 0.20,
            'function_length': 0.15,
            'method_calls': 0.15,
            'exception_handling_missing': 0.10,
            'null_checks_missing': 0.08,
            'resource_management': 0.05,
            'duplicate_patterns': 0.12   # Important for clone detection
        }

        for factor, value in factors.items():
            weight = weights.get(factor, 0.1)
            score += value * weight

        return min(score, 1.0)

    def _remove_java_comments(self, code: str) -> str:
        """移除 Java 代码中的块注释和行注释，减少噪音干扰"""
        if not code:
            return ""
        # 移除多行注释 /* ... */
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        # 移除单行注释 // ...
        code = re.sub(r'//.*', '', code)
        return code

    def _get_java_tree_sitter_parser(self):
        """懒加载 tree-sitter-java Parser（失败则返回 None，走词法回退）。"""
        cache_attr = "_java_ts_parser"
        if getattr(self, cache_attr, None) is not None:
            return getattr(self, cache_attr)
        setattr(self, cache_attr, False)
        if Language is None or Parser is None:
            return None
        try:
            import tree_sitter_java as tsjava  # type: ignore
        except ImportError:
            return None
        try:
            java_lang = Language(tsjava.language())
            parser = Parser()
            parser.language = java_lang
            setattr(self, cache_attr, parser)
            return parser
        except Exception:
            return None

    @staticmethod
    def _levenshtein_type_sequence_similarity(
        a: List[str], b: List[str], max_len: int = 500
    ) -> float:
        """
        Normalized similarity from Levenshtein edit distance on node-type sequences.
        Truncates long sequences for bounded O(n*m) cost.
        """
        if not a or not b:
            return 0.0
        a = a[:max_len]
        b = b[:max_len]
        la, lb = len(a), len(b)
        prev = list(range(lb + 1))
        for i in range(1, la + 1):
            cur = [i] + [0] * lb
            for j in range(1, lb + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            prev = cur
        dist = prev[lb]
        denom = max(la, lb)
        return 1.0 - (dist / denom) if denom else 1.0

    def _preorder_java_node_types(self, root: Any, max_nodes: int) -> Tuple[List[str], int]:
        """前序遍历收集节点类型序列；返回 (types, max_depth)。"""
        if root is None:
            return [], 0
        types: List[str] = []
        stack: List[Tuple[Any, int]] = [(root, 1)]
        max_depth = 1
        while stack and len(types) < max_nodes:
            node, depth = stack.pop()
            max_depth = max(max_depth, depth)
            types.append(node.type)
            children = list(node.children)
            for c in reversed(children):
                if len(types) >= max_nodes:
                    break
                stack.append((c, depth + 1))
        return types, max_depth

    def _tree_sitter_java_type_similarity(
        self, code1: str, code2: str, max_nodes: int
    ) -> Optional[Tuple[float, int, int, int, float]]:
        """
        Java AST node-type sequence: SequenceMatcher + Jaccard + Levenshtein (normalized).
        返回 (融合分, n1, n2, max_depth, lev_sim) 或 None。
        """
        parser = self._get_java_tree_sitter_parser()
        if parser is None:
            return None
        if len(code1) < 3 or len(code2) < 3:
            return None
        try:
            t1 = parser.parse(bytes(code1, "utf8", errors="ignore"))
            t2 = parser.parse(bytes(code2, "utf8", errors="ignore"))
        except Exception:
            return None
        if t1 is None or t2 is None:
            return None
        r1, d1 = self._preorder_java_node_types(t1.root_node, max_nodes)
        r2, d2 = self._preorder_java_node_types(t2.root_node, max_nodes)
        if not r1 or not r2:
            return None
        from difflib import SequenceMatcher

        lev_cap = int(
            self.config.get("clone_similarity", {}).get(
                "levenshtein_max_len", 500
            )
        )
        lev_sim = self._levenshtein_type_sequence_similarity(
            r1, r2, max_len=lev_cap
        )

        seq_sim = SequenceMatcher(None, r1, r2).ratio()
        s1, s2 = set(r1), set(r2)
        jac = len(s1 & s2) / len(s1 | s2) if s1 | s2 else 0.0
        wseq = float(
            self.config.get("clone_similarity", {}).get("weight_sequence", 0.30)
        )
        wjac = float(
            self.config.get("clone_similarity", {}).get("weight_jaccard", 0.25)
        )
        wlev = float(
            self.config.get("clone_similarity", {}).get("weight_levenshtein", 0.45)
        )
        s = wseq * seq_sim + wjac * jac + wlev * lev_sim
        score = min(max(s, 0.0), 1.0)
        return (score, len(r1), len(r2), max(d1, d2), lev_sim)

    def _lexical_java_similarity(self, c1: str, c2: str) -> float:
        """词法 + API 重叠相似度（原正则管线，作 Tree-sitter 不可用时的回退）。"""
        if len(c1) < 5 or len(c2) < 5:
            return 0.0

        api_pattern = (
            r"(?:new\s+([A-Z][a-zA-Z0-9_]*))|(?:\.([a-z][a-zA-Z0-9_]*)\s*\()"
        )

        def extract_apis(code_text: str) -> List[str]:
            matches = re.findall(api_pattern, code_text)
            return [m[0] or m[1] for m in matches if m[0] or m[1]]

        apis1 = extract_apis(c1)
        apis2 = extract_apis(c2)

        api_sim = 0.0
        if apis1 or apis2:
            set_api1, set_api2 = set(apis1), set(apis2)
            union_api = set_api1 | set_api2
            if union_api:
                api_sim = len(set_api1 & set_api2) / len(union_api)

        c1_norm = re.sub(r"\s+", " ", c1)
        c2_norm = re.sub(r"\s+", " ", c2)

        token_pattern = (
            r"[a-zA-Z_][a-zA-Z0-9_]*|[\{\}\(\)\[\]\+\-\*/=<>!&|]+"
        )
        tokens1 = re.findall(token_pattern, c1_norm)
        tokens2 = re.findall(token_pattern, c2_norm)

        if not tokens1 or not tokens2:
            return float(api_sim)

        set1, set2 = set(tokens1), set(tokens2)
        jaccard_sim = (
            len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0.0
        )

        from difflib import SequenceMatcher

        seq_matcher = SequenceMatcher(None, tokens1[:500], tokens2[:500])
        sequence_sim = seq_matcher.ratio()

        keywords = {
            "if",
            "else",
            "for",
            "while",
            "try",
            "catch",
            "finally",
            "return",
            "switch",
            "case",
            "break",
            "throw",
            "new",
            "synchronized",
        }

        kw_freq1 = {k: tokens1.count(k) for k in keywords}
        kw_freq2 = {k: tokens2.count(k) for k in keywords}

        kw_diff = sum(abs(kw_freq1[k] - kw_freq2[k]) for k in keywords)
        kw_total = sum(kw_freq1[k] + kw_freq2[k] for k in keywords)
        struct_sim = 1.0 - (kw_diff / kw_total) if kw_total > 0 else 1.0

        if not apis1 and not apis2:
            return (
                jaccard_sim * 0.35 + sequence_sim * 0.50 + struct_sim * 0.15
            )
        return (
            api_sim * 0.35
            + jaccard_sim * 0.25
            + sequence_sim * 0.20
            + struct_sim * 0.20
        )

    def extract_api_call_sequence(self, code: str) -> List[str]:
        """
        Ordered list of method names from `.method(` calls (API linear sequence / anchors).
        Filters noisy ubiquitous methods.
        """
        if not code:
            return []
        cleaned = self._remove_java_comments(code)
        calls = re.findall(r"\.(\w+)\s*\(", cleaned)
        blacklist = {
            "toString",
            "equals",
            "hashCode",
            "get",
            "set",
            "add",
            "remove",
            "size",
            "length",
            "isEmpty",
            "println",
            "print",
            "append",
            "valueOf",
        }
        return [c for c in calls if c not in blacklist]

    @staticmethod
    def calculate_api_sequence_similarity(
        seq1: List[str], seq2: List[str]
    ) -> float:
        """Jaccard on call multiset + order-aware ratio (trimmed) for API anchors."""
        if not seq1 or not seq2:
            return 0.0
        s1, s2 = set(seq1), set(seq2)
        uni = s1 | s2
        jacc = len(s1 & s2) / len(uni) if uni else 0.0
        from difflib import SequenceMatcher

        ratio = SequenceMatcher(
            None, seq1[:200], seq2[:200]
        ).ratio()
        return min(max(0.5 * jacc + 0.5 * ratio, 0.0), 1.0)

    @staticmethod
    def _complexity_bucket(node_count: int, max_depth: int) -> str:
        if node_count > 400 or max_depth > 25:
            return "high"
        if node_count > 120 or max_depth > 15:
            return "medium"
        return "low"

    def calculate_code_similarity(self, code1: str, code2: str) -> Dict[str, Any]:
        """
        Java 克隆结构相似度：优先 Tree-sitter 节点类型序列 + 词法/API 融合；失败时回退词法管线。

        Returns:
            dict: score, lexical_score, tree_sitter_score, tree_sitter_used, complexity, ...
        """
        empty = {
            "score": 0.0,
            "lexical_score": 0.0,
            "tree_sitter_score": None,
            "tree_sitter_used": False,
            "type_sequence_levenshtein_similarity": None,
            "api_call_anchor_similarity": 0.0,
            "complexity": "low",
            "java_nodes_avg": 0,
            "max_depth": 0,
        }
        if not code1 or not code2:
            return empty

        c1 = self._remove_java_comments(code1).strip()
        c2 = self._remove_java_comments(code2).strip()

        if len(c1) < 5 or len(c2) < 5:
            return empty

        clone_cfg = self.config.get("clone_similarity", {})
        max_nodes = int(clone_cfg.get("max_type_nodes", 8000))
        tw = float(clone_cfg.get("tree_weight", 0.65))
        lw = float(clone_cfg.get("lexical_weight", 0.35))

        lexical = self._lexical_java_similarity(c1, c2)
        lexical = min(max(lexical, 0.0), 1.0)

        api_seq1 = self.extract_api_call_sequence(c1)
        api_seq2 = self.extract_api_call_sequence(c2)
        api_anchor = self.calculate_api_sequence_similarity(api_seq1, api_seq2)
        w_api = float(clone_cfg.get("api_anchor_weight", 0.12))

        ts = self._tree_sitter_java_type_similarity(c1, c2, max_nodes)
        tree_used = ts is not None
        tree_score: Optional[float] = None
        lev_sim: Optional[float] = None
        n1 = n2 = 0
        max_d = 0

        if ts is not None:
            tree_score, n1, n2, max_d, lev_sim = ts
            final = tw * tree_score + lw * lexical
        else:
            final = lexical

        final = (1.0 - w_api) * final + w_api * api_anchor
        final = min(max(final, 0.0), 1.0)

        node_avg = (n1 + n2) // 2 if tree_used else 0
        if not tree_used:
            max_d = 0
            rough_nodes = min(len(c1), len(c2)) // 8
            complexity = self._complexity_bucket(rough_nodes, 8)
        else:
            complexity = self._complexity_bucket(node_avg, max_d)

        return {
            "score": final,
            "lexical_score": lexical,
            "tree_sitter_score": tree_score,
            "tree_sitter_used": tree_used,
            "type_sequence_levenshtein_similarity": lev_sim,
            "api_call_anchor_similarity": api_anchor,
            "complexity": complexity,
            "java_nodes_avg": node_avg,
            "max_depth": max_d,
        }
    
    def _generate_hotspot_reason(self, factors: Dict[str, float]) -> str:
        """Generate human-readable reason for hotspot - enhanced for clone detection."""
        reasons = []
        
        if factors.get('cyclomatic_complexity', 0) > 0.4:
            reasons.append("high cyclomatic complexity")
        if factors.get('nesting_depth', 0) > 0.45:
            reasons.append("deep nesting")
        if factors.get('function_length', 0) > 0.5:
            reasons.append("long function")
        if factors.get('method_calls', 0) > 0.5:
            reasons.append("many method calls")
        if factors.get('exception_handling_missing', 0) > 0.6:
            reasons.append("missing exception handling")
        if factors.get('null_checks_missing', 0) > 0.6:
            reasons.append("missing null checks")
        if factors.get('resource_management', 0) > 0.7:
            reasons.append("potential resource leak")
        if factors.get('duplicate_patterns', 0) > 0.4:
            reasons.append("duplicate code patterns")
        
        if not reasons:
            if sum(factors.values()) > 0.3:
                return "moderate structural complexity"
            return "simple code structure"
        
        return "Suspicious due to: " + ", ".join(reasons)
    
    def _detect_language(self, file_path: str) -> CodeLanguage:
        """Detect language from file extension."""
        ext = Path(file_path).suffix.lower()
        
        mapping = {
            '.py': CodeLanguage.PYTHON,
            '.java': CodeLanguage.JAVA,
            '.js': CodeLanguage.JAVASCRIPT,
            '.ts': CodeLanguage.TYPESCRIPT,
            '.cpp': CodeLanguage.CPP,
            '.cc': CodeLanguage.CPP,
            '.cxx': CodeLanguage.CPP,
            '.c': CodeLanguage.C,
            '.h': CodeLanguage.C,
            '.hpp': CodeLanguage.CPP
        }
        
        return mapping.get(ext, CodeLanguage.PYTHON)


def extract_minimal_context(
    code: str,
    target_lines: tuple[int, int],
    max_context_lines: int = 5
) -> str:
    """Extract minimal context around target lines.
    
    Args:
        code: Full source code
        target_lines: (start, end) line numbers (1-based)
        max_context_lines: Max lines of context before/after
    
    Returns:
        Code snippet with minimal context
    """
    lines = code.split('\n')
    start, end = target_lines
    
    # Adjust to 0-based indexing
    start_idx = max(0, start - 1 - max_context_lines)
    end_idx = min(len(lines), end + max_context_lines)
    
    return '\n'.join(lines[start_idx:end_idx])

