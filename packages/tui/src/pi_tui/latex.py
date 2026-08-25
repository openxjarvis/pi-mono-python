"""
Terminal-friendly LaTeX rendering — mirrors packages/tui/src/latex.ts

Covers inline/display math, fractions, scripts, common symbols, aligned
equations, cases, and matrices. Returns None for unsupported syntax.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .utils import visible_width

SYMBOLS: dict[str, str] = {
    'alpha': 'α',
    'beta': 'β',
    'gamma': 'γ',
    'delta': 'δ',
    'epsilon': 'ϵ',
    'varepsilon': 'ε',
    'zeta': 'ζ',
    'eta': 'η',
    'theta': 'θ',
    'vartheta': 'ϑ',
    'iota': 'ι',
    'kappa': 'κ',
    'varkappa': 'ϰ',
    'lambda': 'λ',
    'mu': 'μ',
    'nu': 'ν',
    'xi': 'ξ',
    'pi': 'π',
    'varpi': 'ϖ',
    'rho': 'ρ',
    'varrho': 'ϱ',
    'sigma': 'σ',
    'varsigma': 'ς',
    'tau': 'τ',
    'upsilon': 'υ',
    'phi': 'ϕ',
    'varphi': 'φ',
    'chi': 'χ',
    'psi': 'ψ',
    'omega': 'ω',
    'Gamma': 'Γ',
    'Delta': 'Δ',
    'Theta': 'Θ',
    'Lambda': 'Λ',
    'Xi': 'Ξ',
    'Pi': 'Π',
    'Sigma': 'Σ',
    'Upsilon': 'Υ',
    'Phi': 'Φ',
    'Psi': 'Ψ',
    'Omega': 'Ω',
    'pm': '±',
    'mp': '∓',
    'times': '×',
    'div': '÷',
    'cdot': '·',
    'ast': '∗',
    'star': '⋆',
    'circ': '∘',
    'bullet': '•',
    'oplus': '⊕',
    'ominus': '⊖',
    'otimes': '⊗',
    'oslash': '⊘',
    'odot': '⊙',
    'bigcirc': '○',
    'dagger': '†',
    'ddagger': '‡',
    'amalg': '⨿',
    'uplus': '⊎',
    'sqcap': '⊓',
    'sqcup': '⊔',
    'triangleleft': '◁',
    'triangleright': '▷',
    'wr': '≀',
    'cap': '∩',
    'cup': '∪',
    'bigcap': '⋂',
    'bigcup': '⋃',
    'bigwedge': '⋀',
    'bigvee': '⋁',
    'bigsqcup': '⨆',
    'biguplus': '⨄',
    'bigoplus': '⨁',
    'bigotimes': '⨂',
    'bigodot': '⨀',
    'setminus': '∖',
    'in': '∈',
    'notin': '∉',
    'ni': '∋',
    'subset': '⊂',
    'supset': '⊃',
    'subseteq': '⊆',
    'supseteq': '⊇',
    'sqsubset': '⊏',
    'sqsupset': '⊐',
    'sqsubseteq': '⊑',
    'sqsupseteq': '⊒',
    'prec': '≺',
    'preceq': '≼',
    'succ': '≻',
    'succeq': '≽',
    'll': '≪',
    'gg': '≫',
    'le': '≤',
    'leq': '≤',
    'leqslant': '≤',
    'ge': '≥',
    'geq': '≥',
    'geqslant': '≥',
    'ne': '≠',
    'neq': '≠',
    'equiv': '≡',
    'approx': '≈',
    'sim': '∼',
    'simeq': '≃',
    'cong': '≅',
    'asymp': '≍',
    'doteq': '≐',
    'propto': '∝',
    'parallel': '∥',
    'perp': '⊥',
    'mid': '∣',
    'vdash': '⊢',
    'dashv': '⊣',
    'models': '⊨',
    'Vdash': '⊩',
    'Vvdash': '⊪',
    'nvdash': '⊬',
    'nvDash': '⊭',
    'forall': '∀',
    'exists': '∃',
    'nexists': '∄',
    'neg': '¬',
    'land': '∧',
    'wedge': '∧',
    'lor': '∨',
    'vee': '∨',
    'to': '→',
    'rightarrow': '→',
    'longrightarrow': '→',
    'leftarrow': '←',
    'longleftarrow': '←',
    'gets': '←',
    'leftrightarrow': '↔',
    'longleftrightarrow': '↔',
    'hookleftarrow': '↩',
    'hookrightarrow': '↪',
    'twoheadleftarrow': '↞',
    'twoheadrightarrow': '↠',
    'leftharpoonup': '↼',
    'leftharpoondown': '↽',
    'rightharpoonup': '⇀',
    'rightharpoondown': '⇁',
    'rightleftharpoons': '⇌',
    'leftrightharpoons': '⇋',
    'nearrow': '↗',
    'searrow': '↘',
    'swarrow': '↙',
    'nwarrow': '↖',
    'rightsquigarrow': '⇝',
    'leadsto': '⇝',
    'Rightarrow': '⇒',
    'Longrightarrow': '⇒',
    'Leftarrow': '⇐',
    'Longleftarrow': '⇐',
    'Leftrightarrow': '⇔',
    'Longleftrightarrow': '⇔',
    'implies': '⇒',
    'iff': '⇔',
    'mapsto': '↦',
    'longmapsto': '↦',
    'uparrow': '↑',
    'downarrow': '↓',
    'partial': '∂',
    'nabla': '∇',
    'int': '∫',
    'iint': '∬',
    'iiint': '∭',
    'oint': '∮',
    'sum': '∑',
    'prod': '∏',
    'coprod': '∐',
    'infty': '∞',
    'emptyset': '∅',
    'varnothing': '∅',
    'angle': '∠',
    'therefore': '∴',
    'because': '∵',
    'aleph': 'ℵ',
    'beth': 'ℶ',
    'gimel': 'ℷ',
    'daleth': 'ℸ',
    'top': '⊤',
    'bot': '⊥',
    'triangle': '△',
    'square': '□',
    'lozenge': '◊',
    'checkmark': '✓',
    'complement': '∁',
    'wp': '℘',
    'prime': '′',
    'ldots': '…',
    'dots': '…',
    'cdots': '⋯',
    'vdots': '⋮',
    'ddots': '⋱',
    'ell': 'ℓ',
    'hbar': 'ℏ',
    'Im': 'ℑ',
    'Re': 'ℜ',
    'langle': '⟨',
    'rangle': '⟩',
    'vert': '|',
    'lvert': '|',
    'rvert': '|',
    'Vert': '‖',
    'lVert': '‖',
    'rVert': '‖',
    'lbrace': '{',
    'rbrace': '}',
    'backslash': '\\',
    'lfloor': '⌊',
    'rfloor': '⌋',
    'lceil': '⌈',
    'rceil': '⌉',
    'colon': ':',
}

NAMED_OPERATORS = {
    "arccos", "arcsin", "arctan", "arg", "cos", "cosh", "cot", "coth", "csc",
    "deg", "det", "dim", "exp", "gcd", "hom", "inf", "ker", "lg", "lim",
    "liminf", "limsup", "ln", "log", "max", "min", "Pr", "sec", "sin", "sinh",
    "sup", "tan", "tanh",
}
LIMIT_OPERATORS = {
    "argmax", "argmin", "inf", "injlim", "lim", "liminf", "limsup",
    "max", "min", "projlim", "sup",
}
DISPLAY_LIMIT_SYMBOLS = {
    "bigcap", "bigcup", "bigodot", "bigoplus", "bigotimes", "bigsqcup",
    "biguplus", "bigvee", "bigwedge", "coprod", "int", "iint", "iiint",
    "oint", "prod", "sum",
}
RELATION_COMMANDS = {
    "Leftarrow", "Leftrightarrow", "Longleftarrow", "Longleftrightarrow",
    "Longrightarrow", "Rightarrow", "Vdash", "Vvdash", "approx", "asymp",
    "cong", "dashv", "doteq", "downarrow", "equiv", "ge", "geq", "geqslant",
    "gets", "gg", "hookleftarrow", "hookrightarrow", "iff", "implies", "in",
    "leadsto", "le", "leftarrow", "leftharpoondown", "leftharpoonup",
    "leftrightarrow", "leftrightharpoons", "leq", "leqslant", "ll",
    "longleftarrow", "longleftrightarrow", "longmapsto", "longrightarrow",
    "mapsto", "mid", "models", "ne", "nearrow", "neq", "ni", "notin",
    "nvdash", "nvDash", "nwarrow", "parallel", "perp", "prec", "preceq",
    "propto", "rightharpoondown", "rightharpoonup", "rightleftharpoons",
    "rightarrow", "rightsquigarrow", "searrow", "sim", "simeq", "sqsubset",
    "sqsubseteq", "sqsupset", "sqsupseteq", "subset", "subseteq", "succ",
    "succeq", "supset", "supseteq", "swarrow", "to", "triangleleft",
    "triangleright", "twoheadleftarrow", "twoheadrightarrow", "uparrow",
    "vdash",
}
NEGATED_SYMBOLS: dict[str, str] = {
    '<': '≮',
    '>': '≯',
    '=': '≠',
    '∈': '∉',
    '∋': '∌',
    '∣': '∤',
    '∥': '∦',
    '∼': '≁',
    '≃': '≄',
    '≅': '≇',
    '≈': '≉',
    '≡': '≢',
    '≤': '≰',
    '≥': '≱',
    '≺': '⊀',
    '≻': '⊁',
    '⊂': '⊄',
    '⊃': '⊅',
    '⊆': '⊈',
    '⊇': '⊉',
    '⊢': '⊬',
    '⊨': '⊭',
    '↔': '↮',
    '←': '↚',
    '→': '↛',
    '⇒': '⇏',
    '⇐': '⇍',
    '⇔': '⇎',
    '≼': '⋠',
    '≽': '⋡',
}
BLACKBOARD: dict[str, str] = {
    'C': 'ℂ',
    'H': 'ℍ',
    'N': 'ℕ',
    'P': 'ℙ',
    'Q': 'ℚ',
    'R': 'ℝ',
    'Z': 'ℤ',
}
SPACING_COMMANDS = {
    ",", ":", ";", " ", ">", "enspace", "enskip", "medspace", "quad",
    "qquad", "thickspace", "thinspace",
}
NEGATIVE_SPACING_COMMANDS = {"!", "negmedspace", "negthickspace", "negthinspace"}
IGNORED_COMMANDS = {
    "displaystyle", "limits", "nolimits", "scriptstyle", "scriptscriptstyle", "textstyle",
}
PLAIN_WRAPPERS = {
    "emph", "mathcal", "mathbf", "mathfrak", "mathit", "mathrm", "mathnormal",
    "mathscr", "mathsf", "mathtt", "mathup", "mbox", "overbrace", "pmb",
    "smash", "substack", "text", "textbf", "textit", "textmd", "textnormal",
    "textrm", "textsc", "textsf", "textsl", "texttt", "textup", "underbrace",
    "bm", "boldsymbol",
}
SUPERSCRIPTS: dict[str, str] = {
    '0': '⁰',
    '1': '¹',
    '2': '²',
    '3': '³',
    '4': '⁴',
    '5': '⁵',
    '6': '⁶',
    '7': '⁷',
    '8': '⁸',
    '9': '⁹',
    '+': '⁺',
    '-': '⁻',
    '=': '⁼',
    '(': '⁽',
    ')': '⁾',
    'a': 'ᵃ',
    'b': 'ᵇ',
    'c': 'ᶜ',
    'd': 'ᵈ',
    'e': 'ᵉ',
    'f': 'ᶠ',
    'g': 'ᵍ',
    'h': 'ʰ',
    'i': 'ⁱ',
    'j': 'ʲ',
    'k': 'ᵏ',
    'l': 'ˡ',
    'm': 'ᵐ',
    'n': 'ⁿ',
    'o': 'ᵒ',
    'p': 'ᵖ',
    'r': 'ʳ',
    's': 'ˢ',
    't': 'ᵗ',
    'u': 'ᵘ',
    'v': 'ᵛ',
    'w': 'ʷ',
    'x': 'ˣ',
    'y': 'ʸ',
    'z': 'ᶻ',
}
SUBSCRIPTS: dict[str, str] = {
    '0': '₀',
    '1': '₁',
    '2': '₂',
    '3': '₃',
    '4': '₄',
    '5': '₅',
    '6': '₆',
    '7': '₇',
    '8': '₈',
    '9': '₉',
    '+': '₊',
    '-': '₋',
    '=': '₌',
    '(': '₍',
    ')': '₎',
    'a': 'ₐ',
    'e': 'ₑ',
    'h': 'ₕ',
    'i': 'ᵢ',
    'j': 'ⱼ',
    'k': 'ₖ',
    'l': 'ₗ',
    'm': 'ₘ',
    'n': 'ₙ',
    'o': 'ₒ',
    'p': 'ₚ',
    'r': 'ᵣ',
    's': 'ₛ',
    't': 'ₜ',
    'u': 'ᵤ',
    'v': 'ᵥ',
    'x': 'ₓ',
}
NAMED_OPERATOR_START = "\uf0004"
NAMED_OPERATOR_END = "\uf0005"
LAYOUT_MARKER_START = "\uf0000"
LAYOUT_MARKER_END = "\uf0001"
PROTECTED_SPACE = "\uf0002"
NEGATIVE_SPACE = "\x00"
_LAYOUT_MARKER_RE = re.compile(r"\uf0000(\d+)\uf0001")


def _format_script(value: str, kind: Literal["sub", "sup"]) -> str:
    value = value.strip()
    replacements = SUBSCRIPTS if kind == "sub" else SUPERSCRIPTS
    cleaned = re.sub(r"\s*([=+-])\s*", r"\1", value)
    if all(ch in replacements for ch in cleaned):
        return "".join(replacements[ch] for ch in cleaned)
    prefix = "_" if kind == "sub" else "^"
    if len(value) == 1 or (kind == "sub" and re.fullmatch(r"[A-Za-z]+", value)):
        return f"{prefix}{value}"
    return f"{prefix}({value})"


def _format_fraction(numerator: str, denominator: str) -> str:
    numerator = numerator.strip()
    denominator = denominator.strip()
    simple_n = bool(re.fullmatch(r"[\w.]+", numerator, re.UNICODE))
    simple_d = bool(re.fullmatch(r"[\d.]+", denominator) or len(denominator) == 1)
    left = numerator if simple_n else f"({numerator})"
    right = denominator if simple_d else f"({denominator})"
    return f"{left}/{right}"


def _format_stacked_fraction(numerator: str, denominator: str) -> str:
    numerator = numerator.strip()
    denominator = denominator.strip()
    width = max(visible_width(numerator), visible_width(denominator), 1)
    top = numerator.center(width)
    bot = denominator.center(width)
    return f"{top}\n{'─' * width}\n{bot}"


def _normalize_output(value: str) -> str:
    value = value.replace(NAMED_OPERATOR_START, "").replace(NAMED_OPERATOR_END, "")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in value.split("\n")]
    return "\n".join(line for i, line in enumerate(lines) if line or (0 < i < len(lines) - 1)).strip()


@dataclass
class _FractionNode:
    numerator: str
    denominator: str
    type: str = "fraction"


@dataclass
class _OperatorNode:
    operator: str
    lower: str | None = None
    upper: str | None = None
    type: str = "operator"


@dataclass
class _MatrixNode:
    lines: list[str]
    baseline: int = 0
    type: str = "matrix"


class LatexParser:
    def __init__(self, source: str, layout_nodes: list, display: bool) -> None:
        self.source = source
        self.layout_nodes = layout_nodes
        self.display = display
        self.position = 0
        self.supported = True
        self.stack_fractions = True

    def render(self) -> str | None:
        rendered = self._parse_sequence()
        if not self.supported or self.position != len(self.source):
            return None
        return _normalize_output(rendered)

    def _parse_sequence(self, end_character: str | None = None) -> str:
        result = ""
        while self.position < len(self.source):
            ch = self.source[self.position]
            if end_character and ch == end_character:
                self.position += 1
                return result
            if ch == "}":
                self.supported = False
                return result
            if ch == "{":
                self.position += 1
                result += self._parse_sequence("}")
                continue
            if ch == "\\":
                command = self._parse_command()
                if command == NEGATIVE_SPACE:
                    result = result.rstrip()
                else:
                    result += command
                continue
            if ch in "^_":
                self.position += 1
                result = result.rstrip()
                script = _format_script(self._parse_required_argument(False), "sub" if ch == "_" else "sup")
                result += script
                continue
            if ch.isspace():
                result += self._parse_whitespace()
                continue
            if ch in "=<>":
                result = f"{result.rstrip()} {ch} "
                self.position += 1
                continue
            if ch == "&":
                self.position += 1
                continue
            if ch == "~":
                self.position += 1
                result += " "
                continue
            result += ch
            self.position += 1
        if end_character:
            self.supported = False
        return result

    def _parse_whitespace(self) -> str:
        while self.position < len(self.source) and self.source[self.position].isspace():
            self.position += 1
        return " "

    def _parse_command(self) -> str:
        self.position += 1
        if self.position >= len(self.source):
            self.supported = False
            return ""
        first = self.source[self.position]
        # Control space split across a line ending is still a space.
        if first in "\n\r":
            self.position += 1
            if first == "\r" and self.position < len(self.source) and self.source[self.position] == "\n":
                self.position += 1
            return " "
        if first.isalpha():
            start = self.position
            while self.position < len(self.source) and self.source[self.position].isalpha():
                self.position += 1
            command = self.source[start:self.position]
        else:
            command = first
            self.position += 1

        if command == "\\":
            return "\n"
        if command in SPACING_COMMANDS:
            return " "
        if command in NEGATIVE_SPACING_COMMANDS:
            return NEGATIVE_SPACE
        if command in IGNORED_COMMANDS:
            return ""
        if command in "{}%$#_&":
            return command
        if command == "|":
            return "‖"
        if command in LIMIT_OPERATORS:
            return self._parse_operator(command, True)
        symbol = SYMBOLS.get(command)
        if symbol is not None:
            if command in DISPLAY_LIMIT_SYMBOLS:
                return self._parse_operator(symbol, True)
            if command in ("cdot", "times") or command in RELATION_COMMANDS:
                return f" {symbol} "
            return symbol
        if command in NAMED_OPERATORS:
            return f"{NAMED_OPERATOR_START}{command}{NAMED_OPERATOR_END}"
        if command in ("frac", "dfrac", "tfrac"):
            should_stack = self.display and self.stack_fractions and command != "tfrac"
            numerator = self._parse_required_argument(not should_stack)
            denominator = self._parse_required_argument(not should_stack)
            if should_stack:
                index = len(self.layout_nodes)
                self.layout_nodes.append(_FractionNode(_normalize_output(numerator), _normalize_output(denominator)))
                return f"{LAYOUT_MARKER_START}{index}{LAYOUT_MARKER_END}"
            return _format_fraction(numerator, denominator)
        if command == "sqrt":
            value = self._parse_required_argument()
            return f"√{value.strip()}" if re.fullmatch(r"[\w.]+", value.strip(), re.UNICODE) else f"√({value.strip()})"
        if command == "not":
            value = self._parse_required_argument()
            first = next((ch for ch in value if not ch.isspace()), "")
            negated = NEGATED_SYMBOLS.get(first)
            if negated:
                return value.replace(first, negated, 1)
            return f"¬{value.strip()}"
        if command == "mathbb":
            value = self._parse_required_argument()
            return "".join(BLACKBOARD.get(ch, ch) for ch in value)
        if command in PLAIN_WRAPPERS:
            value = self._parse_required_argument()
            return value if command.startswith("text") or command == "mbox" else value.strip()
        if command == "begin":
            return self._parse_environment()
        if command == "end":
            self.supported = False
            return ""
        self.supported = False
        return f"\\{command}"

    def _parse_operator(self, operator: str, display_limits: bool) -> str:
        lower = None
        upper = None
        while True:
            pos = self.position
            while pos < len(self.source) and self.source[pos] in " \t":
                pos += 1
            if pos >= len(self.source) or self.source[pos] not in "_^":
                break
            kind = self.source[pos]
            self.position = pos + 1
            value = _normalize_output(self._parse_required_argument(False)).replace(" ", "")
            if kind == "_":
                lower = value
            else:
                upper = value
        if self.display and display_limits and (lower or upper):
            index = len(self.layout_nodes)
            self.layout_nodes.append(_OperatorNode(operator, lower, upper))
            return f"{LAYOUT_MARKER_START}{index}{LAYOUT_MARKER_END}"
        rendered = operator
        if lower:
            rendered += f"[{lower}]"
        if upper:
            rendered += _format_script(upper, "sup")
        return f" {rendered} "

    def _parse_required_argument(self, stack_fractions: bool = True) -> str:
        previous = self.stack_fractions
        self.stack_fractions = previous and stack_fractions
        value = self._parse_required_argument_value()
        self.stack_fractions = previous
        return value

    def _parse_required_argument_value(self) -> str:
        # Skip whitespace including newlines so `{` on the next line is still an argument.
        while self.position < len(self.source) and self.source[self.position].isspace():
            self.position += 1
        if self.position >= len(self.source):
            self.supported = False
            return ""
        if self.source[self.position] == "{":
            self.position += 1
            return self._parse_sequence("}")
        if self.source[self.position] == "\\":
            return self._parse_command()
        value = self.source[self.position]
        self.position += 1
        return value

    def _read_raw_group(self) -> str | None:
        while self.position < len(self.source) and self.source[self.position] in " \t":
            self.position += 1
        if self.position >= len(self.source) or self.source[self.position] != "{":
            self.supported = False
            return None
        self.position += 1
        start = self.position
        depth = 1
        while self.position < len(self.source):
            ch = self.source[self.position]
            if ch == "\\":
                self.position += 2
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    value = self.source[start:self.position]
                    self.position += 1
                    return value
            self.position += 1
        self.supported = False
        return None

    def _parse_environment(self) -> str:
        environment = self._read_raw_group()
        if not environment:
            return ""
        end_marker = f"\\end{{{environment}}}"
        end = self.source.find(end_marker, self.position)
        if end < 0:
            self.supported = False
            return ""
        body = self.source[self.position:end]
        self.position = end + len(end_marker)
        rows = re.split(r"\\\\(?:\[[^\]\n]*\])?", body)
        if environment in ("equation", "equation*", "displaymath"):
            return self._render_nested(body).strip()
        if environment in ("aligned", "align", "align*", "gather", "gathered", "split"):
            return "\n".join(
                self._render_nested(row.replace("&", "")).strip()
                for row in rows
                if self._render_nested(row.replace("&", "")).strip()
            )
        if environment in ("cases", "cases*"):
            rendered_rows = []
            for index, row in enumerate(rows):
                cells = [self._render_nested(cell, False).strip() for cell in row.split("&")]
                if not any(cells):
                    continue
                delimiter = "⎧" if index == 0 else ("⎩" if index == len(rows) - 1 else "⎨")
                value = re.sub(r",\s*$", "", cells[0] if cells else "")
                condition = cells[1] if len(cells) > 1 else ""
                prefix = " " if re.match(r"^(?:if|when|for|otherwise)\b", condition, re.I) else " if "
                rendered_rows.append(f"{delimiter} {value}{prefix + condition if condition else ''}")
            return "\n".join(rendered_rows)
        if environment in ("array", "matrix", "pmatrix", "bmatrix", "vmatrix"):
            matrix = [
                [self._render_nested(cell, False).strip() for cell in row.split("&")]
                for row in rows
            ]
            matrix = [row for row in matrix if any(row)]
            col_count = max((len(row) for row in matrix), default=0)
            widths = [
                max((visible_width(row[col] if col < len(row) else "") for row in matrix), default=0)
                for col in range(col_count)
            ]
            lines = []
            for row in matrix:
                cells = []
                for col in range(col_count):
                    cell = row[col] if col < len(row) else ""
                    cells.append(cell + " " * max(0, widths[col] - visible_width(cell)))
                lines.append(" │ ".join(cells))
            if environment == "pmatrix":
                return "\n".join(f"( {line} )" for line in lines)
            if environment == "bmatrix":
                return "\n".join(f"[ {line} ]" for line in lines)
            if environment == "vmatrix":
                return "\n".join(f"│ {line} │" for line in lines)
            return "\n".join(lines)
        self.supported = False
        return body

    def _render_nested(self, source: str, stack_fractions: bool = True) -> str:
        rendered = LatexParser(source, self.layout_nodes, self.display and stack_fractions).render()
        if rendered is None:
            self.supported = False
            return source
        return rendered


@dataclass
class RenderLatexOptions:
    display: bool = False


def render_latex(
    source: str,
    options: RenderLatexOptions | None = None,
    display: bool = False,
) -> str | None:
    """Render a basic LaTeX math expression as terminal-friendly Unicode text."""
    if options is not None:
        display = options.display
    layout_nodes: list = []
    rendered = LatexParser(source, layout_nodes, display).render()
    if rendered is None:
        return None
    if not layout_nodes:
        return rendered.replace(PROTECTED_SPACE, " ")
    lines = [_expand_layout_line(line, layout_nodes) for line in rendered.split("\n")]
    return "\n".join(lines).strip().replace(PROTECTED_SPACE, " ")


def _expand_layout_line(line: str, nodes: list) -> str:
    parts: list[str] = []
    last = 0
    for match in _LAYOUT_MARKER_RE.finditer(line):
        parts.append(line[last:match.start()])
        node = nodes[int(match.group(1))]
        if isinstance(node, _FractionNode):
            parts.append(_format_stacked_fraction(node.numerator, node.denominator))
        elif isinstance(node, _OperatorNode):
            rendered = node.operator
            if node.lower:
                rendered += f"[{node.lower}]"
            if node.upper:
                rendered += _format_script(node.upper, "sup")
            parts.append(rendered)
        elif isinstance(node, _MatrixNode):
            parts.append(node.lines[0] if node.lines else "")
        last = match.end()
    parts.append(line[last:])
    return "".join(parts)
