# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import os
import re as _re
import sys as _sys
import difflib as _difflib
import subprocess as _subprocess
import importlib.util as _ilu

project = 'Triton Ascend'
copyright = '2026, Huawei'
author = 'Huawei'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.mathjax',
    'myst_parser',
    'sphinx_copybutton',
    'sphinxcontrib.mermaid',
]

# Map ```mermaid code fences to the mermaid directive instead of rendering as code blocks.
myst_fence_as_directive = ['mermaid']

# -- MyST configuration -------------------------------------------------------
# Enable dollar-math extension so that $$...$$ and $...$ syntax is parsed.
myst_enable_extensions = ['dollarmath']
myst_dollar_math = True

# Mock imports for modules that aren't available in the build environment.
autodoc_mock_imports = ['triton']

# Suppress duplicate autosectionlabel warnings caused by subdirectory
# index.md headings sharing names with category headings in main index.md.
suppress_warnings = ["autosectionlabel"]

autosummary_generate = True

# ---------------------------------------------------------------------------
# Build language detection
# ---------------------------------------------------------------------------
_readthedocs_lang = os.environ.get('READTHEDOCS_LANGUAGE')

if _readthedocs_lang:
    _build_lang = _readthedocs_lang.strip().lower().replace('_', '-')
else:
    _build_lang = (os.environ.get('LANGUAGE') or 'en').strip().lower().replace('_', '-')

_is_zh = _build_lang in ('zh-cn', 'zh') or _build_lang.startswith('zh-')
language = 'zh_CN' if _is_zh else 'en'

# ---------------------------------------------------------------------------
# Gettext / i18n
# ---------------------------------------------------------------------------
gettext_compact = False
# Extract code blocks (literal blocks) as translatable units so that Chinese
# comments inside code blocks are also translated (not skipped).
gettext_additional_targets = ['literal-block', 'raw', 'image']
if not _is_zh:
    # English build: read gettext .po translations from locale/en/LC_MESSAGES/.
    locale_dirs = ['../locale/']

# ---------------------------------------------------------------------------
# Community documents whose English build renders the canonical English document
# instead of the machine-translated output.
# Mapping: docname (relative to docs/zh/, without extension) -> path relative to
# the repository root. The first four entries live in the repository root, the
# last two in the English docs tree.
# All of them are intentionally NOT translated by translate_md.py
# (see EXCLUDED_FILE_STEMS there).
# ---------------------------------------------------------------------------
_COMMUNITY_ROOT_DOCS = {
    "community/CODE_OF_CONDUCT_zh": "CODE_OF_CONDUCT.md",
    "community/CONTRIBUTING_zh": "CONTRIBUTING.md",
    "community/GOVERNANCE_zh": "GOVERNANCE.md",
    "community/SECURITYNOTE_zh": "SECURITYNOTE.md",
    "community/community_technical_meeting": "docs/en/community/community_technical_meeting.md",
    "community/roadmap_guide": "docs/en/community/roadmap_guide.md",
}

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(__file__)
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))

# ---------------------------------------------------------------------------
# Shared helpers (used by both Chinese and English builds)
# ---------------------------------------------------------------------------


def _load_module(module_name, file_path):
    """Load a Python module by file path."""
    spec = _ilu.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_name!r} from {file_path!r}")
    module = _ilu.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Triton import + mock setup (shared by both builds)
# ---------------------------------------------------------------------------
_sys.path.insert(0, os.path.join(_REPO, "python"))
_force_mock = (os.environ.get("TRITON_DOCS_FORCE_MOCK", "").lower() in ("1", "true", "yes")
               or os.environ.get("READTHEDOCS") == "True")
if not _force_mock:
    try:
        import triton
    except Exception as _exc:
        print(f"import triton failed ({_exc!r}); building docs with mock stubs")
        _force_mock = True

if _force_mock:
    _load_module(
        "docs.zh._mock._triton_mock",
        os.path.join(_HERE, "_mock", "_triton_mock.py"),
    ).install()

import triton
import triton.language.extra as _tl_extra

# Operator doc stubs — ``tensor`` operator syntax (``x / y``, ``x & y``,
# ``x >= y``, ...) has no ``tl.``-prefixed functions; attach lightweight
# stubs so autosummary can render them like add/sub/mul.
_ops_stubs_path = os.path.join(_HERE, "python-api", "_ops_stubs.py")
_ops_stubs_spec = _ilu.spec_from_file_location("_ops_stubs", _ops_stubs_path)
if _ops_stubs_spec is not None and _ops_stubs_spec.loader is not None:
    _ops_stubs = _ilu.module_from_spec(_ops_stubs_spec)
    _ops_stubs_spec.loader.exec_module(_ops_stubs)
    _ops_stubs.install(triton.language)

_cann_lang_path = os.path.join(_REPO, "third_party", "ascend", "language")
if _cann_lang_path not in _tl_extra.__path__:
    _tl_extra.__path__.append(_cann_lang_path)

# ---------------------------------------------------------------------------
# Sphinx JIT-function patching (shared by both builds)
# ---------------------------------------------------------------------------
import sphinx.ext.autosummary
import sphinx.util.inspect


def _unwrap_jit(fn):
    """Wrap a Sphinx inspection helper so it sees JITFunction.fn instead."""

    def wrapper(obj, **kwargs):
        if isinstance(obj, triton.runtime.JITFunction):
            obj = obj.fn
        return fn(obj, **kwargs)

    return wrapper


# Sphinx <9 uses "get_documenter"(app, obj, parent);
# Sphinx 9+ uses "_get_documenter"(obj, parent).
_doc_fn_name = "_get_documenter" if hasattr(sphinx.ext.autosummary, "_get_documenter") else "get_documenter"
if hasattr(sphinx.ext.autosummary, _doc_fn_name):
    _orig_get_documenter = getattr(sphinx.ext.autosummary, _doc_fn_name)
    import inspect as _inspect
    _takes_app = "app" in _inspect.signature(_orig_get_documenter).parameters

    def _patched_get_documenter(*args, **kwargs):
        # 'obj' is at index 1 for old Sphinx (app, obj, parent),
        # at index 0 for Sphinx 9.x (obj, parent).
        _args = list(args)
        _obj_idx = 1 if _takes_app else 0
        if isinstance(_args[_obj_idx], triton.runtime.JITFunction):
            _args[_obj_idx] = _args[_obj_idx].fn
        return _orig_get_documenter(*_args, **kwargs)

    setattr(sphinx.ext.autosummary, _doc_fn_name, _patched_get_documenter)

sphinx.util.inspect.unwrap_all = _unwrap_jit(sphinx.util.inspect.unwrap_all)
sphinx.util.inspect.signature = _unwrap_jit(sphinx.util.inspect.signature)
sphinx.util.inspect.object_description = _unwrap_jit(sphinx.util.inspect.object_description)

# ---------------------------------------------------------------------------
# Sphinx config (templates, source suffix, HTML theme)
# ---------------------------------------------------------------------------
templates_path = ['_templates']

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- HTML theme: sphinx_book_theme (same setup as the pre-mkdocs vllm-ascend
# docs, e.g. https://docs.vllm.ai/projects/ascend/en/v0.23.0/) ---------------
html_theme = 'sphinx_book_theme'
html_title = 'Triton Ascend'
html_static_path = ['_static']
html_last_updated_fmt = "%b %d, %Y"

html_theme_options = {
    # Repository buttons (top right of every page) and "suggest edit" links.
    'path_to_docs': 'docs/zh',
    'repository_url': 'https://github.com/triton-lang/triton-ascend',
    'repository_branch': 'main',
    'use_repository_button': True,
    'use_edit_page_button': True,
    # Sidebar shows only the project name (no logo image).
    'logo': {
        'text': 'Triton Ascend',
    },
    # No persistent items in the top navbar: pydata-sphinx-theme otherwise
    # renders a search field there that duplicates the sidebar search, and
    # its hidden sidebar-toggle would steal the JS click binding from the
    # visible toggle in the article header. Note this only empties the
    # navbar's content -- the empty top bar (sticky background strip) is
    # hidden separately via _static/custom.css (#pst-header).
    'navbar_persistent': [],
}

# ---------------------------------------------------------------------------
# English-build source-read hooks (community doc replacement + translation
# fallback)
# ---------------------------------------------------------------------------


def _on_source_read(app, docname, source):
    """Replace community docs with their canonical English source (English build).

    During the English build the community documents listed in
    ``_COMMUNITY_ROOT_DOCS`` are excluded from gettext translation, so no .po
    files exist for them. This hook reads the canonical English file (repository
    root, or the English docs tree) and swaps it in before Sphinx parses the
    source.
    """
    if _is_zh:
        return
    en_file = _COMMUNITY_ROOT_DOCS.get(docname)
    if en_file is None:
        return
    en_path = os.path.join(_REPO, en_file)
    try:
        with open(en_path, encoding='utf-8') as f:
            source[0] = f.read()
    except OSError as e:
        print(f"Warning: could not read English community doc {en_path}: {e}")


def _get_po_source_blob(po_path):
    """Read the X-Source-Commit (source blob id) recorded in a .po file."""
    try:
        with open(po_path, encoding='utf-8') as f:
            raw = f.read()
    except OSError:
        return None
    m = _re.search(r'X-Source-Commit:\s*([0-9a-fA-F]{7,40})', raw)
    return m.group(1) if m else None


def _git_cat_file(blob_id, repo_path):
    """Get file content from a git blob object id (works on shallow clones)."""
    try:
        result = _subprocess.run(
            ['git', 'cat-file', 'blob', blob_id],
            capture_output=True,
            text=True,
            cwd=repo_path,
        )
        if result.returncode == 0:
            return result.stdout
    except (OSError, _subprocess.CalledProcessError):
        pass
    return None


def _build_fallback_source(old_source, new_source):
    """Build fallback: keep only content present in both, using old version.

    Diff opcodes:
    - equal   (unchanged) -> use old (translated)
    - replace (modified)  -> use old (translated, shows stale English)
    - delete  (in old)    -> skip (deleted content removed from English)
    - insert  (new)       -> skip (new content hidden until translated)
    """
    old_lines = old_source.splitlines(keepends=True)
    new_lines = new_source.splitlines(keepends=True)
    matcher = _difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    result = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ('equal', 'replace'):
            result.extend(old_lines[i1:i2])
    return ''.join(result)


def _on_source_read_fallback(app, docname, source):
    """Translation fallback: if the source changed since the last translation,
    show only the translated subset (old version) so the English site never
    leaks untranslated Chinese during the translation PR window.

    Skipped for:
    - Chinese build (_is_zh)
    - Community docs (handled by _on_source_read)
    - No .po file (new document, no translation yet)
    - No X-Source-Commit in .po (predates the field)
    - Source unchanged (blob id matches, all translations exist)
    - Blob not available (shallow clone without the object)
    """
    if _is_zh:
        return
    if docname in _COMMUNITY_ROOT_DOCS:
        return
    po_path = os.path.join(_REPO, 'docs', 'locale', 'en', 'LC_MESSAGES', docname + '.po')
    if not os.path.exists(po_path):
        return
    blob_id = _get_po_source_blob(po_path)
    if not blob_id:
        return
    old_source = _git_cat_file(blob_id, _REPO)
    if old_source is None:
        return
    if source[0] == old_source:
        return
    source[0] = _build_fallback_source(old_source, source[0])


# ---------------------------------------------------------------------------
# Sphinx setup
# ---------------------------------------------------------------------------


def setup(app):
    """Sphinx setup (runs for both Chinese and English builds)."""
    from sphinx.highlighting import lexers
    from pygments.lexers import get_lexer_by_name

    lexers['mlir'] = get_lexer_by_name('text')
    lexers['plaintext'] = get_lexer_by_name('text')
    app.add_css_file('custom.css')
    if not _is_zh:
        app.connect('source-read', _on_source_read)
        app.connect('source-read', _on_source_read_fallback)
    return {'version': '0.1', 'parallel_read_safe': True}


readthedocs_version = os.environ.get('READTHEDOCS_VERSION', 'latest')
parts = readthedocs_version.split('.')
version = '.'.join(parts[:2]) if len(parts) >= 2 else ''
release = readthedocs_version
