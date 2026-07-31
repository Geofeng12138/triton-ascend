"""Sphinx extension: translate text inside HTML raw nodes while preserving markup.

Myst-parser treats inline HTML blocks (e.g. ``<table>``, ``<h2>``) as a single
docutils ``raw`` node.  Inside that raw node the visible text (``<th>``,
``<h1>-<h6>`` …) is never exposed to Sphinx gettext, so these strings cannot be
translated and do not appear in the generated .po files.

This extension runs a transform (before Sphinx's own i18n transform) that splits
each HTML ``raw`` node into:

* plain ``nodes.raw`` segments that contain no translatable text, and
* ``HtmlRawTranslatable`` elements (an ``addnodes.translatable`` subclass) that
  carry the wrapped HTML tag (e.g. ``<th style="...">`` / ``<h2 id="...">``)
  together with the text run as a normal translatable child.

As a result:

* ``sphinx-build -b gettext`` emits one .po entry per text run (e.g. "CANN版本"),
* the HTML builder re-emits the same tag (with all attributes) around the
  translated text, so the visual/markup structure is fully preserved.

Enable in conf.py by adding this module to the ``extensions`` list (the module
directory must be on ``sys.path``), or by calling ``setup(app)`` from the
project's own ``setup()`` hook.
"""
import re
from docutils import nodes
from sphinx import addnodes
from sphinx.transforms import SphinxTransform

__all__ = [
    "HtmlRawTranslatable",
    "setup",
]

# Tags whose inner text should be extracted for translation.
# (th), (h1)..(h6) cover table headers and HTML headings.
_HTML_TEXT_RE = re.compile(
    r"(<(?:th|h[1-6])(?:\s[^>]*)?>)(.*?)(</(?:th|h[1-6])>)",
    re.DOTALL | re.IGNORECASE,
)


class HtmlRawTranslatable(nodes.Element, addnodes.translatable):
    """An Element wrapping a text run inside HTML markup (th/h1-h6).

    The HTML opening/closing tag (with all attributes) is stored in the
    ``rawhtml`` attribute as ``open_tag + "\\0" + close_tag``; the visible text
    is kept as a normal child so Sphinx i18n can translate it.
    """

    def preserve_original_messages(self) -> None:
        self._original_messages = [self.astext()]

    def apply_translated_message(self, original_message: str, translated_message: str) -> None:
        del self.children[:]
        self.children.append(nodes.Text(translated_message))

    def extract_original_messages(self):
        return list(getattr(self, "_original_messages", []))


def _split_raw(html: str) -> list:
    """Split ``html`` into ``("rest", raw_html)`` / ``("wrap", "<tag>text</tag>")`` pieces."""
    pieces = []
    rest = html
    while True:
        m = _HTML_TEXT_RE.search(rest)
        if not m:
            if rest:
                pieces.append(("rest", rest))
            break
        pre = rest[:m.start()]
        if pre:
            pieces.append(("rest", pre))
        pieces.append(("wrap", m.group(0)))
        rest = rest[m.end():]
    return pieces


def visit_html_raw_translatable(self, node):
    # Emit the stored opening tag (e.g. '<th style="...">'), then let the
    # writer output the translated children, and finalize with the close tag.
    open_tag, _, _ = node["rawhtml"].partition("\0")
    self.body.append(open_tag)


def depart_html_raw_translatable(self, node):
    _, _, close_tag = node["rawhtml"].partition("\0")
    self.body.append(close_tag)


class HtmlTextTranslation(SphinxTransform):
    """Split HTML raw nodes into translatable text + preserved markup parts."""

    default_priority = 5  # must run before PreserveTranslatableMessages(10)

    def apply(self, **kwargs) -> None:
        for raw in list(self.document.findall(nodes.raw)):
            if raw.get("format", "") not in ("html", "html5", ""):
                continue
            html = raw.astext()
            pieces = _split_raw(html)
            if not any(kind == "wrap" for _, kind in [] if False) and not any(kind == "wrap" for kind, _ in pieces):
                continue

            source = getattr(raw, "source", None)
            line = getattr(raw, "line", None)

            new_container = nodes.container()
            new_container.source = source
            new_container.line = line

            for kind, data in pieces:
                if kind == "wrap":
                    m = _HTML_TEXT_RE.match(data)
                    open_tag, text, close_tag = (
                        m.group(1),
                        m.group(2),
                        m.group(3),
                    )
                    wrapper = HtmlRawTranslatable()
                    wrapper.append(nodes.Text(text))
                    wrapper["rawhtml"] = open_tag + "\0" + close_tag
                    wrapper.source = source
                    wrapper.line = line
                    new_container.append(wrapper)
                else:
                    r = nodes.raw(data, data, format="html")
                    r.source = source
                    r.line = line
                    new_container.append(r)

            raw.replace_self(new_container)


def setup(app):
    app.add_transform(HtmlTextTranslation)
    app.add_node(
        HtmlRawTranslatable,
        html=(visit_html_raw_translatable, depart_html_raw_translatable),
    )
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
