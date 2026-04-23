import sys
import zipfile
from xml.etree import ElementTree as ET


WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def extract_paragraph_text(paragraph_element: ET.Element) -> str:
    text_parts = []
    # Iterate through runs and text elements, including line/tab breaks
    for element in paragraph_element.iter():
        tag = element.tag
        if not isinstance(tag, str):
            continue
        if tag.endswith("}t") and element.text:
            text_parts.append(element.text)
        elif tag.endswith("}tab"):
            text_parts.append("\t")
        elif tag.endswith("}br"):
            text_parts.append("\n")
    return "".join(text_parts)


def extract_docx_text(docx_path: str) -> str:
    with zipfile.ZipFile(docx_path) as zf:
        xml_bytes = zf.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    paragraphs = root.findall(f".//{{{WORD_NS}}}p")
    lines = []
    for p in paragraphs:
        line = extract_paragraph_text(p).strip("\u200b\uFEFF\n\r ")
        lines.append(line)
    # Collapse excessive blank lines but preserve paragraph breaks
    out_lines = []
    last_blank = False
    for line in lines:
        is_blank = (line.strip() == "")
        if is_blank and last_blank:
            # skip multiple consecutive blanks
            continue
        out_lines.append(line)
        last_blank = is_blank
    return "\n".join(out_lines)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python extract_docx_text.py <path-to-docx>", file=sys.stderr)
        sys.exit(2)
    path = sys.argv[1]
    try:
        text = extract_docx_text(path)
        # Ensure UTF-8 stdout
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass
        print(text)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()



import zipfile
from xml.etree import ElementTree as ET


WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def extract_paragraph_text(paragraph_element: ET.Element) -> str:
    text_parts = []
    # Iterate through runs and text elements, including line/tab breaks
    for element in paragraph_element.iter():
        tag = element.tag
        if not isinstance(tag, str):
            continue
        if tag.endswith("}t") and element.text:
            text_parts.append(element.text)
        elif tag.endswith("}tab"):
            text_parts.append("\t")
        elif tag.endswith("}br"):
            text_parts.append("\n")
    return "".join(text_parts)


def extract_docx_text(docx_path: str) -> str:
    with zipfile.ZipFile(docx_path) as zf:
        xml_bytes = zf.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    paragraphs = root.findall(f".//{{{WORD_NS}}}p")
    lines = []
    for p in paragraphs:
        line = extract_paragraph_text(p).strip("\u200b\uFEFF\n\r ")
        lines.append(line)
    # Collapse excessive blank lines but preserve paragraph breaks
    out_lines = []
    last_blank = False
    for line in lines:
        is_blank = (line.strip() == "")
        if is_blank and last_blank:
            # skip multiple consecutive blanks
            continue
        out_lines.append(line)
        last_blank = is_blank
    return "\n".join(out_lines)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python extract_docx_text.py <path-to-docx>", file=sys.stderr)
        sys.exit(2)
    path = sys.argv[1]
    try:
        text = extract_docx_text(path)
        # Ensure UTF-8 stdout
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass
        print(text)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()





