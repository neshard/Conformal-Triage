"""Split multi-answer (checkbox) responses against a known option list."""
import re


def normalize(text):
    text = re.sub(r"\s+", " ", str(text).casefold()).strip()
    return re.sub(r"\s*/\s*", "/", text)


def split_answer(raw, options):
    """Return (selected options, leftover fragments) for one raw answer.

    Options are consumed from the left, longest label first, so labels that
    contain commas are kept whole. Text that matches no option up to the next
    comma becomes a leftover fragment.
    """
    by_norm = sorted(((normalize(o), o) for o in options), key=lambda p: -len(p[0]))
    rest, chosen, leftover = normalize(raw), [], []
    while rest:
        for norm, label in by_norm:
            if rest == norm or rest.startswith(norm + ","):
                chosen.append(label)
                rest = rest[len(norm):]
                break
        else:
            frag, _, rest = rest.partition(",")
            if frag.strip():
                leftover.append(frag.strip())
        rest = rest.lstrip(", ")
    return chosen, leftover
