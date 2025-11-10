from datasets import load_dataset

wl_set = set(r"abcdefghijklmnopqrstuvwxyz ÄÖÜäöüß ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?()[]{}:;-&$@#%£€/\|_+*¥")

def remove_ch(text: str) -> str:
    if text is None:
        return ""

    # remove non-UTF-8 characters
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")

    # remove HTML tags manually
    cleaned = ""
    inside_tag = False
    for ch in text:
        if ch == "<":
            inside_tag = True
            continue
        if ch == ">":
            inside_tag = False
            continue
        if not inside_tag:
            cleaned += ch
    text = cleaned

    # remove urls
    words = []
    for w in text.split():
        if not (w.startswith("http") or w.startswith("www.")):
            words.append(w)
    text = " ".join(words)

    # collapse extra spaces
    while "  " in text:
        text = text.replace("  ", " ")
    text = text.strip()

    #remvoe non-whitelist
    text = "".join(ch for ch in text if ch in wl_set).strip()

    return text

def filter_length(text: str, min_len: int = 5, max_len: int = 64) -> str:
    if not text:
        return ""

    words_count = len(text.split())
    if words_count < min_len or words_count > max_len:
        return ""
    return text

def filter_ratio(src: str, tar: str, max_ratio: float = 3.0) -> tuple[str, str]:
    if not src or not tar:
        return "", ""

    src_len = len(src.split())
    tar_len = len(tar.split())

    if src_len == 0 or tar_len == 0:
        return "", ""

    ratio = max(src_len, tar_len) / min(src_len, tar_len)
    return (src, tar) if ratio <= max_ratio else ("", "")

def clean_tr(src: str, tar: str) -> tuple[str, str]:
    if not src or not tar:
        return "", ""

    src, tar = remove_ch(src), remove_ch(tar)
    src, tar = filter_length(src), filter_length(tar)
    src, tar = filter_ratio(src, tar, max_ratio = 2.0)
    return src, tar

dataset = load_dataset("wmt17", "de-en")

def to_cols(batch):
    tr = batch["translation"]
    return {"de": [item["de"] for item in tr],
            "en": [item["en"] for item in tr]}

dataset = dataset.map(lambda batch: to_cols(batch), remove_columns=["translation"], batched=True)

def clean_batch(batch):
    de_out, en_out = [], []
    for de, en in zip(batch["de"], batch["en"]):
        s, t = clean_tr(de, en)
        de_out.append(s)
        en_out.append(t)
    return {"de": de_out, "en": en_out}

cleaned_ds = dataset.map(clean_batch, batched=True)

cleaned_ds = cleaned_ds.filter(lambda x: bool(x["de"]) and bool(x["en"]))

print(dataset)
print(cleaned_ds)
