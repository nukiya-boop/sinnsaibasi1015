"""心斎橋 禅園 Instagram リール用 PR 動画（1080x1920 / 30秒 / 30fps）を生成する。

使い方:
    FONT_DIR=/path/to/fonts python3 video/make_reel.py

FONT_DIR には以下の TTF を置く（いずれも SIL Open Font License）:
    ShipporiMincho_500Medium.ttf, ShipporiMincho_600SemiBold.ttf,
    CormorantGaramond_500Medium.ttf, CormorantGaramond_500Medium_Italic.ttf
"""
import math
import os
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, "images")
OUT = os.path.join(ROOT, "video", "zenen_shinsaibashi_reel_30s.mp4")
FONT_DIR = os.environ.get("FONT_DIR", os.path.join(ROOT, "video", "fonts"))

W, H, FPS, DURATION = 1080, 1920, 30, 30.0
N_FRAMES = int(FPS * DURATION)
SRC_H = 2300  # 元画像はこの高さまで縮小してから使う（ズーム用の余白込み）

GOLD = (201, 169, 106)
WHITE = (246, 240, 228)


def font(name, size):
    return ImageFont.truetype(os.path.join(FONT_DIR, name), size)


MINCHO_M = "ShipporiMincho_500Medium.ttf"
MINCHO_SB = "ShipporiMincho_600SemiBold.ttf"
SERIF_EN = "CormorantGaramond_500Medium.ttf"
SERIF_EN_I = "CormorantGaramond_500Medium_Italic.ttf"


# ---------------------------------------------------------------- easing
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease_io(x):
    x = clamp(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ease_out(x):
    return 1 - (1 - clamp(x)) ** 3


def ease_sine(x):
    return -(math.cos(math.pi * clamp(x)) - 1) / 2


# ---------------------------------------------------------------- 素材
def load_source(name):
    im = Image.open(os.path.join(IMG_DIR, name)).convert("RGB")
    im = im.resize((round(im.width * SRC_H / im.height), SRC_H), Image.LANCZOS)
    a = np.asarray(im).astype(np.float32) / 255.0
    # 軽いグレーディング：コントラストを少し上げ、暖色寄りに
    a = a + (a * a * (3 - 2 * a) - a) * 0.35
    a[..., 0] *= 1.03
    a[..., 2] *= 0.95
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


# 8枚を選定（重複・類似カットは除外）
# cx/cy: 切り出し中心（0〜1）、z: ズーム倍率。開始→終了で動かす
SHOTS = [
    dict(img="_MG_0943a.jpg", t0=0.0, t1=4.0, cx=(0.36, 0.50), z=(1.18, 1.0)),
    dict(img="_MG_0931a.jpg", t0=3.4, t1=7.6, cx=(0.28, 0.50), z=(1.0, 1.08)),
    dict(img="_MG_0912a.jpg", t0=7.0, t1=10.8, cx=(0.52, 0.36), z=(1.06, 1.14)),
    dict(img="_MG_0955a.jpg", t0=10.2, t1=13.8, cx=(0.40, 0.58), z=(1.12, 1.0)),
    dict(img="_MG_0676a.jpg", t0=13.2, t1=16.8, cx=(0.47, 0.47), z=(1.0, 1.22)),
    dict(img="_MG_0603a.jpg", t0=16.4, t1=20.2, cx=(0.47, 0.47), z=(1.32, 1.08)),
    dict(img="_MG_0659a.jpg", t0=19.6, t1=23.2, cx=(0.36, 0.56), z=(1.05, 1.12)),
    dict(img="_MG_0635a.jpg", t0=22.6, t1=30.0, cx=(0.36, 0.62), z=(1.0, 1.16)),
]

# 隣り合うショット間の切り替え（i → i+1）
TRANSITIONS = ["blur", "leak", "diagonal", "fusuma", "zoomthrough", "sudare", "pushup"]


def build_globals():
    g = {}
    g["src"] = [load_source(s["img"]) for s in SHOTS]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    nx, ny = (xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2)
    vig = 1 - 0.30 * np.clip(nx ** 2 * 0.8 + ny ** 2 * 0.6, 0, 1.6)
    edge = np.clip(1 - np.maximum(0, (np.abs(ny) - 0.70) / 0.30) * 0.35, 0, 1)
    g["vignette"] = (vig * edge)[..., None]
    g["xx"], g["yy"] = xx, yy
    g["diag"] = ((xx / W) * 0.7 + (yy / H) * 0.6) / 1.3
    # 文字を読みやすくするための左右・中央の陰影
    side = np.clip(1 - xx / (W * 0.55), 0, 1) ** 1.6
    g["shade_L"] = side[..., None]
    g["shade_R"] = side[:, ::-1][..., None]
    g["shade_C"] = np.clip(1 - np.sqrt(nx ** 2 * 0.6 + ((yy - H * 0.47) / (H * 0.38)) ** 2), 0, 1)[..., None]
    g["telops"] = build_telops()
    return g


G = None


def init_worker():
    global G
    G = build_globals()


# ---------------------------------------------------------------- ショット描画
def shot_frame(i, t, extra_zoom=1.0):
    s, src = SHOTS[i], G["src"][i]
    u = ease_sine((t - s["t0"]) / (s["t1"] - s["t0"]))
    z = (s["z"][0] + (s["z"][1] - s["z"][0]) * u) * extra_zoom
    cx = s["cx"][0] + (s["cx"][1] - s["cx"][0]) * u
    ch = SRC_H / z
    cw = ch * W / H
    x0 = clamp(cx * src.width - cw / 2, 0, src.width - cw)
    y0 = clamp(0.5 * SRC_H - ch / 2, 0, SRC_H - ch)
    im = src.resize((W, H), Image.BICUBIC, box=(x0, y0, x0 + cw, y0 + ch))
    return np.asarray(im).astype(np.float32)


def fast_blur(a, r):
    if r < 0.5:
        return a
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    small = im.resize((W // 4, H // 4), Image.BILINEAR).filter(ImageFilter.GaussianBlur(r / 4))
    return np.asarray(small.resize((W, H), Image.BILINEAR)).astype(np.float32)


# ---------------------------------------------------------------- 切り替え
def tr_blur(A, B, p, t):
    e = ease_io(p)
    return fast_blur(A, 40 * e) * (1 - e) + fast_blur(B, 40 * (1 - e)) * e


def tr_leak(A, B, p, t):
    e = ease_io(p)
    out = A * (1 - e) + B * e
    cx = W * (-0.2 + 1.4 * p)
    d = ((G["xx"] - cx) ** 2 + ((G["yy"] - H * 0.42) * 0.55) ** 2) / (W * 0.55) ** 2
    glow = np.exp(-d)[..., None] * math.sin(math.pi * p) * 0.85
    warm = np.array([255, 170, 90], np.float32)
    return 255 - (255 - out) * (1 - glow * warm / 255)  # スクリーン合成


def tr_diagonal(A, B, p, t):
    s = 0.22
    m = np.clip((ease_io(p) * (1 + s) - G["diag"]) / s, 0, 1)[..., None]
    return A * (1 - m) + B * m


def tr_fusuma(A, B, p, t):
    # 襖が左右に開いて次の場面が現れる
    e = ease_io(p)
    out = B * (0.55 + 0.45 * e)
    off = int(e * W / 2)
    half = W // 2
    if half - off > 0:
        out[:, : half - off] = A[:, off:half]
        out[:, half + off :] = A[:, half : W - off]
        for x in (half - off, half + off):
            x0, x1 = max(0, x - 10), min(W, x + 10)
            out[:, x0:x1] *= 0.45
    return out


def tr_zoomthrough(A, B, p, t):
    e = ease_io(p)
    za = 1 + 0.9 * e ** 2
    im = Image.fromarray(np.clip(A, 0, 255).astype(np.uint8))
    cw, ch = W / za, H / za
    A2 = np.asarray(im.resize((W, H), Image.BILINEAR,
                              box=((W - cw) / 2, (H - ch) / 2, (W + cw) / 2, (H + ch) / 2))).astype(np.float32)
    A2 = fast_blur(A2, 26 * e)
    B2 = fast_blur(B, 26 * (1 - e))
    flash = math.sin(math.pi * p) * 0.18
    out = A2 * (1 - e) + B2 * e
    return out + (255 - out) * flash


def tr_sudare(A, B, p, t):
    # 簾（すだれ）のように縦の短冊が順に降りて切り替わる
    n = 9
    sw = W / n
    m = np.zeros((H, W), np.float32)
    for k in range(n):
        pk = ease_out((p * 1.7 - k * 0.07) / 1.0)
        edge = pk * (H + 120)
        x0, x1 = int(k * sw), int((k + 1) * sw)
        col = np.clip((edge - G["yy"][:, 0]) / 120, 0, 1)
        m[:, x0:x1] = col[:, None]
    m = m[..., None]
    return A * (1 - m) + B * m


def tr_pushup(A, B, p, t):
    e = ease_io(p)
    off = int(e * H)
    out = np.empty_like(A)
    out[: H - off] = A[off:] * (1 - 0.5 * e)
    out[H - off :] = B[: off]
    y = H - off
    if 0 < y < H:
        out[max(0, y - 2) : y + 2] = np.array(GOLD, np.float32)
    return out


TR_FUNCS = dict(blur=tr_blur, leak=tr_leak, diagonal=tr_diagonal, fusuma=tr_fusuma,
                zoomthrough=tr_zoomthrough, sudare=tr_sudare, pushup=tr_pushup)


# ---------------------------------------------------------------- テロップ
def glyph(ch, f, fill, rotate=0, shadow=True):
    size = f.size
    pad = int(size * 0.5)
    box = int(size * 1.4) + pad * 2
    im = Image.new("RGBA", (box, box), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((box / 2, box / 2), ch, font=f, fill=fill + (255,), anchor="mm")
    if rotate:
        im = im.rotate(rotate, resample=Image.BICUBIC)
    if shadow:
        sh = Image.new("RGBA", im.size, (0, 0, 0, 0))
        sh.putalpha(im.getchannel("A").filter(ImageFilter.GaussianBlur(size * 0.12)).point(lambda v: v * 0.75))
        im = Image.alpha_composite(sh, im)
    return im


def line_image(text, f, fill, tracking, shadow=True):
    widths = [f.getlength(c) for c in text]
    total = sum(widths) + tracking * (len(text) - 1)
    pad = int(f.size * 0.6)
    im = Image.new("RGBA", (int(total) + pad * 2, int(f.size * 1.6) + pad * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    x = pad
    for c, w in zip(text, widths):
        d.text((x, im.height / 2), c, font=f, fill=fill + (255,), anchor="lm")
        x += w + tracking
    if shadow:
        sh = Image.new("RGBA", im.size, (0, 0, 0, 0))
        sh.putalpha(im.getchannel("A").filter(ImageFilter.GaussianBlur(f.size * 0.15)).point(lambda v: v * 0.7))
        im = Image.alpha_composite(sh, im)
    return im


class Telop:
    """文字ごとに時間差でふわっと現れ、まとめて消えるテロップ。"""

    def __init__(self, t_in, t_out, shade=None):
        self.t_in, self.t_out, self.shade = t_in, t_out, shade
        self.items = []  # (rgba ndarray, cx, cy, delay, dur, drift_px, kind)
        self.lines = []  # 伸びる金の罫線 (x0, y0, x1, y1, delay, dur)

    def add(self, im, cx, cy, delay, dur=0.7, drift=22):
        a = np.asarray(im).astype(np.float32)
        self.items.append((a, cx, cy, delay, dur, drift))

    def add_vertical(self, text, f, x, y, delay, step=1.12, gap=0.075, fill=WHITE):
        size = f.size
        for i, c in enumerate(text):
            dx = dy = 0
            rot = 0
            if c in "、。":
                dx, dy = size * 0.58, -size * 0.58
            elif c in "ァィゥェォッャュョぁぃぅぇぉっゃゅょ":
                dx, dy = size * 0.10, -size * 0.10
            elif c in "ー〜":
                rot = -90
            self.add(glyph(c, f, fill, rotate=rot), x + dx, y + i * size * step + dy, delay + i * gap)
        return y + len(text) * size * step

    def add_hline(self, text, f, cx, cy, delay, tracking, fill=WHITE, per_char=True, gap=0.06):
        if not per_char:
            self.add(line_image(text, f, fill, tracking), cx, cy, delay)
            return
        widths = [f.getlength(c) for c in text]
        total = sum(widths) + tracking * (len(text) - 1)
        x = cx - total / 2
        for i, (c, w) in enumerate(zip(text, widths)):
            if c != " ":
                self.add(glyph(c, f, fill), x + w / 2, cy, delay + i * gap)
            x += w + tracking

    def alpha_out(self, t):
        return 1 - ease_io((t - self.t_out) / 0.45)

    def active(self, t):
        return self.t_in <= t <= self.t_out + 0.5

    def render(self, out, t):
        if not self.active(t):
            return out
        ao = self.alpha_out(t)
        if self.shade is not None:
            k = ease_io((t - self.t_in) / 0.6) * ao * 0.55
            out = out * (1 - G["shade_" + self.shade] * k)
        for x0, y0, x1, y1, delay, dur in self.lines:
            e = ease_io((t - self.t_in - delay) / dur)
            if e <= 0:
                continue
            xe, ye = x0 + (x1 - x0) * e, y0 + (y1 - y0) * e
            xa, xb = int(min(x0, xe)), int(max(x0, xe)) + 2
            ya, yb = int(min(y0, ye)), int(max(y0, ye)) + 2
            out[ya:yb, xa:xb] = out[ya:yb, xa:xb] * (1 - ao) + np.array(GOLD, np.float32) * ao
        for a, cx, cy, delay, dur, drift in self.items:
            e = ease_out((t - self.t_in - delay) / dur)
            if e <= 0:
                continue
            al = e * ao
            h, w = a.shape[:2]
            x0 = int(round(cx - w / 2))
            y0 = int(round(cy - h / 2 + (1 - e) * drift))
            xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
            if xa >= xb or ya >= yb:
                continue
            src = a[ya - y0 : yb - y0, xa - x0 : xb - x0]
            m = src[..., 3:4] / 255.0 * al
            out[ya:yb, xa:xb] = out[ya:yb, xa:xb] * (1 - m) + src[..., :3] * m
        return out


def scene_telop(t_in, t_out, side, jp, en, y_top=380):
    """縦書きの見出し + 回転させた英字 + 金の罫線。side は 'L' か 'R'。"""
    tp = Telop(t_in, t_out, shade=side)
    fj = font(MINCHO_SB, 78)
    fe = font(SERIF_EN_I, 38)
    x = W - 190 if side == "R" else 190
    ex = x - 92 if side == "R" else x + 92
    tp.lines.append((x, y_top - 150, x, y_top - 40, 0.0, 0.55))
    tp.add_vertical(jp, fj, x, y_top + 20, 0.25)
    en_im = line_image(en, fe, GOLD, 6).rotate(-90, expand=True)
    tp.add(en_im, ex, y_top + 30 + en_im.height / 2 - 40, 0.75, dur=0.9, drift=-30)
    return tp


def build_telops():
    T = []
    # オープニング：店名
    op = Telop(0.45, 3.25, shade="C")
    op.add_hline("SHINSAIBASHI", font(SERIF_EN, 40), W / 2, 742, 0.05, 16, fill=GOLD, gap=0.035)
    op.add_hline("禅園", font(MINCHO_SB, 190), W / 2, 880, 0.35, 70, gap=0.22)
    op.lines.append((W / 2, 1025, W / 2 - 150, 1025, 0.9, 0.7))
    op.lines.append((W / 2, 1025, W / 2 + 150, 1025, 0.9, 0.7))
    op.add_hline("懐石・割烹", font(MINCHO_M, 42), W / 2, 1085, 1.2, 18)
    T.append(op)

    T.append(scene_telop(4.05, 7.0, "R", "和の趣に、異国の情緒を。", "Japanese elegance, exotic charm"))
    T.append(scene_telop(7.55, 10.2, "L", "ランチから、ディナーまで", "Lunch & Dinner"))
    T.append(scene_telop(10.75, 13.2, "R", "ご友人と、ご家族と", "Table seats"))
    T.append(scene_telop(13.75, 16.1, "L", "障子の向こうに、完全個室", "Private rooms"))
    T.append(scene_telop(16.85, 19.6, "R", "お顔合わせ・ご結納に", "Family occasions"))
    T.append(scene_telop(20.05, 22.6, "L", "ご接待・ご会食に", "Business dining"))
    T.append(scene_telop(23.1, 25.9, "R", "最大三十名様のご宴会まで", "Banquets up to 30 guests"))

    # エンドカード
    ed = Telop(26.9, 99, shade=None)
    fj = font(MINCHO_SB, 70)
    ed.add_hline("大切なひとときに、", fj, W / 2, 560, 0.0, 6, gap=0.05)
    ed.add_hline("ふさわしいお席を。", fj, W / 2, 660, 0.35, 6, gap=0.05)
    ed.lines.append((W / 2, 770, W / 2 - 120, 770, 0.9, 0.6))
    ed.lines.append((W / 2, 770, W / 2 + 120, 770, 0.9, 0.6))
    ed.add_hline("禅園", font(MINCHO_SB, 120), W / 2, 880, 1.0, 44, gap=0.15)
    ed.add_hline("ZEN-EN  SHINSAIBASHI", font(SERIF_EN, 34), W / 2, 985, 1.25, 10, fill=GOLD, per_char=False)
    fi = font(MINCHO_M, 36)
    info = [
        "ランチ・ご会食・ご接待・お顔合わせ・ご宴会",
        "テーブル席／完全個室　全70席",
        "御堂筋線「心斎橋駅」徒歩1分",
        "ご予約　06-6241-7027",
    ]
    for k, s in enumerate(info):
        ed.add_hline(s, fi, W / 2, 1110 + k * 66, 1.5 + k * 0.15, 3, per_char=False)
    T.append(ed)
    return T


# ---------------------------------------------------------------- フレーム合成
def render(fi):
    t = fi / FPS
    idx = [i for i, s in enumerate(SHOTS) if s["t0"] <= t < s["t1"] or (i == len(SHOTS) - 1 and t >= s["t0"])]
    if len(idx) == 1:
        out = shot_frame(idx[0], t)
    else:
        i = idx[0]
        a0, a1 = SHOTS[i + 1]["t0"], SHOTS[i]["t1"]
        p = (t - a0) / (a1 - a0)
        out = TR_FUNCS[TRANSITIONS[i]](shot_frame(i, t), shot_frame(i + 1, t), p, t)

    # オープニングの暗転からのフェードイン
    if t < 1.0:
        out = out * ease_out(t / 1.0)
    # エンドカード：最後のカットをぼかして暗くする
    if t > 26.2:
        e = ease_io((t - 26.2) / 1.0)
        out = fast_blur(out, 30 * e) * (1 - 0.62 * e)

    out = out * G["vignette"]
    for tp in G["telops"]:
        out = tp.render(out, t)
    return np.clip(out, 0, 255).astype(np.uint8).tobytes()


def main():
    preview = "--preview" in sys.argv
    if preview:
        init_worker()
        times = [float(a) for a in sys.argv[sys.argv.index("--preview") + 1 :]]
        os.makedirs("/tmp/reel_preview", exist_ok=True)
        for t in times:
            buf = render(int(t * FPS))
            Image.frombytes("RGB", (W, H), buf).save(f"/tmp/reel_preview/f_{t:05.2f}.jpg", quality=85)
        return

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-profile:v", "high",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", OUT]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(os.cpu_count(), initializer=init_worker) as pool:
        for k, buf in enumerate(pool.imap(render, range(N_FRAMES), chunksize=4)):
            ff.stdin.write(buf)
            if k % 90 == 0:
                print(f"{k}/{N_FRAMES}", flush=True)
    ff.stdin.close()
    ff.wait()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
