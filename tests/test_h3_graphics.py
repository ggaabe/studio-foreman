"""The h3 ad graphics (foreman/h3/graphics.py), on generic captures drawn here with Pillow:

  G1  the palette and the logo: tones derived from the brand, the brand's PNG scaled, else a plain name pill
  G2  the screen insert: crop and paint-outs, the page colour around the card, the stack inside the safe zone
      and above the caption band, the button pressed and the ripple at the tap time only
  G3  the price card: lines re-stacked with lead, underline boxes mapped through it, the pop, the pulse, the
      underline wipe
  G4  the end card: footnotes fading in, the URL pulse, a long card pushed faster and more lines shrunk, both
      inside the safe zone
  G5  the headline: at most two lines inside the safe zone, refused when it cannot fit
  G6  bad boxes, texts and files: LayoutError (content) or FileNotFoundError (infra), never a partial file
  G7  each insert encodes exactly n frames, bt709, the shared mp4 timescale, and 6 s of it never reads as a freeze

Frames are random-access (frame(i)), so the motion tests draw only the frames they inspect."""
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest

try:
    from ad_fixtures import FFMPEG
except ImportError:  # pragma: no cover
    from tests.ad_fixtures import FFMPEG

try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    from foreman.h3 import graphics as gfx
except ImportError:  # pragma: no cover
    gfx = None

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUP = "/System/Library/Fonts/Supplemental/"
DISPLAY = next((p for p in (SUP + "Arial Black.ttf", SUP + "Arial Bold.ttf",
                            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf") if os.path.exists(p)), None)
BOLD = next((p for p in (SUP + "Arial Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
             if os.path.exists(p)), None)
needs_media = unittest.skipUnless(gfx and DISPLAY and BOLD, "numpy, Pillow and a bold font are needed for the graphics")
needs_ffmpeg = unittest.skipUnless(gfx and DISPLAY and BOLD and FFMPEG, "numpy, Pillow, a font and ffmpeg are needed")

BRAND = {"name": "ExampleCo", "bg": "#101418", "accent": "#2563EB", "accent_pressed": "#1D4ED8",
         "highlight": "#FACC15", "text": "#FFFFFF", "muted": "#E5E7EB", "logo_png": None}
ACCENT, PRESSED = (37, 99, 235), (29, 78, 216)
FONTS = {"display": DISPLAY, "bold": BOLD, "caption": DISPLAY, "logo": None}
CROP, PAINT, TAP = [50, 190, 550, 630], [[95, 290, 420, 332, "#FFFFFF"]], [120, 420, 480, 490]
LINES = ["Plans start at $9.50 per form", "One form, everything included", "Upload, check and pay online",
         "Filed for you the same day", "Questions? Write to us any time"]
BASES = [80 + 34 * j for j in range(5)]          # baselines of the pricing lines
CARD_FILL = (250, 250, 250)


def _f(size):
    return ImageFont.truetype(BOLD, size)


def home_capture(path: str) -> str:
    """A light page: a white panel with a heading, a claim line (painted out) and a wide accent button."""
    im = Image.new("RGB", (600, 900), (246, 246, 248))
    d = ImageDraw.Draw(im)
    d.rectangle((60, 200, 539, 619), fill=(255, 255, 255), outline=(220, 220, 226), width=2)
    d.text((100, 260), "Upload a file", font=_f(34), fill=(17, 24, 39), anchor="ls")
    d.text((100, 320), "Fast and secure", font=_f(24), fill=(107, 114, 128), anchor="ls")
    d.rectangle((TAP[0], TAP[1], TAP[2] - 1, TAP[3] - 1), fill=ACCENT)
    d.text(((TAP[0] + TAP[2]) / 2, (TAP[1] + TAP[3]) / 2), "Start", font=_f(30), fill=(255, 255, 255), anchor="mm")
    d.text((100, 720), "Outside the crop", font=_f(30), fill=(17, 24, 39), anchor="ls")
    im.save(path)
    return path


def pricing_capture(path: str) -> str:
    im = Image.new("RGB", (600, 400), CARD_FILL)
    d = ImageDraw.Draw(im)
    for line, base in zip(LINES, BASES):
        d.text((40, base), line, font=_f(22), fill=(60, 60, 70), anchor="ls")
    im.save(path)
    return path


def words_box(j: int, before: str, words: str) -> list:
    """[x0, y0, x1, baseline] of `words` following `before` on pricing line j."""
    f = _f(22)
    x0 = 40 + round(f.getlength(before))
    return [x0, BASES[j] - 18, 40 + round(f.getlength(before + words)), BASES[j]]


def price_card(png: str, **over) -> dict:
    tail = 40 + round(_f(22).getlength("Questions? "))
    card = {"big": "$9.50", "lines": ["One form, everything included", "Filed for you"], "png": png,
            "crop": [30, 50, 570, 232], "lead": 10, "pulse_line": 0,
            "paint": [[tail, 190, 570, 232, "#FAFAFA"]],
            "underline": [words_box(0, "Plans start at ", "$9.50 per form"),
                          words_box(1, "One form, ", "everything included")]}
    card.update(over)
    return card


ENDCARD = {"lead": "File it from", "big": "$9.50", "lines": ["One form", "everything included"],
           "footnote": ["No account needed", "for a single form."], "url": "example.com"}


def drawn_box(im, i, n, brand=BRAND):
    """Bounds of the pixels drawn over the bare background of frame i of n."""
    d = np.abs(np.asarray(im).astype(np.int16) - np.asarray(gfx.brand_bg(i, n, brand)).astype(np.int16)).sum(2) > 3
    ys, xs = np.where(d.any(1))[0], np.where(d.any(0))[0]
    return (int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1) if len(ys) else None


def inside(b, box) -> bool:
    return b[0] >= box[0] and b[1] >= box[1] and b[2] <= box[2] and b[3] <= box[3]


class GraphicsCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="foreman-h3-graphics-")
        self.home = home_capture(os.path.join(self.tmp, "home.png"))
        self.pricing = pricing_capture(os.path.join(self.tmp, "pricing.png"))
        self.capture = {"png": self.home, "crop": CROP, "paint": PAINT, "tap": TAP, "tap_color": None}

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


@needs_media
class G1PaletteAndLogo(GraphicsCase):
    def test_tones_derive_from_the_brand(self):
        p = gfx.palette({"bg": "#0B1220", "accent": "#2563EB", "accent_pressed": "#1D4ED8", "highlight": "#FFD700",
                         "text": "#FFFFFF", "muted": "#E5E7EB"})
        self.assertEqual((p["bg"], p["accent"], p["highlight"]), ((11, 18, 32), (37, 99, 235), (255, 215, 0)))
        self.assertEqual(p["pill"], (20, 27, 40))         # bg 3.7% of the way to text
        self.assertEqual(p["border"], (41, 47, 59))       # bg 12.2% of the way to text
        self.assertEqual(p["headline"], (50, 108, 236))   # the accent 6% lighter
        light = gfx.palette(dict(BRAND, bg="#FFFFFF", text="#000000"))
        self.assertLess(light["border"][0], 255)  # a light brand's hairline is darker than its background
        with self.assertRaises(gfx.LayoutError):
            gfx.palette(dict(BRAND, accent="red"))
        with self.assertRaises(gfx.LayoutError):
            gfx.palette({k: v for k, v in BRAND.items() if k != "muted"})

    def test_the_brand_png_is_scaled_and_capped(self):
        png = os.path.join(self.tmp, "logo.png")
        im = Image.new("RGBA", (300, 100), (0, 0, 0, 0))
        ImageDraw.Draw(im).rounded_rectangle((0, 0, 299, 99), 50, fill=(200, 30, 90, 255))
        im.save(png)
        brand = dict(BRAND, logo_png=png)
        self.assertEqual(gfx.logo(brand, FONTS, 62).size, (186, 62))
        self.assertEqual(gfx.logo(brand, FONTS, 110, max_w=120).size, (120, 40))
        with self.assertRaises(FileNotFoundError):
            gfx.logo(dict(BRAND, logo_png=os.path.join(self.tmp, "none.png")), FONTS, 62)

    def test_without_a_png_the_name_is_drawn_on_a_pill(self):
        pill = gfx.logo(BRAND, FONTS, 110)
        self.assertEqual(pill.height, 110)
        q = np.asarray(pill).astype(int)
        self.assertEqual(q[0, 0, 3], 0)  # rounded ends: transparent corners
        tone = np.array(gfx.palette(BRAND)["logo_text"])
        self.assertGreater(((np.abs(q[..., :3] - tone).sum(2) < 30) & (q[..., 3] > 250)).sum(), 400)
        self.assertLessEqual(gfx.logo(dict(BRAND, name="A much longer example brand name"), FONTS, 110, 400).width, 400)
        self.assertIsNone(gfx.logo(dict(BRAND, name=""), FONTS, 110))


@needs_media
class G2ScreenInsert(GraphicsCase):
    def test_the_card_is_the_painted_crop_on_the_page_colour(self):
        pill, pxy, card, cxy, size, bc, pcard, c = gfx.screen_layout(self.capture, BRAND, FONTS)
        self.assertEqual(card.size, size)
        s = (size[0] - 28) / (CROP[2] - CROP[0])
        q = np.asarray(card).astype(int)
        # the claim line under the paint box is gone: plain white
        x0, y0 = round(14 + (PAINT[0][0] - CROP[0]) * s) + 3, round(14 + (PAINT[0][1] - CROP[1]) * s) + 3
        x1, y1 = round(14 + (PAINT[0][2] - CROP[0]) * s) - 3, round(14 + (PAINT[0][3] - CROP[1]) * s) - 3
        self.assertTrue((q[y0:y1, x0:x1, :3] == 255).all())
        # the heading above it is still there
        hy = round(14 + (240 - CROP[1]) * s)
        self.assertLess(q[hy - 8:hy + 8, x0:x1, :3].min(), 100)
        # the card's padding is the page colour sampled around the crop
        self.assertEqual(tuple(q[size[1] // 2, 7, :3]), (246, 246, 248))
        # the pressed card differs only around the button
        dq = np.abs(np.asarray(pcard).astype(int) - q).sum(2) > 0
        ys, xs = np.where(dq)
        bx0, by0 = 14 + (TAP[0] - CROP[0] - 2) * s, 14 + (TAP[1] - CROP[1] - 2) * s
        bx1, by1 = 14 + (TAP[2] - CROP[0] + 2) * s, 14 + (TAP[3] - CROP[1] + 2) * s
        self.assertTrue(xs.min() >= bx0 - 6 and xs.max() <= bx1 + 6 and ys.min() >= by0 - 6 and ys.max() <= by1 + 6)
        self.assertEqual(pill.height, 62)

    def test_tap_presses_the_button_for_4_frames_and_ripples_for_8(self):
        n, tap_at = 48, 0.5
        rf = 12
        fr = gfx.screen_frames(self.capture, n, BRAND, FONTS, tap_at=tap_at)
        zt = gfx._screen_push(n)
        _, _, _, _, _, (bx, by), _, c = gfx.screen_layout(self.capture, BRAND, FONTS, zoom=1 + zt)
        frames = {i: fr(i) for i in (0, rf - 1, rf, rf + 3, rf + 4, rf + 8, n - 1)}

        def at(i):
            z = 1 + zt * i / (n - 1)
            return c[0] + (bx - c[0]) * z, c[1] + (by - c[1]) * z

        for i, want in ((rf - 1, ACCENT), (rf, PRESSED), (rf + 3, PRESSED), (rf + 4, ACCENT)):
            px, py = at(i)
            side = np.asarray(frames[i]).astype(int)[round(py) - 22, round(px) - 200]  # button fill, clear of the ripple
            self.assertLessEqual(np.abs(side - np.array(want)).sum(), 6, (i, side))

        def glow(i):
            px, py = at(i)
            return np.asarray(frames[i]).astype(float)[round(py) - 50:round(py) + 50, round(px) - 50:round(px) + 50].mean()
        self.assertGreater(glow(rf), glow(rf - 1) + 20)
        self.assertLess(abs(glow(rf + 8) - glow(rf - 1)), 3)
        for i, im in frames.items():
            self.assertTrue(inside(drawn_box(im, i, n), (70, 272, 930, gfx.CAPTION_TOP)), i)

    def test_no_tap_box_means_no_press_and_a_tap_time_is_refused(self):
        cap = dict(self.capture, tap=None)
        fr = gfx.screen_frames(cap, 24, BRAND, FONTS)
        self.assertEqual(fr(0).size, (gfx.W, gfx.H))
        with self.assertRaises(gfx.LayoutError):
            gfx.screen_frames(cap, 24, BRAND, FONTS, tap_at=0.5)

    def test_a_long_insert_pushes_faster_and_still_fits(self):
        self.assertEqual(gfx._screen_push(48), 0.03)
        self.assertGreater(gfx._screen_push(144), 0.05)
        fr = gfx.screen_frames(self.capture, 144, BRAND, FONTS)
        self.assertTrue(inside(drawn_box(fr(143), 143, 144), (70, 272, 930, gfx.CAPTION_TOP)))


@needs_media
class G3PriceCard(GraphicsCase):
    def test_lead_restacks_the_lines_and_moves_the_underlines(self):
        card = price_card(self.pricing)
        im, box = gfx._capture(self.pricing, card["crop"], card["paint"], "price card")
        src = np.asarray(im.crop(box)).astype(int)
        out, fill, unders = gfx._price_crop(card)
        self.assertEqual(fill, CARD_FILL)
        self.assertEqual(out.size, (540, 182 + 4 * 10))
        o = np.asarray(out).astype(int)
        for j, base in enumerate(BASES):
            a, b = base - 50 - 20, base - 50 + 6  # line j's ink rows in the crop
            self.assertTrue(np.array_equal(o[a + 10 * j:b + 10 * j], src[a:b]), j)
        u0, u1 = card["underline"]
        self.assertEqual(unders, [(u0[0] - 30, BASES[0] - 50, u0[2] - 30), (u1[0] - 30, BASES[1] - 50 + 10, u1[2] - 30)])
        # the painted tail of the last line is gone, "Questions?" stays
        tail = card["paint"][0][0] - 30
        self.assertTrue((o[-40:, tail:] == np.array(CARD_FILL)).all())
        self.assertLess(o[BASES[4] - 50 + 40 - 14:BASES[4] - 50 + 40, :tail].min(), 150)
        flat, _, flat_unders = gfx._price_crop(dict(card, lead=0))
        self.assertEqual(flat.size, (540, 182))
        self.assertEqual(flat_unders[1][1], BASES[1] - 50)

    def test_pop_pulse_and_underline_wipe(self):
        n, pf, uf = 48, 12, 24
        card = price_card(self.pricing)
        fr = gfx.price_frames(card, n, BRAND, FONTS, pulse_at=pf / 24, underline_at=uf / 24)
        items, c = gfx.price_layout(card, BRAND, FONTS)
        it = {nm: (s, x, y) for nm, s, x, y in items}
        z = lambda i: 1 + 0.04 * i / (n - 1)
        gold = np.array(gfx.palette(BRAND)["highlight"])

        def line_width(im, i):
            s, x, y = it[("line", 0)]
            ya, yb = c[1] + (y - c[1]) * z(i), c[1] + (y + s.height - c[1]) * z(i)
            band = np.asarray(im).astype(int)[int(ya) - 6:int(yb) + 6].min(2) > 200
            xs = np.where(band.any(0))[0]
            return xs[-1] - xs[0] + 1

        def bar_px(im, i):
            s, x, y = it[("underline", 0)]
            yu = c[1] + (y + 3 - c[1]) * z(i)
            rows = np.asarray(im).astype(int)[int(yu) - 2:int(yu) + 3]
            return int((np.abs(rows - np.array(ACCENT)).sum(2) < 60).any(0).sum())

        f0, f3 = fr(0), fr(3)
        gold_px = [int((np.abs(np.asarray(f).astype(int) - gold).sum(2) < 40).sum()) for f in (f0, f3)]
        self.assertLess(gold_px[0], gold_px[1])  # the price pops in from 0.85
        widths = [line_width(fr(i), i) for i in range(pf - 1, pf + 8)]
        self.assertAlmostEqual(max(widths[1:]) / widths[0], 1.08, delta=0.02)
        self.assertLess(abs(widths[-1] - widths[0] * z(pf + 7) / z(pf - 1)), 3, widths)  # back to rest, pushed
        wipe = {i: bar_px(fr(i), i) for i in (uf - 1, uf, uf + 2, uf + 5)}
        s, _, _ = it[("underline", 0)]
        self.assertEqual(wipe[uf - 1], 0)
        self.assertTrue(0 < wipe[uf] < wipe[uf + 2] < wipe[uf + 5], wipe)
        self.assertLess(abs(wipe[uf + 5] - s.width * z(uf + 5)), 6)
        for i in (0, pf + 3, n - 1):
            self.assertTrue(inside(drawn_box(fr(i), i, n), gfx.SAFE), i)

    def test_no_capture_centres_the_texts_and_events_need_their_targets(self):
        card = {"big": "$9.50", "lines": ["One form"], "png": None, "crop": None, "underline": None, "pulse_line": 0}
        fr = gfx.price_frames(card, 24, BRAND, FONTS, pulse_at=0.2)
        b = drawn_box(fr(10), 10, 24)
        self.assertLess(abs((b[1] + b[3]) / 2 - gfx.GLOW[1]), 40)
        with self.assertRaises(gfx.LayoutError):
            gfx.price_frames(card, 24, BRAND, FONTS, underline_at=0.5)
        with self.assertRaises(gfx.LayoutError):
            gfx.price_frames(dict(card, lines=[]), 24, BRAND, FONTS, pulse_at=0.5)

    def test_a_tall_crop_under_four_lines_shrinks_into_the_safe_zone(self):
        tall = os.path.join(self.tmp, "tall.png")
        Image.new("RGB", (300, 1400), (200, 200, 210)).save(tall)
        card = {"big": "$9.50", "lines": ["One form", "everything included", "filed for you", "same day"], "png": tall,
                "crop": None, "underline": None, "pulse_line": 3}
        fr = gfx.price_frames(card, 24, BRAND, FONTS)
        for i in (0, 23):
            self.assertTrue(inside(drawn_box(fr(i), i, 24), gfx.SAFE), i)


@needs_media
class G4EndCard(GraphicsCase):
    def test_footnotes_fade_in_and_the_url_pulses(self):
        n, pf = 48, 12
        fr = gfx.endcard_frames(ENDCARD, n, BRAND, FONTS, pulse_at=pf / 24)
        els = {nm: (s, x, y) for nm, s, x, y, _ in gfx.endcard_layout(ENDCARD, BRAND, FONTS)}

        def z(i):
            return 1.04 - 0.04 * gfx._ease_out(i / 5) if i <= 5 else 1 + 0.03 * (i - 6) / (n - 7)
        s, x, y = els[("foot", 0)]
        peak = []
        for i in range(10):
            ya = gfx.GLOW[1] + (y - gfx.GLOW[1]) * z(i)
            d = np.asarray(fr(i)).astype(int) - np.asarray(gfx.brand_bg(i, n, BRAND)).astype(int)
            peak.append(int(d[int(ya) - 4:int(ya) + s.height + 4].max()))
        self.assertTrue(all(p < 15 for p in peak[:3]), peak)
        self.assertGreater(peak[3], 25)
        self.assertGreater(peak[8], 200)
        self.assertTrue(all(peak[i] <= peak[i + 1] + 1 for i in range(2, 9)), peak)
        widths = []
        for i in range(pf - 1, pf + 8):
            q = np.asarray(fr(i)).astype(int)
            xs = np.where((np.abs(q[880:1050] - np.array(ACCENT)).sum(2) < 40).any(0))[0]
            widths.append(xs[-1] - xs[0] + 1)
        self.assertAlmostEqual(max(widths[1:]) / widths[0], 1.06, delta=0.015)
        for i in (0, 3, pf + 3, n - 1):
            self.assertTrue(inside(drawn_box(fr(i), i, n), gfx.SAFE), i)

    def test_the_v6_length_keeps_its_push_and_a_long_card_pushes_faster_inside_the_safe_zone(self):
        self.assertEqual(gfx._endcard_push(53), 0.03)
        self.assertEqual(gfx._endcard_push(61), 0.03)
        self.assertAlmostEqual(gfx._endcard_push(240), 0.03 * 233 / 54)
        fr = gfx.endcard_frames(ENDCARD, 240, BRAND, FONTS)
        for i in (0, 120, 239):
            self.assertTrue(inside(drawn_box(fr(i), i, 240), gfx.SAFE), i)
        a, b = (np.asarray(fr(i)).astype(float) for i in (200, 201))
        self.assertGreater(np.abs(a - b).mean(), 0.1)  # moves more per frame than a 144-frame card at 1.03 (0.065)

    def test_more_lines_shrink_the_card_into_the_safe_zone(self):
        card = dict(ENDCARD, lines=["One form", "everything included", "filed the same day", "no account"],
                    footnote=["No account needed", "for a single form.", "Prices include tax."])
        fr = gfx.endcard_frames(card, 24, BRAND, FONTS)
        for i in (0, 8, 23):
            self.assertTrue(inside(drawn_box(fr(i), i, 24), gfx.SAFE), i)
        for over in ({"url": ""}, {"big": None}, {"lines": ["one", " "]}):
            with self.assertRaises(gfx.LayoutError, msg=str(over)):
                gfx.endcard_layout(dict(ENDCARD, **over), BRAND, FONTS)


@needs_media
class G5Headline(GraphicsCase):
    def test_fits_the_safe_zone_in_two_lines_at_most(self):
        out = gfx.headline("File a form today from the phone in your pocket", os.path.join(self.tmp, "h", "head.png"),
                           BRAND, FONTS)
        im = Image.open(out)
        self.assertEqual((im.size, im.mode), ((gfx.W, gfx.H), "RGBA"))
        a = np.asarray(im)
        ys, xs = np.where(a[..., 3] > 200)
        self.assertTrue(inside((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1), gfx.SAFE))
        ys, xs = np.where(a[..., 3] > 8)
        self.assertTrue(inside((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1), (60, 250, 940, 1240)))
        self.assertEqual(tuple(a[300, gfx.CX - 300 + 30, :3]), gfx.palette(BRAND)["headline"])
        self.assertEqual(os.listdir(os.path.dirname(out)), ["head.png"])

    def test_refuses_what_cannot_fit(self):
        for text in ("   ", "W" * 40):
            with self.assertRaises(gfx.LayoutError):
                gfx.headline(text, os.path.join(self.tmp, "x.png"), BRAND, FONTS)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "x.png")))


@needs_media
class G6BadInput(GraphicsCase):
    def test_boxes_must_be_integers_inside_the_capture(self):
        for over in ({"crop": [0, 0, 700, 100]}, {"crop": [50.0, 190, 550, 630]}, {"crop": [100, 100, 100, 200]},
                     {"tap": [10, 10, 40, 40]}, {"paint": [[0, 0, 10, 10]]}, {"paint": [[0, 0, 10, 10, "white"]]}):
            with self.assertRaises(gfx.LayoutError, msg=str(over)):
                gfx.screen_frames(dict(self.capture, **over), 24, BRAND, FONTS)
        for over in ({"underline": [[0, 0, 20, 20]]}, {"lead": -1}, {"lines": ["a", "b", "c", "d", "e"]}, {"big": " "},
                     {"pulse_line": 5}):
            with self.assertRaises(gfx.LayoutError, msg=str(over)):
                gfx.price_frames(price_card(self.pricing, **over), 24, BRAND, FONTS, pulse_at=0.1)

    def test_missing_files_are_infrastructure(self):
        with self.assertRaises(FileNotFoundError):
            gfx.screen_frames(dict(self.capture, png=os.path.join(self.tmp, "none.png")), 24, BRAND, FONTS)
        with self.assertRaises(FileNotFoundError):
            gfx.endcard_frames(ENDCARD, 24, BRAND, dict(FONTS, bold=os.path.join(self.tmp, "none.ttf")))
        with self.assertRaises(FileNotFoundError):
            gfx.endcard(ENDCARD, 2, os.path.join(self.tmp, "e.mp4"), os.path.join(self.tmp, "no-ffmpeg"), BRAND, FONTS)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "e.mp4")))

    def test_frame_counts_are_positive_integers(self):
        for n in (0, 2.5, -3, True):
            with self.assertRaises(ValueError):
                gfx.endcard_frames(ENDCARD, n, BRAND, FONTS)


def _encode_job(path: str, fn: str, args: list, kw: dict) -> list:
    with open(path, "w") as f:
        json.dump({"fn": fn, "args": args, "kw": kw}, f)
    code = ("import json, sys; sys.path.insert(0, sys.argv[2]); from foreman.h3 import graphics as g; "
            "a = json.load(open(sys.argv[1])); getattr(g, a['fn'])(*a['args'], **a['kw'])")
    return [sys.executable, "-I", "-c", code, path, REPO]


@needs_ffmpeg
class G7Encode(GraphicsCase):
    def test_each_insert_is_n_frames_bt709_and_never_freezes_over_6s(self):
        n = 144
        outs = {k: os.path.join(self.tmp, f"{k}.mp4") for k in ("screen", "price", "endcard")}
        jobs = {"screen": ("screen_insert", [self.capture, n, outs["screen"], FFMPEG, BRAND, FONTS], {"tap_at": 3.0}),
                "price": ("price_insert", [price_card(self.pricing), n, outs["price"], FFMPEG, BRAND, FONTS],
                          {"pulse_at": 1.0, "underline_at": 2.0}),
                "endcard": ("endcard", [ENDCARD, n, outs["endcard"], FFMPEG, BRAND, FONTS], {"pulse_at": 0.8})}
        procs = {k: subprocess.Popen(_encode_job(os.path.join(self.tmp, f"{k}.json"), *job), stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE) for k, job in jobs.items()}  # three at once: ~3x faster
        for k, p in procs.items():
            out, err = p.communicate(timeout=600)
            self.assertEqual(p.returncode, 0, err.decode()[-2000:])
            self.assertEqual(out, b"")  # nothing on stdout: the worker's protocol owns it
        for k, path in outs.items():
            info = gfx.probe(path, FFMPEG)
            self.assertEqual(info["frames"], n, k)
            self.assertAlmostEqual(info["duration"], n / 24, places=3)
            self.assertEqual((info["size"], info["pix_fmt"], info["audio"]), ((1080, 1920), "yuv420p", False), k)
            self.assertIn("bt709", info["color"], k)
            self.assertIn("tv", info["color"], k)
            self.assertEqual(gfx.freezes(path, FFMPEG), [], k)
        self.assertEqual(sorted(os.listdir(self.tmp)), sorted(["endcard.json", "endcard.mp4", "home.png", "price.json",
                                                              "price.mp4", "pricing.png", "screen.json", "screen.mp4"]))

    def test_the_timescale_is_shared_and_failures_leave_no_file(self):
        out = os.path.join(self.tmp, "seg", "e.mp4")
        self.assertEqual(gfx.endcard(ENDCARD, 3, out, FFMPEG, BRAND, FONTS), out)
        r = subprocess.run([FFMPEG, "-hide_banner", "-i", out], capture_output=True, text=True)
        self.assertRegex(r.stderr, r"Video: .*, 24 fps, .*12288 tbn")

        def broken(i):
            if i == 2:
                raise RuntimeError("drawing failed")
            return gfx.brand_bg(i, 4, BRAND)
        bad = os.path.join(self.tmp, "seg", "bad.mp4")
        with self.assertRaises(RuntimeError):
            gfx._encode(broken, 4, bad, FFMPEG, 24)
        fake = os.path.join(self.tmp, "ffmpeg")
        with open(fake, "w") as f:
            f.write("#!/bin/sh\necho 'no encoder here' >&2\nexit 1\n")
        os.chmod(fake, os.stat(fake).st_mode | stat.S_IXUSR)
        with self.assertRaisesRegex(RuntimeError, "no encoder here"):
            gfx._encode(lambda i: gfx.brand_bg(i, 4, BRAND), 4, bad, fake, 24)
        self.assertEqual(os.listdir(os.path.dirname(out)), ["e.mp4"])


if __name__ == "__main__":
    unittest.main()
