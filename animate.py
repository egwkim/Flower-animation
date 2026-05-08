import json
import numpy as np
from manim import *

PATH_JSON = "output/flower_path.json"


def load_path_points(path_file):
    with open(path_file, "r") as f:
        return np.array(json.load(f), dtype=float)  # shape: (N, 2) as [y, x]


def transform_points(pts):
    # Center
    center_x = pts[:, 1].max() / 2
    center_y = pts[:, 0].max() / 2
    center = np.array([center_x, center_y])
    shifted = pts - center

    # Normalize
    max_size = np.linalg.norm(shifted, axis=1).max()
    scaled = shifted / max_size

    # invert y-axis to match manim's coordinate system
    scaled[:, 0] *= -1

    return scaled, center, max_size


def fourier_coefficients(z):
    n = len(z)
    coeffs = np.fft.fft(z) / n
    freqs = np.fft.fftfreq(n, d=1 / n)
    pairs = list(zip(freqs, coeffs))
    return pairs


class FourierDraw(Scene):
    def construct(self):
        pts = load_path_points(PATH_JSON)
        pts, center, max_size = transform_points(pts)
        z = pts[:, 1] + 1j * pts[:, 0]
        pairs = fourier_coefficients(z)

        # compute_vectors = min(2000, len(pairs))
        compute_vectors = len(pairs)
        render_vectors = 100
        draw_scale = 3.2

        # Keep biggest terms; this is the key quality/speed tradeoff.
        compute_pairs = sorted(pairs, key=lambda fc: abs(fc[1]), reverse=True)[
            :compute_vectors
        ]
        render_pairs = sorted(pairs, key=lambda fc: abs(fc[1]), reverse=True)[
            :render_vectors
        ]

        # Load and add darkened background image
        bg_image = ImageMobject("output/flower_recolored.png")
        bg_image.move_to(ORIGIN)
        bg_image.set_opacity(0.7)  # Darken to emphasize animation
        bg_image.scale((1080 / 8) / max_size * draw_scale)
        self.add(bg_image)

        t = ValueTracker(0.0)

        # How many steps in a full traversal of the path
        steps = 20000

        def tip_point(tt):
            p = 0 + 0j
            for freq, coef in compute_pairs:
                p += coef * np.exp(TAU * 1j * freq * tt)
            return np.array([draw_scale * p.real, draw_scale * p.imag, 0.0])

        def build_trace():
            trace = VGroup()
            prev_state = {"t": t.get_value(), "point": tip_point(t.get_value())}

            def update_trace(mob):
                # Don't update path after one full loop
                cur_t = min(t.get_value(), 1.0)
                prev_t = prev_state["t"]

                if cur_t <= prev_t:
                    return mob

                # Sample i/steps where prev_t < i/steps <= cur_t
                sample = np.arange(int(prev_t*steps)+1, int(cur_t*steps)+1) / steps

                start_point = prev_state["point"]
                for tt in sample:
                    end_point = tip_point(tt)
                    mob.add(Line(start_point, end_point, stroke_width=2.5, color=WHITE))
                    start_point = end_point

                prev_state["t"] = cur_t
                prev_state["point"] = start_point
                return mob

            trace.add_updater(update_trace)
            return trace

        def build_chain():
            tt = t.get_value()
            center = np.array([0.0, 0.0, 0.0])

            circles = VGroup()
            vectors = VGroup()

            for freq, coef in render_pairs:
                radius = draw_scale * abs(coef)
                phase = np.angle(coef)
                ang = TAU * freq * tt + phase

                end = center + np.array(
                    [radius * np.cos(ang), radius * np.sin(ang), 0.0]
                )

                if radius > 1e-4:
                    circles.add(
                        Circle(radius=radius, stroke_width=1.2, color=BLUE_D).move_to(
                            center
                        )
                    )

                vectors.add(Line(center, end, stroke_width=2.0, color=YELLOW))
                center = end

            return VGroup(circles, vectors)

        chain = always_redraw(build_chain)
        trace = build_trace()

        self.add(trace, chain)

        run_time = 40
        self.play(t.animate.set_value(2.0), run_time=run_time, rate_func=linear)
        self.wait(0.5)
