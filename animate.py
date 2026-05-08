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


class FourierDraw(ZoomedScene):
    def __init__(self, **kwargs):
        super().__init__(
            zoom_factor=0.2,
            zoomed_display_height=3,
            **kwargs,
        )

    def construct(self):
        # How many steps in a full traversal of the path
        steps = 90 * 60 * 20
        # After slowing down, t increases 1/(90*60) per frame
        # This draws 20 line segments per frame

        # How many vectors to use when computing the path
        compute_vectors = 500
        # How many arrows and circles to render
        render_vectors = 100

        draw_scale = 3.2

        zoomed_camera = self.zoomed_camera
        zoomed_display = self.zoomed_display
        frame = zoomed_camera.frame
        zoomed_frame = zoomed_display.display_frame
        zd_rect = BackgroundRectangle(
            zoomed_display, fill_opacity=0, buff=MED_SMALL_BUFF
        )
        self.add_foreground_mobject(zd_rect)
        unfold_camera = UpdateFromFunc(
            zd_rect, lambda rect: rect.replace(zoomed_display)
        )

        original_capture_mobjects = zoomed_camera.capture_mobjects

        def custom_capture_mobjects(mobjects, **kwargs):
            for mob in mobjects:
                if isinstance(mob, Arrow):
                    mob.save_state()
                    mob.set_stroke(width=0.5)
                elif isinstance(mob, Circle):
                    mob.save_state()
                    mob.set_opacity(0)

            original_capture_mobjects(mobjects, **kwargs)

            for mob in mobjects:
                if isinstance(mob, (Arrow, Circle)):
                    mob.restore()

        zoomed_camera.capture_mobjects = custom_capture_mobjects

        pts = load_path_points(PATH_JSON)
        pts, center, max_size = transform_points(pts)

        n = len(pts)

        render_vectors = min(render_vectors, n)

        z = pts[:, 1] + 1j * pts[:, 0]

        coeffs_original = np.fft.fft(z) / n
        coeffs = coeffs_original * draw_scale
        freqs = np.fft.fftfreq(n, d=1 / n)

        # Sort by frequency magnitude for visual clarity
        sorted_indices = np.argsort(np.abs(freqs))
        coeffs = coeffs[sorted_indices]
        freqs = freqs[sorted_indices]

        radii = np.abs(coeffs)

        # Load and add darkened background image
        bg_image = ImageMobject("output/flower_recolored.png")
        bg_image.move_to(ORIGIN)
        bg_image.set_opacity(0.7)  # Darken to emphasize animation
        bg_image.scale((1080 / 8) / max_size * draw_scale)
        self.add(bg_image)

        t = ValueTracker(0.0)

        circles = VGroup()
        arrows = VGroup()
        path = VMobject(color=WHITE, stroke_width=2.0)

        center = np.array([0.0, 0.0, 0.0])
        for i in range(render_vectors):
            coeff = coeffs[i]
            radius = radii[i]
            end = center + complex_to_R3(coeff)
            if radius > 1e-2:
                circles.add(
                    Circle(radius=radius, stroke_width=1.2, color=BLUE_D).move_to(
                        center
                    )
                )
            if radius > 1e-4:
                arrows.add(Arrow(center, end, stroke_width=2.0, buff=0, color=WHITE))
            center = end

        def compute_position(tt):
            coords = np.array([0.0, 0.0, 0.0])
            for i in range(compute_vectors):
                coeff = coeffs[i]
                freq = freqs[i]
                z = coeff * np.exp(1j * TAU * freq * tt)
                coords += complex_to_R3(z)
            return coords

        path.prev_t = 0.0
        path.start_new_path(compute_position(0.0))

        def update_path(self):
            cur_t = min(t.get_value(), 1.0)
            prev_t = path.prev_t
            if cur_t < prev_t:
                return

            # Sample i/steps where prev_t < i/steps <= cur_t
            sample = np.arange(int(prev_t * steps) + 1, int(cur_t * steps) + 1) / steps
            path.add_points_as_corners([compute_position(tt) for tt in sample])
            path.prev_t = cur_t

        def update_chain(self):
            tt = t.get_value()
            center = np.array([0.0, 0.0, 0.0])
            circle_cnt = 0
            vector_cnt = 0
            for i in range(render_vectors):
                coeff = coeffs[i]
                radius = radii[i]
                freq = freqs[i]
                z = coeff * np.exp(1j * TAU * freq * tt)
                end = center + complex_to_R3(z)
                if radius > 1e-2:
                    circles[circle_cnt].move_to(center)
                    circle_cnt += 1
                if radius > 1e-4:
                    arrows[vector_cnt].put_start_and_end_on(center, end)
                    vector_cnt += 1
                center = end
            frame.move_to(center)

        bg_image.add_updater(update_chain)
        path.add_updater(update_path)

        self.add(circles)
        self.add(arrows)
        self.add(path)

        self.play(t.animate.set_value(0.5), run_time=15, rate_func=linear)

        frame_rect = Rectangle(
            width=frame.width,
            height=frame.height,
            color=WHITE,
            stroke_width=zoomed_frame.stroke_width,
        )
        frame_rect.move_to(frame.get_center())
        self.play(Create(frame_rect, introducer=False), run_time=1)
        self.wait(0.5)
        self.activate_zooming(animate=False)
        self.remove(frame_rect)
        self.play(
            self.get_zoomed_display_pop_out_animation(), unfold_camera, run_time=1
        )
        self.wait(0.3)
        self.play(t.animate.set_value(1.0), run_time=45, rate_func=linear)
        self.wait(0.5)
