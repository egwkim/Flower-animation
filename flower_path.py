import heapq
import json
import os

import cv2
import numpy as np

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_as_image(arr, output_name, mask=None, color=None, colormap=None):
    """Save an array as an image."""
    if arr.dtype == bool:
        arr = arr.astype(np.uint8)

    normalized_arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)

    if color is not None:
        img = np.zeros((*arr.shape, 3), dtype=np.uint8)
        for i in range(3):
            img[:, :, i] = (normalized_arr.astype(np.uint16) * color[i] / 255).astype(
                np.uint8
            )
    elif colormap is not None:
        img = cv2.applyColorMap(normalized_arr, colormap)
    else:
        img = normalized_arr

    if mask is not None:
        if img.ndim == 3:
            img[~mask] = np.array([0, 0, 0], dtype=np.uint8)
        else:
            img[~mask] = 0
    cv2.imwrite(os.path.join(OUTPUT_DIR, output_name), img)


class UnionFind:
    """Disjoint Set Union (Union-Find) with path compression and union by rank."""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n

    def find(self, x):
        """Find the root of x with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        """Union two sets. Returns True if a merge happened, False if already in same set."""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False

        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        self.components -= 1
        return True


class Flower:
    def __init__(self, image_path="flower.jpg"):
        self.image_path = image_path

        self.image = cv2.imread(image_path)

        self.cropped_image = None
        self.bbox = None  # x1, y1, x2, y2
        self.flower_mask = None

        self.preprocessed_image = None

        self.gradient_magnitude = None

        self.costmap = None
        self.is_local_max = None

        self.raw_edges = None
        self.edges = None
        self.num_edges = None

        self.pixel_mask = None

        self.path = None

    def detect_flower_region(self):
        """
        Detects the box that contains the flower and return the box and cropped image.
        """
        # Convert to HSV color space and create a mask for pink color
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        lower_pink1 = np.array([0, 50, 50])
        upper_pink1 = np.array([10, 255, 255])
        lower_pink2 = np.array([170, 50, 50])
        upper_pink2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_pink1, upper_pink1)
        mask2 = cv2.inRange(hsv, lower_pink2, upper_pink2)
        pink_mask = cv2.bitwise_or(mask1, mask2)

        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        pink_mask = cv2.morphologyEx(pink_mask, cv2.MORPH_OPEN, kernel)
        pink_mask = cv2.morphologyEx(pink_mask, cv2.MORPH_CLOSE, kernel)

        num_labels, labels = cv2.connectedComponents(~pink_mask)

        # Assume the flower is fully inside the image, so all four corners are connected background.
        # Get the label at the top-left corner
        bg_label = labels[0, 0]

        # Mark everything that is NOT the background as the flower
        self.flower_mask = labels != bg_label

        # Calculate bbox
        ys, xs = np.where(self.flower_mask)
        if ys.size == 0:
            raise ValueError("No pink flower detected in the image.")
        else:
            top = int(ys.min())
            bottom = int(ys.max())
            left = int(xs.min())
            right = int(xs.max())
            self.bbox = (left, top, right + 1, bottom + 1)

        x1, y1, x2, y2 = self.bbox
        self.cropped_image = self.image[y1:y2, x1:x2]
        self.flower_mask = self.flower_mask[y1:y2, x1:x2]

        return self.bbox, self.cropped_image

    def save_mask_image(self, output_name="flower_mask.png"):
        """
        Save the flower mask as an image for visualization.
        """
        if self.flower_mask is None:
            raise ValueError(
                "Flower mask not available. Call detect_flower_region() first."
            )
        save_as_image(self.flower_mask.astype(np.uint8), output_name)

    def preprocess_image(self):
        """
        Preprocess the cropped image for better edge detection.

        """
        if self.cropped_image is None:
            raise ValueError(
                "Cropped image not available. Call detect_flower_region() first."
            )

        gray = cv2.cvtColor(self.cropped_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        blurred[self.flower_mask == 0] = 0
        self.preprocessed_image = cv2.equalizeHist(blurred)
        return self.preprocessed_image

    def save_recolored_image(self):
        """
        Convert to grayscale and then apply red color
        """
        if self.cropped_image is None:
            raise ValueError(
                "Cropped image not available. Call detect_flower_region() first."
            )
        gray = cv2.cvtColor(self.cropped_image, cv2.COLOR_BGR2GRAY)
        recolored = np.zeros_like(self.cropped_image)
        recolored[..., 2] = gray
        save_as_image(recolored, "flower_recolored.png", mask=self.flower_mask)

    def compute_costmap(self, penalty=0.2):
        """
        Generate a costmap based on Canny edge detection.

        The cost of a pixel is the magnitude of the gradient at a pixel,
        multiplied by a penalty factor if it is not a local maximum along the gradient direction.
        """
        if self.cropped_image is None:
            raise ValueError(
                "Cropped image not available. Call detect_flower_region() first."
            )
        if self.flower_mask is None:
            raise ValueError(
                "Flower mask not available. Call detect_flower_region() first."
            )

        if self.preprocessed_image is None:
            self.preprocess_image()

        img = self.preprocessed_image.astype(np.float32)
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gx, gy)
        direction = cv2.phase(gx, gy, angleInDegrees=False)

        h, w = magnitude.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        dx = np.cos(direction)
        dy = np.sin(direction)

        p1 = cv2.remap(
            magnitude,
            xx + dx,
            yy + dy,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        p2 = cv2.remap(
            magnitude,
            xx - dx,
            yy - dy,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        is_local_max = (magnitude >= p1) & (magnitude >= p2)

        flower_pixels = self.flower_mask > 0
        is_local_max &= flower_pixels

        magnitude[~is_local_max] *= penalty
        inverted_magnitude = magnitude.max() - magnitude
        self.costmap = cv2.normalize(
            inverted_magnitude, None, 0, 65535, cv2.NORM_MINMAX
        ).astype(np.uint16)
        self.gradient_magnitude = magnitude
        self.costmap[~flower_pixels] = 0
        self.is_local_max = is_local_max
        return self.costmap

    def save_costmap_image(self, output_name="flower_costmap.png"):
        """
        Visualize the costmap as a heatmap and save the image.
        """
        if self.costmap is None:
            raise ValueError("Costmap not generated. Call compute_costmap() first.")

        inv_costmap = self.costmap.max() - self.costmap
        save_as_image(
            inv_costmap, output_name, mask=self.flower_mask, color=(0, 255, 0)
        )

    def save_costmap_summary_image(self):
        if self.costmap is None:
            raise ValueError("Costmap not generated. Call compute_costmap() first.")
        mask = self.flower_mask.astype(np.uint8)
        hist = cv2.calcHist([self.costmap], [0], mask, [256], [0, 65536])
        hist = hist.flatten()

        # Render histogram to an image and save instead of showing a GUI window
        h, w = 200, 256
        hist_img = np.zeros((h, w, 3), dtype=np.uint8)
        if hist.max() > 0:
            hist_norm = (hist / hist.max() * (h - 1)).astype(int)
        else:
            hist_norm = np.zeros_like(hist, dtype=int)

        for x, val in enumerate(hist_norm):
            cv2.line(hist_img, (x, h - 1), (x, h - 1 - int(val)), (0, 255, 0), 1)

        save_as_image(hist_img, "costmap_histogram.png")

        return hist_img

    def auto_threshold(self, low_percentile=80, high_percentile=95):
        """
        Automatically determine low and high thresholds for hysteresis based on percentiles.
        It only considers pixels makred as local maxima.
        """

        # Calculate median with pixels within the mask only
        if self.preprocessed_image is None:
            raise ValueError(
                "Preprocessed image not available. Call preprocess_image() first."
            )

        candidate_magnitudes = self.gradient_magnitude[self.is_local_max]

        low_threshold = np.percentile(candidate_magnitudes, low_percentile)
        high_threshold = np.percentile(candidate_magnitudes, high_percentile)
        return low_threshold, high_threshold

    def canny_edge_detection(self, low_threshold, high_threshold):
        """
        Perform Canny edge detection on the preprocessed image.
        """
        if self.preprocessed_image is None:
            raise ValueError(
                "Preprocessed image not available. Call preprocess_image() first."
            )

        if low_threshold > high_threshold:
            raise ValueError(
                "Low threshold must be less than or equal to high threshold."
            )

        canny_edges = cv2.Canny(
            self.preprocessed_image,
            float(low_threshold),
            float(high_threshold),
            apertureSize=3,
            L2gradient=True,
        )

        self.raw_edges = canny_edges
        return canny_edges

    def save_raw_edge_image(self, output_name="flower_edges_raw.png"):
        """
        Save the raw edges as an image.
        """
        if self.raw_edges is None:
            raise ValueError(
                "Edge mask not generated. Call canny_edge_detection() first."
            )

        save_as_image(self.raw_edges, output_name, color=(0, 255, 0))

    def save_edge_image(self, output_name="flower_edges.png"):
        """
        Save the detected edges as an image.
        """
        edge_image = np.zeros_like(self.cropped_image)

        if self.edges is None:
            raise ValueError(
                "Processed edges not available. Call process_edges() first."
            )
        rng = np.random.default_rng()
        for edge_id in range(1, self.edges.max() + 1):
            pixel = np.array(rng.integers(0, 255), dtype=np.uint8)
            color = cv2.applyColorMap(pixel, cv2.COLORMAP_HSV)[0, 0]
            edge_image[self.edges == edge_id] = color

        save_as_image(edge_image, output_name)
        return edge_image

    def process_edges(self, min_component_size=160):
        """
        Process edges to prepare for dijkstra merging.

        Find connected components and remove too small components. Save the processed edges in self.edges.
        """
        if self.raw_edges is None:
            raise ValueError(
                "Raw edges not available. Call canny_edge_detection() first."
            )

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            self.raw_edges, connectivity=8
        )

        # Keep background (0) and components that pass the size threshold.
        keep = np.zeros(num_labels, dtype=bool)
        keep[0] = True
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_component_size

        # Build a lookup table from old labels to compacted labels.
        lut = np.zeros(num_labels, dtype=np.int32)
        num_kept = np.count_nonzero(keep)
        lut[keep] = np.arange(num_kept, dtype=np.int32)

        self.edges = lut[labels]
        self.num_edges = num_kept - 1  # Exclude background
        return self.edges

    def dijkstra_merging(self, verbose=False, progress_output_dir="dijkstra_progress"):
        """
        Multi-source Dijkstra merging to find the approximate optimal solution.
        """

        visited = np.zeros_like(self.edges, dtype=bool)
        comp_id = np.zeros_like(self.edges, dtype=np.int32)
        prev_x = np.full_like(self.edges, -1, dtype=np.int32)
        prev_y = np.full_like(self.edges, -1, dtype=np.int32)
        # Initialize costs to a large value so relaxations work correctly
        INF = np.iinfo(np.int32).max // 4
        costs = np.full_like(self.edges, INF, dtype=np.int32)

        uf = UnionFind(self.num_edges + 1)  # +1 for background

        num_edges = self.num_edges
        pixels = np.argwhere(self.edges > 0)
        pixels_mask = self.edges > 0
        # initialize component ids and seed costs for all edge pixels
        comp_id[pixels[:, 0], pixels[:, 1]] = self.edges[pixels[:, 0], pixels[:, 1]]
        costs[pixels[:, 0], pixels[:, 1]] = 0
        q = [(0, int(self.edges[y, x]), y, x) for y, x in pixels]
        heapq.heapify(q)

        max_y, max_x = self.edges.shape[0], self.edges.shape[1]

        def add_path_to_pixels_mask(end_y, end_x):
            y, x = end_y, end_x
            cnt = 0

            new_path = np.zeros_like(pixels_mask)

            while True:
                pixels_mask[y, x] = True
                new_path[y, x] = True
                cnt += 1
                if prev_x[y, x] == -1 and prev_y[y, x] == -1:
                    break
                y, x = prev_y[y, x], prev_x[y, x]
            if verbose:
                print(
                    f"Added path of length {cnt} to pixels. Total pixels: {len(np.argwhere(pixels_mask))}"
                )
            return new_path

        if verbose > 1:
            os.makedirs(progress_output_dir, exist_ok=True)

        def save_progress_image():
            progress_image = np.zeros((*self.edges.shape, 3), dtype=np.uint8)

            flower_pixels = self.flower_mask > 0
            cost_values = cv2.normalize(
                costs.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)

            progress_image[..., 1][flower_pixels] = 255 - cost_values[flower_pixels]
            progress_image[pixels_mask] = np.array([255, 255, 255], dtype=np.uint8)

            save_as_image(
                progress_image,
                f"{progress_output_dir}/{num_edges:03d}.png",
                mask=self.flower_mask,
            )

        save_progress_image()

        while num_edges > 1 and q:
            cost, id, y, x = heapq.heappop(q)
            root_id = uf.find(id)

            if visited[y, x]:
                continue
            visited[y, x] = True

            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < max_y and 0 <= nx < max_x and self.flower_mask[ny, nx]:

                        # Check if the neighbor is unvisited
                        if not visited[ny, nx]:
                            new_cost = int(cost) + int(self.costmap[ny, nx])
                            # Relaxation: only update if we found a better path
                            if new_cost < costs[ny, nx]:
                                comp_id[ny, nx] = id
                                costs[ny, nx] = new_cost
                                prev_x[ny, nx] = x
                                prev_y[ny, nx] = y
                                heapq.heappush(q, (new_cost, id, ny, nx))

                        # Check if it belongs to a different component
                        else:
                            neighbor_root_id = uf.find(comp_id[ny, nx])
                            if neighbor_root_id != root_id:
                                if uf.union(root_id, neighbor_root_id):
                                    num_edges -= 1
                                    path1 = add_path_to_pixels_mask(ny, nx)
                                    path2 = add_path_to_pixels_mask(y, x)
                                    if verbose:
                                        print(
                                            f"Merged components {root_id} and {neighbor_root_id}. Remaining: {num_edges}"
                                        )
                                        if verbose > 1:
                                            save_progress_image()
                                            save_as_image(
                                                path1 | path2,
                                                f"{progress_output_dir}/path_{num_edges:03d}.png",
                                            )
        self.pixel_mask = pixels_mask
        return self.pixel_mask

    def save_merged_image(self, output_name="flower_edges_merged.png"):
        """
        Save the final path as an image.
        """
        if self.pixel_mask is None:
            raise ValueError("Path not computed. Call compute_path() first.")

        save_as_image(self.pixel_mask, output_name, color=(0, 255, 0))

    def save_merged_with_preprocessed(
        self, output_name="flower_edges_overlaid.png", alpha=0.3
    ):
        """
        Save the final path overlaid on the preprocessed image for better visualization.
        """
        if self.pixel_mask is None:
            raise ValueError("Path not computed. Call compute_path() first.")
        if self.preprocessed_image is None:
            raise ValueError(
                "Preprocessed image not available. Call preprocess_image() first."
            )

        # Create a color version of the preprocessed image
        img = np.zeros((*self.preprocessed_image.shape, 3), dtype=np.uint8)
        img[..., 2] = self.preprocessed_image

        edges = np.zeros_like(img, dtype=np.uint8)
        edges[self.pixel_mask] = np.array([0, 255, 0], dtype=np.uint8)

        cv2.addWeighted(img, alpha, edges, 1 - alpha, 0, img)

        save_as_image(img, output_name)

    def load_merged_image(self, input_name="flower_edges_merged.png"):
        """
        Load a merged image and set the pixel mask accordingly.
        """
        img = cv2.imread(os.path.join(OUTPUT_DIR, input_name))
        if img is None:
            raise ValueError(f"Image {input_name} not found in output directory.")
        self.pixel_mask = np.any(img > 0, axis=2)

    def greedy_search(self, n=1000):
        """
        Greedy search to find the lowest cost pixel reachable. Return the list of pixels visited in order.

        Arguments:
            n: Number of pixels to search.
        """
        if self.costmap is None:
            raise ValueError("Costmap not generated. Call generate_cost() first.")

    def save_greedy_animation(self, output_name="flower_greedy_animation.mp4"):
        """
        Animate the order of pixels visited by the greedy search and save the animation as a video.
        """

    def compute_path(self):
        """
        Generate a path from given set of pixels using right hand rule.
        """
        if self.pixel_mask is None:
            raise ValueError("Path not computed. Call dijkstra_merging() first.")

        mask = self.pixel_mask

        def is_valid(pos):
            y, x = pos
            return 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]

        # Select a pixel to start
        # Find the top left most pixel, then go down until we meet the bottom edge.
        start = np.argwhere(mask)[0]
        while mask[start[0] + 1, start[1]]:
            start[0] += 1

        # Orientations
        oris = np.array(
            [
                (1, 0),
                (1, 1),
                (0, 1),
                (-1, 1),
                (-1, 0),
                (-1, -1),
                (0, -1),
                (1, -1),
            ]
        )

        # Start from facing downwards
        ori = 0  # orientation id, initially facing downwards

        path = [start.copy()]
        pos = start

        # We will rotate the orientation ccw to find the next pixel
        # After finding the next pixel, invert the orientation and repeat the process
        while True:
            while is_valid(pos + oris[ori]) and not mask[tuple(pos + oris[ori])]:
                ori = (ori + 1) % 8
            pos = pos + oris[ori]
            if np.array_equal(pos, start):
                break
            if is_valid(pos):
                path.append(pos.copy())
            ori = (ori + 4) % 8  # invert orientation
            ori = (ori + 1) % 8  # rotate ccw for next traversal
        self.path = path
        return path

    def save_path(self, output_name="flower_path.json"):
        """
        Save the path as a JSON file containing the list of pixel coordinates.
        """
        if self.path is None:
            raise ValueError("Path not computed. Call compute_path() first.")

        path_tuple = [(int(p[0]), int(p[1])) for p in self.path]
        with open(os.path.join(OUTPUT_DIR, output_name), "w") as f:
            json.dump(path_tuple, f, separators=(",", ":"))


def main():
    image_path = "flower.jpg"

    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found in current directory")
        return

    f = Flower(image_path)
    print("Detecting flower region and preprocessing image...")
    _, cropped = f.detect_flower_region()
    save_as_image(cropped, "flower_cropped.png")
    f.save_mask_image()

    f.save_recolored_image()

    preprocessed = f.preprocess_image()
    save_as_image(preprocessed, "flower_preprocessed.png")

    print("Computing costmap...")
    f.compute_costmap()
    f.save_costmap_image()
    f.save_costmap_summary_image()

    print("Detecting and processing edges...")
    f.canny_edge_detection(*f.auto_threshold(90, 97.5))
    f.save_raw_edge_image()
    f.process_edges(30)
    f.save_edge_image()

    print("Performing Dijkstra merging...")
    f.dijkstra_merging(verbose=True)
    f.save_merged_image()
    f.save_merged_with_preprocessed()

    print("Finding closed path...")
    f.compute_path()
    f.save_path()


if __name__ == "__main__":
    main()
