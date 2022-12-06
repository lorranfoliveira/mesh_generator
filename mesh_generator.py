from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as geos
from descartes import PolygonPatch
from matplotlib import path, patches
from matplotlib.collections import PatchCollection


def strait_equation(point1: tuple, point2: tuple, x: float, y: float) -> float:
    x1, y1 = point1
    x2, y2 = point2
    dx = x2 - x1
    dy = y2 - y1

    if np.isclose(dx, 0):
        if np.isclose(x, x1):
            return 0
        else:
            return -1
    else:
        m = dy / dx
        return y - y1 - m * (x - x1)


def is_collinear(point1: tuple, point2: tuple, point3: tuple, point4: tuple) -> bool:
    test1 = np.isclose(strait_equation(point1, point2, point3[0], point3[1]), 0)
    test2 = np.isclose(strait_equation(point1, point2, point4[0], point4[1]), 0)
    return test1 and test2


class MeshGenerator:
    def __init__(self, domain: geos.Polygon, d: float):
        # Initial domain
        self.domain = domain
        self.bounds = self.domain.bounds
        self.d = d

        # Expanded domain
        self.expanded_domain: geos.Polygon = self.domain.buffer(self.buffer_distance(), join_style=2)
        self.expanded_boundary = self.expanded_domain.boundary

    def buffer_distance(self, p: float = 0.1) -> float:
        return p * self.d

    def polygon_length_x(self) -> float:
        return self.bounds[2] - self.bounds[0]

    def polygon_length_y(self) -> float:
        return self.bounds[3] - self.bounds[1]

    def background_grid_nodes_number_x(self) -> int:
        return int(np.floor(self.polygon_length_x() / self.d) + 1)

    def background_grid_nodes_number_y(self) -> int:
        return int(np.floor(self.polygon_length_y() / self.d) + 1)

    def background_nodes(self) -> list[tuple]:
        data_points: list[tuple] = []
        x_ini = self.bounds[0]
        y_ini = self.bounds[1]
        nx = self.background_grid_nodes_number_x()
        ny = self.background_grid_nodes_number_y()

        for i in range(ny):
            for j in range(nx):
                point = geos.Point(x_ini + j * self.d, y_ini + i * self.d)
                if self.expanded_domain.contains(point):
                    data_points.append((point.x, point.y))

        return data_points

    def generate_bars(self) -> tuple[list[tuple], list[tuple]]:
        data_points = self.background_nodes()
        combs = list(combinations(np.arange(len(data_points)), 2))
        segments: list[geos.LineString] = []

        # Combine all to all nodes
        data_segments: list[tuple] = []
        c = 1
        for i in range(len(combs)):
            seg = geos.LineString([data_points[combs[i][0]],
                                   data_points[combs[i][1]]])

            if not seg.intersects(self.expanded_boundary):
                segments.append(seg)
                data_segments.append(combs[i])
                c += 1

        return data_segments, data_points

    def non_overlapping_bars(self) -> tuple[list[tuple], list[tuple]]:
        segments, points = self.generate_bars()
        final_segments: list[tuple] = [segments[0]]

        print('Deleting overlapping bars...')
        c = 1
        n = len(segments)
        for seg_ref in segments:
            overlaps = False
            for seg in final_segments:
                # The segments must be different
                if seg != seg_ref:
                    p1_seg = points[seg[0]]
                    p2_seg = points[seg[1]]
                    p1_seg_ref = points[seg_ref[0]]
                    p2_seg_ref = points[seg_ref[1]]

                    # The segments must be the same inclination angle
                    if is_collinear(p1_seg, p2_seg, p1_seg_ref, p2_seg_ref):
                        # Length of elements
                        len_seg = np.linalg.norm([p2_seg[0] - p1_seg[0], p2_seg[1] - p1_seg[1]])
                        len_seg_ref = np.linalg.norm([p2_seg_ref[0] - p1_seg_ref[0], p2_seg_ref[1] - p1_seg_ref[1]])

                        d_11 = np.linalg.norm([p1_seg[0] - p1_seg_ref[0], p1_seg[1] - p1_seg_ref[1]])
                        d_12 = np.linalg.norm([p1_seg[0] - p2_seg_ref[0], p1_seg[1] - p2_seg_ref[1]])
                        d_21 = np.linalg.norm([p2_seg[0] - p1_seg_ref[0], p2_seg[1] - p1_seg_ref[1]])
                        d_22 = np.linalg.norm([p2_seg[0] - p2_seg_ref[0], p2_seg[1] - p2_seg_ref[1]])

                        d_max = max(d_11, d_12, d_21, d_22)
                        len_sum = len_seg + len_seg_ref

                        if d_max < len_sum:
                            overlaps = True
                            break

            if (not overlaps) and (c > 1):
                final_segments.append(seg_ref)

            print(f"progress: {100 * c / n:.2f}%", end="\r")
            c += 1

        print(f'Eliminated bars: {n - len(final_segments)}')
        print(f'Final bars: {len(final_segments)}')

        return final_segments, points

    def draw(self):
        data_segments, data_points = self.non_overlapping_bars()

        fig, ax = plt.subplots(1, 2)
        win = plt.get_current_fig_manager()
        win.window.state('zoomed')
        ax[0].axis('equal')
        ax[1].axis('equal')

        x_min, y_min, x_max, y_max = self.expanded_domain.bounds
        dx = x_max - x_min
        dy = y_max - y_min
        ax[0].set_xlim(x_min - 0.1 * dx, x_max + 0.1 * dx)
        ax[0].set_ylim(y_min - 0.1 * dy, y_max + 0.1 * dy)

        ax[1].set_xlim(x_min - 0.1 * dx, x_max + 0.1 * dx)
        ax[1].set_ylim(y_min - 0.1 * dy, y_max + 0.1 * dy)

        bars = []

        for seg in data_segments:
            vertices = [data_points[seg[0]], data_points[seg[1]]]
            codes = [path.Path.MOVETO, path.Path.LINETO]
            bars.append(patches.PathPatch(path.Path(vertices, codes), linewidth=0.7, edgecolor='black'))

        ax[0].axis('off')
        ax[0].add_patch(PolygonPatch(self.domain))

        ax[1].add_collection(PatchCollection(bars, match_original=True))
        ax[1].axis('off')

        plt.grid(b=None)
        plt.show()

    def mesh_data(self):
        data_segments, data_points = self.non_overlapping_bars()
        return data_segments, data_points

# Domain 0

# Domain 1
# quad = geos.Polygon([(0, 0), (0, 4), (4, 4), (4, 0)])
# hole = scale(geos.Point(2, 2).buffer(1), 0.5)
# polygon = quad.difference(hole)
# msh = MeshGenerator(polygon, 0.5)

# Domain 2
# polygon = geos.Point(0, 0).buffer(2)
# msh = MeshGenerator(polygon, 0.5)

# Domain 3
# p = geos.Point(0, 0).buffer(2)
# polygon = scale(p, 1, 0.5).union(scale(p, 0.5, 1))
# msh = MeshGenerator(polygon, 0.5)

# Domain 4
# p = scale(geos.Point(0, 0).buffer(2), 0.5, 1)
# hole = rotate(p, -45).union(rotate(p, 45))
#
# polygon = p.buffer(2.01).difference(hole)

# msh = MeshGenerator(poly, 5)

# msh.draw()
