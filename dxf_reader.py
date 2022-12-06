import ezdxf
import numpy as np
from descartes import PolygonPatch
from ezdxf.groupby import groupby
from matplotlib import pyplot as plt
from shapely import geometry as geos
from shapely.ops import linemerge


class DxfReader:
    LAYER_GEOMETRY = 'geometry'
    LAYER_HOLES = 'holes'
    LAYER_LOAD = 'load'
    LAYER_TRACKER = 'tracker'
    LAYERS_NAMES = [LAYER_GEOMETRY,
                    LAYER_HOLES,
                    LAYER_LOAD,
                    LAYER_TRACKER]

    def __init__(self, dxf_file_name: str):
        self.dxf_file_name = dxf_file_name
        self.dxf_layers: dict = groupby(entities=ezdxf.readfile(self.dxf_file_name).modelspace(), dxfattrib='layer')

    def lines_dictionary(self) -> dict[str, np.ndarray]:
        lines_dictionary = {}
        for layer in self.dxf_layers:
            if any(layer.startswith(tracked_layers) for tracked_layers in DxfReader.LAYERS_NAMES):
                lines_dictionary[layer] = []
                for obj_layer in self.dxf_layers[layer]:
                    if isinstance(obj_layer, ezdxf.entities.line.Line):
                        lines_dictionary[layer].append(np.round(np.array(obj_layer.dxf.start)[:2], 3).tolist() +
                                                       np.round(np.array(obj_layer.dxf.end)[:2], 3).tolist())
                lines_dictionary[layer] = np.array(lines_dictionary[layer])
        return lines_dictionary

    def geometry_boundary(self) -> list[tuple]:
        lines_layers = self.lines_dictionary()
        lines = []
        for line in lines_layers[DxfReader.LAYER_GEOMETRY]:
            lines.append(geos.LineString([(line[0], line[1]),
                                          (line[2], line[3])]))
        return list(linemerge(geos.MultiLineString(lines)).coords)

    def holes_boundary(self) -> list:
        lines_layers = self.lines_dictionary()

        if DxfReader.LAYER_HOLES in lines_layers:
            lines = []
            for line in lines_layers[DxfReader.LAYER_HOLES]:
                lines.append(geos.LineString([(line[0], line[1]),
                                              (line[2], line[3])]))

            holes = linemerge(geos.MultiLineString(lines))

            if isinstance(holes, geos.LineString):
                return [list(holes.coords)]
            elif isinstance(holes, geos.MultiLineString):
                return [list(geo.coords) for geo in holes.geoms]

    def polygon(self) -> geos.Polygon:
        return geos.Polygon(self.geometry_boundary(), self.holes_boundary())

    def draw(self):
        polygon = self.polygon()

        fig, ax = plt.subplots()
        win = plt.get_current_fig_manager()
        win.window.state('zoomed')
        ax.axis('equal')

        x_min, y_min, x_max, y_max = polygon.bounds
        dx = x_max - x_min
        dy = y_max - y_min
        ax.set_xlim(x_min - 0.1 * dx, x_max + 0.1 * dx)
        ax.set_ylim(y_min - 0.1 * dy, y_max + 0.1 * dy)
        ax.axis('off')
        ax.add_patch(PolygonPatch(polygon))

        plt.grid(b=None)
        plt.show()



