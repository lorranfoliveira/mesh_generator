from dxf_reader import DxfReader
from mesh_generator import MeshGenerator

if __name__ == '__main__':
    dxf = DxfReader('hook.dxf')
    mesh = MeshGenerator(dxf.polygon(), 10)
    # mesh.draw()
    mesh.draw()
