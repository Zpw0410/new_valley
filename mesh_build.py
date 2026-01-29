"""
功能：
- 读取 NS / NE 文件
- 生成支持悬挂点的三角网格
- 在【三角化阶段】直接生成 boundary（基于 NS 拓扑）
- 顶点去重时同步 remap boundary
- 输出可直接用于 anuga.Domain 的 points / triangles / boundary
"""

import numpy as np
from typing import List, Dict, Tuple

# ============================================================
# NS 相关
# ============================================================

class HydroSide:
    def __init__(
        self,
        index: int,
        direction: int,
        bounds: Tuple[float, float, float, float],
        left_grid_index: int,
        right_grid_index: int,
        bottom_grid_index: int,
        top_grid_index: int,
        altitude: float = 0.0,
        type_: int = 0,
        cx: float = 0.0,
        cy: float = 0.0
    ):
        self.index = index
        self.direction = direction
        self.bounds = bounds

        self.left_grid_index = left_grid_index
        self.right_grid_index = right_grid_index
        self.bottom_grid_index = bottom_grid_index
        self.top_grid_index = top_grid_index

        self.altitude = altitude
        self.type = type_

        # ⭐ 正确的边界判定
        if self.direction == 2:  # 垂直边
            self.is_boundary = (left_grid_index == 0 or right_grid_index == 0)
        elif self.direction == 1:# 水平边
            self.is_boundary = (bottom_grid_index == 0 or top_grid_index == 0)

        self.cx = cx
        self.cy = cy



def parse_ns_line(row: list) -> HydroSide:
    index = int(row[0])
    direction = int(row[1])

    left = int(row[2])
    right = int(row[3])
    bottom = int(row[4])
    top = int(row[5])

    length = float(row[6])
    cx, cy, cz = float(row[7]), float(row[8]), float(row[9])
    type_ = int(row[10])

    if direction == 1:  # 水平边
        min_x = cx - length / 2
        max_x = cx + length / 2
        min_y = max_y = cy
    else:               # 垂直边
        min_y = cy - length / 2
        max_y = cy + length / 2
        min_x = max_x = cx

    bounds = (min_x, min_y, max_x, max_y)

    return HydroSide(
        index=index,
        direction=direction,
        bounds=bounds,
        left_grid_index=left,
        right_grid_index=right,
        bottom_grid_index=bottom,
        top_grid_index=top,
        altitude=cz,
        type_=type_,
        cx=cx,
        cy=cy
    )



def load_ns_file(path: str) -> Dict[int, HydroSide]:
    side_map = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split()]
            # 保持你原来的解析：小数当 float，否则 int
            row = [float(p) if "." in p else int(p) for p in parts]
            s = parse_ns_line(row)
            side_map[s.index] = s
    return side_map


# ============================================================
# NE 相关
# ============================================================

class HydroElement:
    def __init__(
        self,
        index: int,
        bounds: Tuple[float, float, float, float],
        left_edges: List[int],
        right_edges: List[int],
        bottom_edges: List[int],
        top_edges: List[int],
        altitude: float,
        type_: int
    ):
        self.index = index
        self.bounds = bounds
        self.left_edges = left_edges
        self.right_edges = right_edges
        self.bottom_edges = bottom_edges
        self.top_edges = top_edges
        self.altitude = altitude
        self.type = type_

    @property
    def center(self):
        min_x, min_y, max_x, max_y = self.bounds
        return ((min_x + max_x) / 2, (min_y + max_y) / 2, self.altitude)


def parse_ne_line(row: list, side_map: Dict[int, HydroSide]) -> HydroElement:
    idx = 0
    index = int(row[idx]); idx += 1
    Ln = int(row[idx]); idx += 1
    Rn = int(row[idx]); idx += 1
    Bn = int(row[idx]); idx += 1
    Tn = int(row[idx]); idx += 1

    left_edges = [int(row[idx+i]) for i in range(Ln)]; idx += Ln
    right_edges = [int(row[idx+i]) for i in range(Rn)]; idx += Rn
    bottom_edges = [int(row[idx+i]) for i in range(Bn)]; idx += Bn
    top_edges = [int(row[idx+i]) for i in range(Tn)]; idx += Tn

    cx = float(row[idx]); idx += 1
    cy = float(row[idx]); idx += 1
    cz = float(row[idx]); idx += 1
    type_ = int(row[idx])

    min_x = side_map[left_edges[0]].bounds[0] if left_edges else cx
    max_x = side_map[right_edges[0]].bounds[2] if right_edges else cx
    min_y = side_map[bottom_edges[0]].bounds[1] if bottom_edges else cy
    max_y = side_map[top_edges[0]].bounds[3] if top_edges else cy

    return HydroElement(
        index,
        (min_x, min_y, max_x, max_y),
        left_edges, right_edges, bottom_edges, top_edges,
        cz, type_
    )


# ============================================================
# 修改 load_ne_file：忽略 altitude==-9999 的元素
# ============================================================

def load_ne_file(path: str, side_map: Dict[int, HydroSide]) -> List[HydroElement]:
    elems = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split()]
            row = [float(p) if "." in p else int(p) for p in parts]
            elem = parse_ne_line(row, side_map)
            if elem.altitude == -9999:
                # 忽略此 element
                continue
            elems.append(elem)
    return elems



# ============================================================
# 三角化 + boundary 生成（核心）
# ============================================================

def tri_area(a, b, c):
    return abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1])) / 2


def elements_to_triangles(elements: List[HydroElement], side_map: Dict[int, HydroSide]):
    """
    返回：
      points: List[(x,y)]
      triangles: List[(i0,i1,i2)]
      elevations: List[z] 与 points 一一对应
      tri_boundary_flags: List[dict] 与 triangles 一一对应，dict: edge_id -> tag (例如 'ocean')
      triangle_types
    """
    points: List[Tuple[float, float]] = []
    elevations: List[float] = []
    triangles: List[Tuple[int, int, int]] = []
    tri_boundary_flags: List[Dict[int, str]] = []
    triangle_types: List[int] = []
    point_index_map = {}

    filtered_edges_count = 0  # 统计被过滤掉的边（按检查次数计数）

    def add_point(x, y, z):
        key = (round(float(x), 9), round(float(y), 9))
        if key in point_index_map:
            return point_index_map[key]
        idx = len(points)
        points.append((float(x), float(y)))
        elevations.append(float(z))
        point_index_map[key] = idx
        return idx

    def side_endpoints(side: HydroSide):
        min_x, min_y, max_x, max_y = side.bounds
        return (min_x, min_y), (max_x, max_y)

    def extract_hanging(edge_ids: List[int], is_vertical: bool):
        """
        如果 edge_ids 中包含多条边（挂点），返回这些边在该侧的内部端点（排除角点），
        以点列表形式返回（按该方向排序）。
        """
        if len(edge_ids) <= 1:
            return []
        pts = []
        # 将每条 side 的两个端点加入集合
        for sid in edge_ids:
            s = side_map[sid]
            a, b = side_endpoints(s)
            pts.append(a)
            pts.append(b)
        # 去重
        pts = list(set(pts))
        # 按 y 排序（竖向）或按 x 排序（横向），然后去掉首尾角点
        if is_vertical:
            pts = sorted(pts, key=lambda p: (p[1], p[0]))
        else:
            pts = sorted(pts, key=lambda p: (p[0], p[1]))
        # 返回排除第一个和最后一个（即剔除角点）的内部点
        return pts[1:-1]

    def is_boundary_refined(edge_ids: List[int]):
        """
        对一条 element 边（可能由多条 side 组成），判断是否应视为 boundary。
        规则：
          - 只有 side.is_boundary 为 True 才考虑
          - 如果 side.cx >= 808411.0 and side.cy >= 837066.0 则踢掉（视为非 boundary）
          - 否则认为是 boundary
        同时统计被踢掉的 side 数量（filtered_edges_count）
        """
        nonlocal filtered_edges_count
        any_keep = False
        for sid in edge_ids:
            side = side_map[sid]
            if not side.is_boundary:
                continue
            # # 如果该 side 的中心点同时大于两个阈值，则踢掉
            # if side.cx >= 808411.0 or side.cy >= 837066.0:
            #     filtered_edges_count += 1
            #     # 这个 side 被过滤（不算 boundary）
            #     continue
            # 否则保留为 boundary
            any_keep = True
        return any_keep

    # 主循环：逐元素生成三角形并同时建立 tri_boundary_flags
    for elem in elements:
        min_x, min_y, max_x, max_y = elem.bounds
        z = elem.altitude

        # 提取每侧的挂点（如果有）
        left_hang   = extract_hanging(elem.left_edges,   True)
        right_hang  = extract_hanging(elem.right_edges,  True)
        bottom_hang = extract_hanging(elem.bottom_edges, False)
        top_hang    = extract_hanging(elem.top_edges,    False)

        has_hanging = (
            len(left_hang) > 0 or
            len(right_hang) > 0 or
            len(bottom_hang) > 0 or
            len(top_hang) > 0
        )

        # 先判断该 element 四边是否为 boundary（并做过滤计数）
        bottom_is_boundary = is_boundary_refined(elem.bottom_edges)
        right_is_boundary  = is_boundary_refined(elem.right_edges)
        top_is_boundary    = is_boundary_refined(elem.top_edges)
        left_is_boundary   = is_boundary_refined(elem.left_edges)

        if has_hanging:
            # 以中心点做扇形剖分
            cx, cy, cz = elem.center
            center_idx = add_point(cx, cy, cz)

            # 构建边界顶点序列并标注每条边所属的 element 边 (bottom/right/top/left)
            boundary_pts = []
            edge_side_labels = []  # edge_side_labels[i] 表示边 boundary_pts[i] -> boundary_pts[i+1] 属于哪个 side

            # bottom 从左到右
            boundary_pts.append((min_x, min_y))
            edge_side_labels.append('bottom')
            for p in bottom_hang:
                boundary_pts.append(p)
                edge_side_labels.append('bottom')
            boundary_pts.append((max_x, min_y))
            edge_side_labels.append('right')

            # right 从下到上
            for p in right_hang:
                boundary_pts.append(p)
                edge_side_labels.append('right')
            boundary_pts.append((max_x, max_y))
            edge_side_labels.append('top')

            # top 从右到左 (注意使用 reversed)
            for p in reversed(top_hang):
                boundary_pts.append(p)
                edge_side_labels.append('top')
            boundary_pts.append((min_x, max_y))
            edge_side_labels.append('left')

            # left 从上到下 (reversed)
            for p in reversed(left_hang):
                boundary_pts.append(p)
                edge_side_labels.append('left')
            # 回到起点时，最后一条边 boundary_pts[-1] -> boundary_pts[0] 属于 bottom? 实际属于 left
            # 当前 edge_side_labels 长度 == len(boundary_pts)，并且 edge i 表示 boundary_pts[i] -> boundary_pts[(i+1)%N]
            # 最后一个 label 已经正确设置为 'left' 上面的 append 操作已设置

            # 把 boundary_pts 转成点索引
            b_idx = [add_point(float(px), float(py), z) for (px, py) in boundary_pts]
            N = len(b_idx)
            # if N < 3:
            #     # 防御：若不足以构成多边形，退回到简单对角拆分（极少见）
            #     p0 = add_point(min_x, min_y, z)
            #     p1 = add_point(max_x, min_y, z)
            #     p2 = add_point(max_x, max_y, z)
            #     p3 = add_point(min_x, max_y, z)
            #     if tri_area(points[p0], points[p1], points[p2]) > 1e-12:
            #         triangles.append((p0, p1, p2))
            #         tri_boundary_flags.append({2: 'ocean' if bottom_is_boundary else None,
            #                                    0: 'ocean' if right_is_boundary else None})
            #     if tri_area(points[p0], points[p2], points[p3]) > 1e-12:
            #         triangles.append((p0, p2, p3))
            #         tri_boundary_flags.append({0: 'ocean' if top_is_boundary else None,
            #                                    2: 'ocean' if left_is_boundary else None})
            #     continue

            # 以中心为顶点逐边形成三角形。对每个三角形，边 boundary_between(b_i,b_{i+1}) 属于 edge_side_labels[i]
            for i in range(N):
                a, b, c = center_idx, b_idx[i], b_idx[(i+1) % N]
                # 排除退化三角形
                if a == b or b == c or c == a:
                    continue
                if tri_area(points[a], points[b], points[c]) <= 1e-12:
                    continue

                triangles.append((a, b, c))
                # 边界标注：三角形顶点顺序 (a,b,c)，边 opposite vertex 0 是 (b,c) -> 就是 polygon 边
                tag_dict = {}
                side_label = edge_side_labels[i]
                if side_label == 'bottom' and bottom_is_boundary:
                    tag_dict[0] = 'ocean'
                elif side_label == 'right' and right_is_boundary:
                    tag_dict[0] = 'ocean'
                elif side_label == 'top' and top_is_boundary:
                    tag_dict[0] = 'ocean'
                elif side_label == 'left' and left_is_boundary:
                    tag_dict[0] = 'ocean'
                # 只在需要时加入字典（否则空字典也行，但保持与之前风格一致）
                tri_boundary_flags.append(tag_dict)
                triangle_types.append(elem.type)
            continue

        # no hanging: simple split (保留你原来的对角线拆分方式)
        p0 = add_point(min_x, min_y, z)
        p1 = add_point(max_x, min_y, z)
        p2 = add_point(max_x, max_y, z)
        p3 = add_point(min_x, max_y, z)

        # triangle 1: (p0, p1, p2)
        if p0 != p1 and p1 != p2 and p2 != p0 and tri_area(points[p0], points[p1], points[p2]) > 1e-12:
            triangles.append((p0, p1, p2))
            tri_boundary_flags.append({
                2: 'ocean' if bottom_is_boundary else None,
                0: 'ocean' if right_is_boundary else None
            })
            triangle_types.append(elem.type)

        # triangle 2: (p0, p2, p3)
        if p0 != p2 and p2 != p3 and p3 != p0 and tri_area(points[p0], points[p2], points[p3]) > 1e-12:
            triangles.append((p0, p2, p3))
            tri_boundary_flags.append({
                0: 'ocean' if top_is_boundary else None,
                1: 'ocean' if left_is_boundary else None
            })
            triangle_types.append(elem.type)

    # 打印统计
    print(f"总共过滤掉了 {filtered_edges_count} 条边（按检查次数计数）")
    return points, triangles, elevations, tri_boundary_flags,triangle_types


# ============================================================
# 去重 + boundary remap
# ============================================================

def remove_duplicate_vertices(points, triangles, elevations, tri_boundary_flags):
    pts = np.asarray(points)
    elevs = np.asarray(elevations)
    tris = np.asarray(triangles, dtype=int)

    keys = np.round(pts, 9)
    _, unique_idx, inverse = np.unique(
        keys.view([('x', float), ('y', float)]),
        return_index=True, return_inverse=True
    )

    new_pts = pts[unique_idx]
    new_elevs = np.zeros(len(unique_idx))
    counts = np.zeros(len(unique_idx))

    for i, ni in enumerate(inverse):
        new_elevs[ni] += elevs[i]
        counts[ni] += 1
    # 防止除以 0（理论上不会发生）
    counts[counts == 0] = 1
    new_elevs /= counts

    new_tris = []
    new_boundary = {}

    for old_tid, tri in enumerate(tris):
        nt = tuple(inverse[i] for i in tri)
        tid = len(new_tris)
        new_tris.append(nt)
        # tri_boundary_flags[old_tid] 可能为 None 或含有若干 (edge_id -> tag)
        flags = tri_boundary_flags[old_tid] if old_tid < len(tri_boundary_flags) else {}
        if flags:
            for edge_id, tag in flags.items():
                if tag:
                    new_boundary[(tid, edge_id)] = tag

    return new_pts, np.array(new_tris), new_elevs, new_boundary


def extract_boundary_edges(triangles, boundary_dict):
    """
    返回：List[(v0, v1)]，每条 boundary 边只保留一次
    """
    edge_set = set()

    for (tid, edge_id), tag in boundary_dict.items():
        if not tag:
            continue
        tri = triangles[tid]

        if edge_id == 0:
            a, b = tri[1], tri[2]
        elif edge_id == 1:
            a, b = tri[2], tri[0]
        elif edge_id == 2:
            a, b = tri[0], tri[1]
        else:
            continue

        # 无向边去重
        edge_set.add(tuple(sorted((a, b))))

    return list(edge_set)


# -------------------------
# PLY 导出（保留）
# -------------------------
def export_ply(points, triangles, elevations, path="mesh_output.ply"):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(triangles)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for i, (x, y) in enumerate(points):
            f.write(f"{x} {y} {elevations[i]}\n")
        for (i,j,k) in triangles:
            f.write(f"3 {i} {j} {k}\n")
    print(f"PLY 已导出: {path}")


def export_boundary_ply(points, elevations, edges, path="mesh_boundary.ply"):
    """
    导出 boundary 为 PLY（line / edge）
    """
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element edge {len(edges)}\n")
        f.write("property int vertex1\nproperty int vertex2\n")
        f.write("end_header\n")

        for i, (x, y) in enumerate(points):
            f.write(f"{x} {y} {elevations[i]}\n")

        for a, b in edges:
            f.write(f"{a} {b}\n")

    print(f"Boundary PLY 已导出: {path}")


def export_ns_boundary_sides_ply(
    side_map: Dict[int, HydroSide],
    path: str = "ns_boundary_sides.ply"
):
    """
    只导出 NS 中 is_boundary == True 的 side，为 line PLY
    """
    vertices = []
    edges = []
    vertex_index = {}

    def add_vertex(x, y, z):
        key = (round(x, 9), round(y, 9), round(z, 9))
        if key in vertex_index:
            return vertex_index[key]
        idx = len(vertices)
        vertices.append((x, y, z))
        vertex_index[key] = idx
        return idx

    for side in side_map.values():
        if not side.is_boundary:
            continue

        min_x, min_y, max_x, max_y = side.bounds
        z = side.altitude

        v0 = add_vertex(min_x, min_y, z)
        v1 = add_vertex(max_x, max_y, z)

        # 避免零长度边
        if v0 != v1:
            edges.append((v0, v1))

    # 写 PLY
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element edge {len(edges)}\n")
        f.write("property int vertex1\nproperty int vertex2\n")
        f.write("end_header\n")

        for x, y, z in vertices:
            f.write(f"{x} {y} {z}\n")

        for a, b in edges:
            f.write(f"{a} {b}\n")

    print(f"NS boundary side PLY 已导出: {path}")


# ============================================================
# 对外接口（你直接用这个）
# ============================================================

def build_mesh_from_ns_ne(ns_path, ne_path):
    side_map = load_ns_file(ns_path)
    elements = load_ne_file(ne_path, side_map)

    pts, tris, elevs, tri_bflags, tri_types = elements_to_triangles(elements, side_map)
    pts, tris, elevs, boundary = remove_duplicate_vertices(
        pts, tris, elevs, tri_bflags
    )

    print(f"mesh: points={len(pts)}, triangles={len(tris)}, boundary edges={len(boundary)}")

    

    export_ply(pts, tris, elevs, "mesh_output.ply")
        # ⭐ 新增：boundary
    boundary_edges = extract_boundary_edges(tris, boundary)
    export_boundary_ply(pts, elevs, boundary_edges, "mesh_boundary.ply")
     # ⭐ 先把 NS 原始 boundary 导出来
    export_ns_boundary_sides_ply(side_map, "ns_boundary_sides.ply")

    # =========================
    # ⭐ 新增：打印 x/y 范围
    # =========================
    pts_array = np.array(pts)
    min_x, max_x = pts_array[:, 0].min(), pts_array[:, 0].max()
    min_y, max_y = pts_array[:, 1].min(), pts_array[:, 1].max()
    print(f"mesh X range: {min_x:.3f} ~ {max_x:.3f}")
    print(f"mesh Y range: {min_y:.3f} ~ {max_y:.3f}")
    return pts, tris, elevs, boundary, tri_types

