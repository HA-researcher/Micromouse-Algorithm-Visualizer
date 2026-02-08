import streamlit as st
import numpy as np
from collections import deque
import time

# --- 定数定義 ---
EMPTY = 0
WALL = 1
START = 2
GOAL = 3
PATH = 4    # 最短経路
VISITED = 5 # 探索済み

# 色定義 (HTMLカラーコード)
COLORS = {
    EMPTY: "#FFFFFF",   # 白: 通路
    WALL: "#333333",    # 黒: 壁
    START: "#00FF00",   # 緑: スタート
    GOAL: "#FF0000",    # 赤: ゴール
    PATH: "#FFFF00",    # 黄: 最短経路
    VISITED: "#CCEOFF"  # 薄青: 探索済み
}

def init_maze(size: int, wall_prob: float = 0.2) -> np.ndarray:
    """
    迷路を初期化します。
    計算量: O(N^2)
    """
    maze = np.zeros((size, size), dtype=int)
    
    # 外周壁
    maze[0, :] = maze[-1, :] = WALL
    maze[:, 0] = maze[:, -1] = WALL
    
    # 内部壁
    inner_area = (slice(1, -1), slice(1, -1))
    maze[inner_area] = np.random.choice(
        [EMPTY, WALL], 
        size=(size-2, size-2), 
        p=[1-wall_prob, wall_prob]
    )
    
    # スタート(1,1)とゴール(size-2, size-2)の設定
    maze[1, 1] = START
    maze[size-2, size-2] = GOAL
    
    # スタート・ゴール周辺の壁削除（詰み防止）
    maze[1, 2] = maze[2, 1] = EMPTY
    maze[size-2, size-3] = maze[size-3, size-2] = EMPTY
    
    return maze

def calculate_distance_map(maze: np.ndarray, start_node: tuple):
    """
    指定された開始ノードからの距離マップ（歩数マップ）を作成します。
    BFSアルゴリズムを使用。
    
    Args:
        maze: 迷路配列
        start_node: 探索開始座標 (y, x)
    Returns:
        dist_map: 各座標の距離（到達不能は-1）
        visited_order: 探索順序リスト
    """
    h, w = maze.shape
    queue = deque([start_node])
    dist_map = np.full((h, w), -1)
    visited_order = []
    
    dist_map[start_node] = 0
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # 上下左右
    
    while queue:
        cy, cx = queue.popleft()
        visited_order.append((cy, cx))
        
        current_dist = dist_map[cy, cx]
        
        for dy, dx in directions:
            ny, nx = cy + dy, cx + dx
            
            # 範囲外、壁、訪問済みチェック
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            if maze[ny, nx] == WALL or dist_map[ny, nx] != -1:
                continue
            
            dist_map[ny, nx] = current_dist + 1
            queue.append((ny, nx))
            
    return dist_map, visited_order

def trace_path_from_map(dist_map: np.ndarray, start_node: tuple, target_node: tuple):
    """
    距離マップを使って、start_nodeからtarget_nodeへの最短経路を復元します。
    (足立法のように、数字が小さい方へ進むイメージ)
    """
    h, w = dist_map.shape
    path = []
    
    # そもそも到達不能なら空
    if dist_map[start_node] == -1 or dist_map[target_node] == -1:
        return []

    # 経路復元ロジック
    # ゴール(target)からスタート(start)へ向かって、距離が減る方へ進む
    # (BFSならこれで最短経路になる)
    curr = target_node
    path.append(curr)
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while curr != start_node:
        cy, cx = curr
        current_val = dist_map[cy, cx]
        found_next = False
        
        for dy, dx in directions:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w:
                # 距離がちょうど1小さい隣接セルを探す
                if dist_map[ny, nx] == current_val - 1:
                    curr = (ny, nx)
                    path.append(curr)
                    found_next = True
                    break
        
        if not found_next:
            break # ここに来ることは理論上ないはず
            
    return path # 逆順にする必要はない（Goal -> Startで復元した場合）

def render_grid_html(maze, path_set, visited_set, dist_map=None, show_numbers=False):
    """
    HTML描画関数。数字表示オプションを追加。
    """
    h, w = maze.shape
    html = '<div style="display: flex; flex-direction: column; align-items: center; font-family: monospace;">'
    
    # マスのサイズ調整 (数字が入るなら少し大きく)
    cell_size = 25 if show_numbers else 20
    font_size = 10 if show_numbers else 0
    
    for y in range(h):
        html += '<div style="display: flex;">'
        for x in range(w):
            cell_type = maze[y, x]
            color = COLORS[cell_type]
            
            # 色の優先順位
            if (y, x) in path_set:
                color = COLORS[PATH]
            elif (y, x) in visited_set:
                color = COLORS[VISITED]
            
            if cell_type == START: color = COLORS[START]
            if cell_type == GOAL: color = COLORS[GOAL]

            # 数字の表示内容
            text = ""
            if show_numbers and dist_map is not None:
                d = dist_map[y, x]
                if d != -1 and cell_type != WALL:
                    text = str(d)

            # HTML構築
            html += f'''
                <div style="
                    width:{cell_size}px; 
                    height:{cell_size}px; 
                    background-color:{color}; 
                    border: 1px solid #ddd; 
                    display: flex;
                    align-items: center; 
                    justify_content: center;
                    font-size: {font_size}px;
                    color: #000;
                    ">
                    {text}
                </div>'''
        html += '</div>'
    html += '</div>'
    return html

def main():
    st.set_page_config(page_title="MicroMouse Visualizer", layout="wide")
    
    st.title("MicroMouse Algorithm Visualizer")
    st.write("マイクロマウス活動：探索アルゴリズム比較デモ")

    # --- Sidebar ---
    with st.sidebar:
        st.header("設定")
        grid_size = st.slider("迷路サイズ (Grid Size)", 10, 25, 15)
        wall_prob = st.slider("壁の密度 (Wall Density)", 0.0, 0.5, 0.25)
        
        # アルゴリズム選択
        algo_mode = st.radio(
            "アルゴリズム選択",
            ("BFS (Start → Goal)", "足立法 (Goal → Start Map)")
        )
        
        show_numbers = st.checkbox("歩数(距離)を表示する", value=True)
        
        if st.button("迷路を再生成 (Reset)"):
            st.session_state.maze = init_maze(grid_size, wall_prob)
            st.session_state.solved = False
            if 'dist_map' in st.session_state: del st.session_state.dist_map

    # --- Session Init ---
    if 'maze' not in st.session_state:
        st.session_state.maze = init_maze(grid_size, wall_prob)
        st.session_state.solved = False

    col1, col2 = st.columns([2, 1])
    
    h, w = st.session_state.maze.shape
    start_pos = (1, 1)
    goal_pos = (h-2, w-2)

    with col2:
        st.subheader("Algorithm Info")
        
        if algo_mode == "BFS (Start → Goal)":
            st.info("""
            **幅優先探索 (BFS)**
            スタート地点から全方位に探索を広げ、ゴールを見つけます。
            - 始点: Start (緑)
            - 探索順: Startから近い順
            """)
        else:
            st.success("""
            **足立法 (Adachi's Method)**
            ゴールからの距離（歩数）マップを作成します。
            マウスは「今のマスより数字が小さいマス」へ進むことで最短経路を辿れます。
            - 始点: Goal (赤)
            - Step Map: ゴールまでの距離
            """)

        if st.button("実行 (Run)"):
            with st.spinner("Calculating..."):
                start_time = time.time()
                
                if algo_mode == "BFS (Start → Goal)":
                    # BFS: スタートから探索
                    dist_map, visited = calculate_distance_map(st.session_state.maze, start_pos)
                    # 経路はゴールから戻る
                    path = trace_path_from_map(dist_map, goal_pos, start_pos) # 引数順注意: start->targetへの経路を復元する場合、逆から辿るロジックならこうなる
                    
                else: # 足立法
                    # 足立法: ゴールから歩数マップを作る
                    dist_map, visited = calculate_distance_map(st.session_state.maze, goal_pos)
                    # 経路はスタートから「数字が減る方」へ進む（実質同じロジックで復元可能）
                    path = trace_path_from_map(dist_map, start_pos, goal_pos)

                end_time = time.time()
                
                st.session_state.dist_map = dist_map
                st.session_state.path = path
                st.session_state.visited = visited
                st.session_state.solved = True
                
            st.write(f"計算時間: {(end_time - start_time)*1000:.2f} ms")

    with col1:
        st.subheader("Visualizer")
        
        path_set = set()
        visited_set = set()
        dist_map = None
        
        if st.session_state.solved:
            path_set = set(st.session_state.path)
            visited_set = set(st.session_state.visited)
            dist_map = st.session_state.dist_map
        
        # 描画
        maze_html = render_grid_html(
            st.session_state.maze, 
            path_set, 
            visited_set, 
            dist_map, 
            show_numbers
        )
        st.markdown(maze_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
