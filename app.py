import streamlit as st
import numpy as np
from collections import deque
import heapq
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

# --- 迷路生成 ---
def init_maze(size: int, wall_prob: float = 0.2) -> np.ndarray:
    maze = np.zeros((size, size), dtype=int)
    maze[0, :] = maze[-1, :] = WALL
    maze[:, 0] = maze[:, -1] = WALL
    inner_area = (slice(1, -1), slice(1, -1))
    maze[inner_area] = np.random.choice([EMPTY, WALL], size=(size-2, size-2), p=[1-wall_prob, wall_prob])
    maze[1, 1] = START
    maze[size-2, size-2] = GOAL
    maze[1, 2] = maze[2, 1] = EMPTY
    maze[size-2, size-3] = maze[size-3, size-2] = EMPTY
    return maze

# --- アルゴリズム実装 ---

def solve_bfs(maze, start, goal):
    """幅優先探索 (Start -> Goal)"""
    h, w = maze.shape
    queue = deque([start])
    visited = set([start])
    parent = {start: None}
    visited_history = []

    while queue:
        curr = queue.popleft()
        visited_history.append(curr)
        
        if curr == goal:
            break
        
        for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
            ny, nx = curr[0]+dy, curr[1]+dx
            if 0 <= ny < h and 0 <= nx < w and maze[ny, nx] != WALL and (ny, nx) not in visited:
                visited.add((ny, nx))
                parent[(ny, nx)] = curr
                queue.append((ny, nx))
                
    return reconstruct_path(parent, goal), visited_history, None

def solve_dfs(maze, start, goal):
    """深さ優先探索"""
    h, w = maze.shape
    stack = [start]
    visited = set([start])
    parent = {start: None}
    visited_history = []

    while stack:
        curr = stack.pop()
        visited_history.append(curr)
        
        if curr == goal:
            break
        
        # 探索順序を調整 (上右下左)
        for dy, dx in [(-1,0), (0,1), (1,0), (0,-1)]:
            ny, nx = curr[0]+dy, curr[1]+dx
            if 0 <= ny < h and 0 <= nx < w and maze[ny, nx] != WALL and (ny, nx) not in visited:
                visited.add((ny, nx))
                parent[(ny, nx)] = curr
                stack.append((ny, nx))

    return reconstruct_path(parent, goal), visited_history, None

def solve_astar(maze, start, goal):
    """A*探索"""
    h, w = maze.shape
    pq = [(0, start)]
    g_score = {start: 0}
    parent = {start: None}
    visited_history = []
    visited_set = set()

    while pq:
        _, curr = heapq.heappop(pq)
        
        if curr in visited_set: continue
        visited_set.add(curr)
        visited_history.append(curr)
        
        if curr == goal:
            break
            
        for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
            ny, nx = curr[0]+dy, curr[1]+dx
            if 0 <= ny < h and 0 <= nx < w and maze[ny, nx] != WALL:
                new_g = g_score[curr] + 1
                if new_g < g_score.get((ny, nx), float('inf')):
                    g_score[(ny, nx)] = new_g
                    f_score = new_g + (abs(goal[0]-ny) + abs(goal[1]-nx))
                    heapq.heappush(pq, (f_score, (ny, nx)))
                    parent[(ny, nx)] = curr

    return reconstruct_path(parent, goal), visited_history, None

def solve_adachi(maze, start, goal):
    """
    足立法 (Adachi's Method)
    ゴールからの歩数マップ(Step Map)を作成し、数字が小さくなる方へ進む。
    """
    h, w = maze.shape
    # 1. ゴールからの距離マップを作成 (BFS from Goal)
    queue = deque([goal])
    dist_map = np.full((h, w), -1)
    dist_map[goal] = 0
    visited_history = [] # マップ作成の過程

    while queue:
        cy, cx = queue.popleft()
        visited_history.append((cy, cx))
        
        for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
            ny, nx = cy+dy, cx+dx
            if 0 <= ny < h and 0 <= nx < w and maze[ny, nx] != WALL and dist_map[ny, nx] == -1:
                dist_map[ny, nx] = dist_map[cy, cx] + 1
                queue.append((ny, nx))
    
    # 2. スタートからゴールへ、数字が小さくなる方へ進む（経路復元）
    path = []
    curr = start
    if dist_map[start] != -1: #到達可能なら
        path.append(curr)
        while curr != goal:
            cy, cx = curr
            current_dist = dist_map[cy, cx]
            moved = False
            for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                ny, nx = cy+dy, cx+dx
                if 0 <= ny < h and 0 <= nx < w:
                    # 隣が「今の距離 - 1」ならそこが進むべき道
                    if dist_map[ny, nx] == current_dist - 1:
                        curr = (ny, nx)
                        path.append(curr)
                        moved = True
                        break
            if not moved: break

    return path, visited_history, dist_map

def reconstruct_path(parent, goal):
    path = []
    curr = goal
    while curr is not None:
        path.append(curr)
        curr = parent.get(curr)
    return path[::-1] if len(path) > 1 else []

# --- 描画関数 ---
def render_grid_html(maze, path_set, visited_set, dist_map=None, show_numbers=False):
    h, w = maze.shape
    html = '<div style="display: flex; flex-direction: column; align-items: center; margin-bottom: 20px; font-family: monospace;">'
    
    # マスサイズ調整: 数字表示ありなら少し大きく
    cell_size = 24 if show_numbers else 20
    font_size = 10 if show_numbers else 0
    
    for y in range(h):
        html += '<div style="display: flex;">'
        for x in range(w):
            cell_type = maze[y, x]
            color = COLORS[cell_type]
            
            # 色の優先順位: 経路 > 探索済み > デフォルト
            if (y, x) in path_set: color = COLORS[PATH]
            elif (y, x) in visited_set: color = COLORS[VISITED]
            
            if cell_type == START: color = COLORS[START]
            if cell_type == GOAL: color = COLORS[GOAL]
            
            # 数字表示 (足立法用)
            text = ""
            if show_numbers and dist_map is not None:
                d = dist_map[y, x]
                if d != -1 and cell_type != WALL:
                    text = str(d)

        # スタイルを1行の文字列として定義
            style = (f"width:{cell_size}px; height:{cell_size}px; "
                     f"background-color:{color}; border: 1px solid #ddd; "
                     "display: flex; align-items: center; justify-content: center; "
                     f"font-size: {font_size}px; color: #000;")
            
            # HTMLタグを隙間なく連結
            html += f'<div style="{style}">{text}</div>'
            
        html += '</div>'
    html += '</div>'
    return html

# --- Main Application ---
def main():
    st.set_page_config(page_title="MicroMouse Visualizer", layout="wide")

    tab_sim, tab_info = st.tabs(["🧩 アルゴリズム実験室", "📢 サークル紹介"])

    # 共通サイドバー
    with st.sidebar:
        st.header("設定 (Settings)")
        grid_size = st.slider("迷路サイズ", 10, 40, 20)
        wall_prob = st.slider("壁の密度", 0.0, 0.4, 0.25)
        
        st.subheader("アルゴリズム選択")
        algo_type = st.radio(
            "Mode",
            ["足立法 (Adachi's Method)", "BFS (幅優先探索)", "DFS (深さ優先探索)", "A* (エースター探索)"]
        )
        
        # 足立法選択時のみ表示するオプション
        show_numbers_opt = False
        if "足立法" in algo_type:
            show_numbers_opt = st.checkbox("歩数マップを表示 (Show Steps)", value=True)
        
        if st.button("迷路再生成 / Reset"):
            st.session_state.maze = init_maze(grid_size, wall_prob)
            st.session_state.solved = False
            if 'dist_map' in st.session_state: del st.session_state.dist_map

    if 'maze' not in st.session_state:
        st.session_state.maze = init_maze(grid_size, wall_prob)
        st.session_state.solved = False

    # --- Tab 1: シミュレータ ---
    # --- Tab 1: シミュレータ ---
    with tab_sim:
        col1, col2 = st.columns([2, 1])
        h, w = st.session_state.maze.shape
        start, goal = (1, 1), (h-2, w-2)

        # ★変更点1: 描画エリア(placeholder)を先に作っておく
        with col1:
            st.subheader("Visualizer")
            grid_placeholder = st.empty()

        with col2:
            st.subheader("実行パネル")
            
            # (中略: アルゴリズム説明のif文などはそのまま)
            if "BFS" in algo_type:
                st.info("**BFS (幅優先探索)**\n\nStartから全方位にしらみつぶしに探します。最短経路を保証します。")
            elif "DFS" in algo_type:
                st.warning("**DFS (深さ優先探索)**\n\n行けるところまで突っ走ります。最短経路は保証されません。")
            elif "A*" in algo_type:
                st.success("**A* (エースター探索)**\n\n「ゴールへの推定距離」を使って賢く探索します。計算コストが低いです。")
            else:
                st.error("**足立法 (Adachi's Method)**\n\nGoalからStartに向かって「歩数マップ」を作ります。マウスは数字が小さい方へ進みます。")

            # ★変更点2: ボタンを押した時の処理
            if st.button("探索開始 (Run)"):
                start_time = time.time()
                dist_map_result = None
                
                # 1. まず計算する
                if "足立法" in algo_type:
                    path, visited, dist_map_result = solve_adachi(st.session_state.maze, start, goal)
                elif "BFS" in algo_type:
                    path, visited, _ = solve_bfs(st.session_state.maze, start, goal)
                elif "DFS" in algo_type:
                    path, visited, _ = solve_dfs(st.session_state.maze, start, goal)
                else: # A*
                    path, visited, _ = solve_astar(st.session_state.maze, start, goal)
                
                elapsed = (time.time() - start_time) * 1000
                
                # 2. アニメーション実行 (探索の過程を描画)
                visited_so_far = set()
                # 足立法の時はdist_mapを表示したいので最初から渡す
                current_dist_map = dist_map_result if "足立法" in algo_type else None

                # visitedリストを順番になぞって描画更新
                for v_cell in visited:
                    visited_so_far.add(v_cell)
                    # 少し処理を間引く(毎回描画すると遅すぎる場合)
                    # if len(visited_so_far) % 2 == 0: 
                    html = render_grid_html(st.session_state.maze, set(), visited_so_far, current_dist_map, show_numbers_opt)
                    grid_placeholder.markdown(html, unsafe_allow_html=True)
                    time.sleep(0.02) # ★ここでスピード調整 (0.01~0.05くらい)

                # 3. 最後に「最短経路(黄色)」を重ねて完了表示
                st.session_state.path = path
                st.session_state.visited = visited
                st.session_state.dist_map = dist_map_result
                st.session_state.solved = True
                st.session_state.stats = (len(path), len(visited), elapsed)

            # 結果表示パネル
            if st.session_state.solved:
                p_len, v_count, t_ms = st.session_state.stats
                st.metric("最短経路ステップ数", f"{p_len} steps")
                st.metric("探索したマスの数", f"{v_count} cells")
                st.metric("計算時間", f"{t_ms:.2f} ms")

        # ★変更点3: ボタンを押していない時(初期状態や再描画時)の表示
        # solvedなら結果を、そうでなければ初期状態を表示
        path_set = set(st.session_state.path) if st.session_state.solved else set()
        visited_set = set(st.session_state.visited) if st.session_state.solved else set()
        d_map = st.session_state.get('dist_map', None)
        
        # アニメーション以外のタイミングで表示を維持するため
        grid_placeholder.markdown(
            render_grid_html(st.session_state.maze, path_set, visited_set, d_map, show_numbers_opt), 
            unsafe_allow_html=True
        )
    # --- Tab 2: サークル紹介 ---
    with tab_info:
        st.title("マイクロマウスサークルへようこそ！")
        
        # カラム分けは残すが、c2（右側）の中身は空にする
        c1, c2 = st.columns(2)
        with c1:
            st.image("https://placehold.co/600x400/222/FFF?text=MicroMouse+Robot+Image", caption="自作マウス機体例")
            st.markdown("""
            ### マイクロマウスとは？
            16×16マスの迷路を自律走行ロボットが走り、ゴールまでのタイムを競う競技です。
            **「ハードウェア設計」 × 「ソフトウェア制御」** の両方が学べる、エンジニアへの近道です！
            """)
        
        # c2にはサークルの予定などを追記予定

if __name__ == "__main__":
    main()
