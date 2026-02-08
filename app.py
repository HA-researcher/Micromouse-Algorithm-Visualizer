import streamlit as st
import numpy as np
from collections import deque
import time

# --- 定数定義 ---
EMPTY = 0
WALL = 1
START = 2
GOAL = 3
PATH = 4
VISITED = 5

COLORS = {
    EMPTY: "#FFFFFF",   # 白: 通路
    WALL: "#333333",    # 黒: 壁
    START: "#00FF00",   # 緑: スタート
    GOAL: "#FF0000",    # 赤: ゴール
    PATH: "#FFFF00",    # 黄: 最短経路
    VISITED: "#CCEOFF"  # 薄青: 探索済みエリア
}

# --- 関数定義 ---

def init_maze(height: int, width: int, wall_prob: float = 0.2) -> np.ndarray:
    """
    迷路（グリッド）を初期化します。
    計算量: O(HW) - グリッド全体のセルを生成するため
    """
    # 全体を空きマスで初期化
    maze = np.zeros((height, width), dtype=int)
    
    # 外周を壁にする
    maze[0, :] = maze[-1, :] = WALL
    maze[:, 0] = maze[:, -1] = WALL
    
    # 内部にランダムな壁を生成（スタート・ゴール予定地付近は空けておくなどの処理は簡易化）
    inner_walls = np.random.choice([EMPTY, WALL], size=(height-2, width-2), p=[1-wall_prob, wall_prob])
    maze[1:-1, 1:-1] = inner_walls
    
    # スタートとゴールを設定 (左上と右下)
    maze[1, 1] = START
    maze[height-2, width-2] = GOAL
    
    return maze

def bfs_solver(maze: np.ndarray, start_pos: tuple, goal_pos: tuple):
    """
    幅優先探索(BFS)を用いて最短経路を探索します。
    
    Args:
        maze: グリッドの状態
        start_pos: (y, x)
        goal_pos: (y, x)
        
    Returns:
        path: 最短経路の座標リスト
        visited_history: 探索した順序のリスト（可視化用）
        
    計算量 (Time Complexity): O(H*W)
      - H: グリッドの高さ, W: グリッドの幅
    """
    h, w = maze.shape
    queue = deque([start_pos])
    
    # 距離と経路復元用の親ノードを記録するテーブル
    # dist = -1 で未訪問を表現
    dist = np.full((h, w), -1)
    prev = np.full((h, w), None) # 経路復元用: どこから来たか
    
    dist[start_pos] = 0
    visited_history = []
    
    # 方向ベクトル (上、下、左、右)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    found = False
    
    while queue:
        cy, cx = queue.popleft()
        visited_history.append((cy, cx))
        
        if (cy, cx) == goal_pos:
            found = True
            break
            
        for dy, dx in directions:
            ny, nx = cy + dy, cx + dx
            
            # 範囲外チェック（外壁があるので不要だが安全のため）
            if not (0 <= ny < h and 0 <= nx < w):
                continue
                
            # 壁チェック または 訪問済みチェック
            if maze[ny, nx] == WALL or dist[ny, nx] != -1:
                continue
                
            # 更新
            dist[ny, nx] = dist[cy, cx] + 1
            prev[ny, nx] = (cy, cx) # 親を記録
            queue.append((ny, nx))
            
    # 経路復元 (ゴールからスタートへ遡る)
    path = []
    if found:
        cur = goal_pos
        while cur is not None:
            path.append(cur)
            try:
                # np.fullの初期値Noneから取り出す際、タプルまたはNoneになる
                cur = prev[cur[0], cur[1]]
                if isinstance(cur, np.ndarray): # numpy arrayの要素として保存された場合の対策
                    cur = tuple(cur)
            except:
                cur = None
        path.reverse()
    
    return path, visited_history

def render_grid_html(maze, path_set, visited_set, current_step_visited=None):
    """
    HTML/CSSを用いてグリッドを描画します。
    Streamlitの標準テーブルより視認性を高めるための工夫です。
    """
    h, w = maze.shape
    html = '<div style="display: flex; flex-direction: column; align-items: center;">'
    
    cell_size = 25 # ピクセル
    
    for y in range(h):
        html += '<div style="display: flex;">'
        for x in range(w):
            cell_type = maze[y, x]
            color = COLORS[cell_type]
            
            # オーバーレイ（経路や探索済みの描画）
            if (y, x) in path_set:
                color = COLORS[PATH]
            elif (y, x) in visited_set:
                color = COLORS[VISITED]
            
            # スタート・ゴールは常に上書き表示
            if cell_type == START: color = COLORS[START]
            if cell_type == GOAL: color = COLORS[GOAL]

            html += f'<div style="width:{cell_size}px; height:{cell_size}px; background-color:{color}; border: 1px solid #ddd;"></div>'
        html += '</div>'
    html += '</div>'
    return html

# --- Main Application ---

def main():
    st.set_page_config(page_title="MicroMouse Visualizer", layout="wide")
    
    st.title("MicroMouse Algorithm Visualizer")
    st.markdown("""
    ### 概要
    マイクロマウス（自律走行ロボット）の経路探索アルゴリズム「幅優先探索 (BFS)」の挙動を可視化するツールです。
    サークル内の新入生技術講習およびアルゴリズム検証用に開発しました。
    
    ### 技術仕様
    - Language: Python 3
    - Algorithm: Breadth-First Search (BFS)
    - Time Complexity: $O(H \\times W)$ ... $H, W$ はグリッドの高さと幅
    - Visualization: Streamlit
    """)

    # --- Sidebar Controls ---
    st.sidebar.header("Configuration")
    grid_size = st.sidebar.slider("Grid Size (N x N)", 5, 30, 15)
    wall_prob = st.sidebar.slider("Wall Density", 0.0, 0.5, 0.2)
    
    # リセットボタン管理
    if 'maze' not in st.session_state or st.sidebar.button("Generate New Maze"):
        st.session_state.maze = init_maze(grid_size, grid_size, wall_prob)
        st.session_state.solved = False
        st.session_state.path = []
        st.session_state.visited_history = []

    # --- Main Area ---
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Maze Preview")
        
        # プレースホルダーを作成（ここに動的に描画）
        maze_placeholder = st.empty()
        
        # 初期状態の描画
        current_maze = st.session_state.maze
        start_pos = (1, 1)
        goal_pos = (grid_size-2, grid_size-2)
        
        # まだ解いていない場合、初期状態を表示
        if not st.session_state.solved:
            maze_placeholder.markdown(render_grid_html(current_maze, set(), set()), unsafe_allow_html=True)

    with col2:
        st.subheader("Control & Stats")
        if st.button("Start Simulation (BFS)"):
            with st.spinner("Solving..."):
                path, visited = bfs_solver(current_maze, start_pos, goal_pos)
                st.session_state.path = path
                st.session_state.visited_history = visited
                st.session_state.solved = True
            
            st.success("Solved!")
            st.metric("Shortest Path Length", f"{len(path)} steps")
            st.metric("Total Visited Nodes", f"{len(visited)} cells")

    # アニメーション実行 (解き終わった直後、または状態保持されている場合)
    if st.session_state.solved:
        # 簡易アニメーション: 探索済みを一気に描画するか、少しずつ描画するか
        # Streamlitでのアニメーションは再描画コストがかかるため、今回は結果表示を優先
        
        with col1:
            maze_placeholder.markdown(
                render_grid_html(current_maze, set(st.session_state.path), set(st.session_state.visited_history)), 
                unsafe_allow_html=True
            )
            
        with col2:
            st.info("""
            アルゴリズム解説 (BFS):
            1. スタート地点から等距離にある全ノードを探索キューに入れます。
            2. 薄い青色のエリアは、探索済み（キューから取り出された）マスです。
            3. 黄色のラインは、ゴール到達後に親ノードを遡って復元した「最短経路」です。
            """)

if __name__ == "__main__":
    main()
