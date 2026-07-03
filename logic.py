from typing import Dict, List, Set, Tuple, Optional
import math
from heapq import heappush, heappop

Point = Tuple[int, int]
DIRECTIONS: Dict[str, Point] = {
    "up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)
}

# === Параметры (обновлены) ===
STARVING_THRESHOLD = 40
CRITICAL_HUNGER = 22
ENEMY_REPULSION = 18
LONGER_ENEMY_REPULSION = 35      # Новый сильный коэффициент
FOOD_WEIGHT = 5
SPACE_BONUS = 2.3
HEAD_TO_HEAD_PENALTY = 600_000   # Ещё сильнее
WALL_PENALTY = 12
LENGTH_ADVANTAGE = 45

def get_info() -> Dict[str, str]:
    return {
        "apiversion": "1",
        "author": "pro-cautious",
        "color": "#1e8449",
        "head": "safe",
        "tail": "bolt",
        "version": "3.2",
    }

def choose_move(game_state: Dict) -> str:
    board = game_state["board"]
    you = game_state["you"]
    w, h = board["width"], board["height"]
    head: Point = (you["head"]["x"], you["head"]["y"])
    my_len = you["length"]
    health = you["health"]
    my_id = you["id"]

    occupied = _occupied_cells(board["snakes"])
    enemy_heads = _get_enemy_heads(board["snakes"], my_id)
    foods = [(f["x"], f["y"]) for f in board["food"]]

    best_dir = "up"
    best_score = float("-inf")

    for dir_name, delta in DIRECTIONS.items():
        nxt: Point = (head[0] + delta[0], head[1] + delta[1])
        if not _in_bounds(nxt, w, h) or nxt in occupied:
            continue

        score = _evaluate_position(
            nxt, head, my_len, health, occupied, enemy_heads, foods,
            board["snakes"], my_id, w, h
        )

        if score > best_score:
            best_score = score
            best_dir = dir_name

    return best_dir


def _evaluate_position(nxt, head, my_len, health, occupied, enemy_heads, foods, snakes, my_id, w, h):
    score = 0.0

    # === УСИЛЕННЫЙ FEAR ОТ ДЛИННЫХ ВРАГОВ ===
    longer_near = False
    for eh in enemy_heads:
        dist = _manhattan(nxt, eh)
        min_dist_to_any = min((_manhattan(nxt, e) for e in enemy_heads), default=999)
        
        # Общий repulsion
        score += (min_dist_to_any ** 1.6) * ENEMY_REPULSION
        
        # Специальный страх от более длинных
        enemy = next((s for s in snakes if (s["head"]["x"], s["head"]["y"]) == eh), None)
        if enemy and enemy["length"] > my_len:
            score += (1.0 / max(1, dist)) * LONGER_ENEMY_REPULSION * 4.5
            if dist <= 4:
                longer_near = True

    # Head-to-head (особенно опасно против длинных)
    if _is_dangerous_head_collision(nxt, snakes, my_len):
        score -= HEAD_TO_HEAD_PENALTY
        # Дополнительный штраф, если это длинный враг
        if longer_near:
            score -= HEAD_TO_HEAD_PENALTY * 0.6

    # A* к еде
    if foods:
        score += _food_a_star_score(nxt, foods, occupied, snakes, my_id, w, h, health)

    # Пространство
    space = _advanced_flood_fill(nxt, occupied, snakes, my_id, w, h, my_len * 2)
    score += space * SPACE_BONUS

    # Стены
    if _is_near_wall(nxt, w, h, margin=2):
        score -= WALL_PENALTY

    return score


def _is_dangerous_head_collision(pos: Point, snakes: List[Dict], my_len: int) -> bool:
    for snake in snakes:
        e_len = snake["length"]
        if e_len < my_len - 2:   # сильно короче — терпимо
            continue
        ehead = (snake["head"]["x"], snake["head"]["y"])
        for d in DIRECTIONS.values():
            if (ehead[0] + d[0], ehead[1] + d[1]) == pos:
                return True
    return False


# === Остальные функции остаются прежними (A*, flood fill и т.д.) ===
def _food_a_star_score(nxt, foods, occupied, snakes, my_id, w, h, health):
    if health > STARVING_THRESHOLD + 20:
        return -sum(1.0 / max(1, _manhattan(nxt, f)) for f in foods) * 4

    best = -999
    for food in foods:
        path = _a_star(nxt, food, occupied, snakes, my_id, w, h)
        if path and len(path) > 0:
            dist = len(path)
            value = max(0, 45 - dist) * max(1, STARVING_THRESHOLD - health + 20) / max(1, dist)
            best = max(best, value)
    return best * FOOD_WEIGHT


def _a_star(start: Point, goal: Point, occupied: Set[Point], snakes, my_id, w, h) -> Optional[List[Point]]:
    open_set = []
    heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: _manhattan(start, goal)}

    while open_set:
        _, current = heappop(open_set)
        if current == goal:
            return _reconstruct_path(came_from, current)

        for d in DIRECTIONS.values():
            neighbor = (current[0] + d[0], current[1] + d[1])
            if not _in_bounds(neighbor, w, h):
                continue
            if neighbor in occupied and not _will_tail_free(neighbor, snakes, my_id):
                continue

            tentative_g = g_score.get(current, 0) + 1
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + _manhattan(neighbor, goal)
                heappush(open_set, (f_score[neighbor], neighbor))
    return None


def _reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]


def _advanced_flood_fill(start, occupied, snakes, my_id, w, h, limit):
    seen = {start}
    stack = [start]
    count = 0
    freed = 0
    while stack and count < limit:
        pos = stack.pop()
        count += 1
        for d in DIRECTIONS.values():
            nbr = (pos[0] + d[0], pos[1] + d[1])
            if nbr in seen or not _in_bounds(nbr, w, h):
                continue
            if nbr in occupied:
                if _will_tail_free(nbr, snakes, my_id):
                    freed += 1
                    seen.add(nbr)
                    stack.append(nbr)
                continue
            seen.add(nbr)
            stack.append(nbr)
    return count + freed * 7


def _will_tail_free(pos, snakes, my_id):
    for snake in snakes:
        body = snake.get("body", [])
        if len(body) < 3: continue
        tail = (body[-1]["x"], body[-1]["y"])
        if pos == tail:
            return True
    return False


def _occupied_cells(snakes):
    occ = set()
    for s in snakes:
        for b in s.get("body", []):
            occ.add((b["x"], b["y"]))
    return occ


def _get_enemy_heads(snakes, my_id):
    return [(s["head"]["x"], s["head"]["y"]) for s in snakes if s["id"] != my_id]


def _in_bounds(p: Point, w: int, h: int) -> bool:
    return 0 <= p[0] < w and 0 <= p[1] < h


def _manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _is_near_wall(p: Point, w: int, h: int, margin: int = 2) -> bool:
    return p[0] < margin or p[0] >= w - margin or p[1] < margin or p[1] >= h - margin
