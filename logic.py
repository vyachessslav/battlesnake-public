from typing import Dict, List, Set, Tuple, Optional
import math
from heapq import heappush, heappop

Point = Tuple[int, int]
DIRECTIONS: Dict[str, Point] = {
    "up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)
}

# Параметры (тонко настроены под 11x11)
STARVING_THRESHOLD = 40
CRITICAL_HUNGER = 22
ENEMY_REPULSION = 18
FOOD_WEIGHT = 5
SPACE_BONUS = 2.2
HEAD_TO_HEAD_PENALTY = 250_000
LENGTH_ADVANTAGE = 35
WALL_PENALTY = 12

def get_info() -> Dict[str, str]:
    return {
        "apiversion": "1",
        "author": "pro-cautious",
        "color": "#1e8449",
        "head": "safe",
        "tail": "bolt",
        "version": "3.0",
    }

def choose_move(game_state: Dict) -> str:
    return choose_move_pro(game_state)

def choose_move_pro(game_state: Dict) -> str:
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

    # 1. Repulsion от врагов
    if enemy_heads:
        min_dist = min(_manhattan(nxt, eh) for eh in enemy_heads)
        score += min_dist ** 1.6 * ENEMY_REPULSION
        repulsion = sum(1.0 / max(1, d) for d in (_manhattan(nxt, eh) for eh in enemy_heads))
        score += repulsion * ENEMY_REPULSION * 3.5

    # 2. A* + Food logic
    if foods:
        food_score = _food_a_star_score(nxt, foods, occupied, snakes, my_id, w, h, health)
        score += food_score

    # 3. Advanced space
    space = _advanced_flood_fill(nxt, occupied, snakes, my_id, w, h, my_len * 2)
    score += space * SPACE_BONUS

    # 4. Head-to-head & length advantage
    if _is_head_to_head_danger(nxt, snakes, my_len):
        enemy_max_len = _max_threat_length(nxt, snakes)
        if enemy_max_len >= my_len:
            score -= HEAD_TO_HEAD_PENALTY
        else:
            score += LENGTH_ADVANTAGE * (my_len - enemy_max_len)

    # 5. Wall penalty
    if _is_near_wall(nxt, w, h, margin=2):
        score -= WALL_PENALTY

    return score


def _food_a_star_score(nxt, foods, occupied, snakes, my_id, w, h, health):
    """Используем A* для оценки лучшего направления к еде."""
    if health > STARVING_THRESHOLD + 20:
        return -sum(1.0 / max(1, _manhattan(nxt, f)) for f in foods) * 3  # избегать

    best_food_score = -999
    for food in foods:
        path = _a_star(nxt, food, occupied, snakes, my_id, w, h)
        if path is None:
            continue
        dist = len(path)
        if dist == 0:
            continue
        value = (30 - dist) * (STARVING_THRESHOLD - health + 10) / dist
        best_food_score = max(best_food_score, value)

    return best_food_score * FOOD_WEIGHT


def _a_star(start: Point, goal: Point, occupied: Set[Point], snakes, my_id, w, h) -> Optional[List[Point]]:
    """Простой A* с учётом будущих хвостов."""
    open_set = []
    heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: _manhattan(start, goal)}

    while open_set:
        _, current = heappop(open_set)
        if current == goal:
            # reconstruct path (не обязательно полностью)
            return _reconstruct_path(came_from, current)

        for d in DIRECTIONS.values():
            neighbor = (current[0] + d[0], current[1] + d[1])
            if not _in_bounds(neighbor, w, h):
                continue
            if neighbor in occupied and not _will_tail_free(neighbor, snakes, my_id):
                continue

            tentative_g = g_score[current] + 1
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


# === Вспомогательные функции (улучшенные) ===
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
        if len(snake.get("body", [])) < 3:
            continue
        tail = (snake["body"][-1]["x"], snake["body"][-1]["y"])
        if pos == tail:
            return True
    return False


def _is_head_to_head_danger(pos, snakes, my_len):
    for snake in snakes:
        if snake["length"] < my_len - 1:
            continue
        ehead = (snake["head"]["x"], snake["head"]["y"])
        if _manhattan(pos, ehead) <= 1:
            return True
    return False


def _max_threat_length(pos, snakes):
    max_l = 0
    for s in snakes:
        if _manhattan(pos, (s["head"]["x"], s["head"]["y"])) <= 2:
            max_l = max(max_l, s["length"])
    return max_l


def _is_near_wall(p: Point, w: int, h: int, margin: int = 2) -> bool:
    return (p[0] < margin or p[0] >= w - margin or
            p[1] < margin or p[1] >= h - margin)


def _occupied_cells(snakes):
    occ = set()
    for s in snakes:
        for b in s["body"]:
            occ.add((b["x"], b["y"]))
    return occ


def _get_enemy_heads(snakes, my_id):
    return [(s["head"]["x"], s["head"]["y"]) for s in snakes if s["id"] != my_id]


def _in_bounds(p: Point, w: int, h: int) -> bool:
    return 0 <= p[0] < w and 0 <= p[1] < h


def _manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
