import heapq
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

Point = Tuple[int, int]

DIRECTIONS: Dict[str, Point] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}


def get_info() -> Dict[str, str]:
    """Метаданные для GET /"""
    return {
        "apiversion": "1",
        "author": "adaptive_centrist",
        "color": "#6434eb",
        "head": "smart-caterpillar",
        "tail": "weight",
        "version": "2.0.0",
    }


def choose_move(game_state: Dict) -> str:
    """Основная точка входа – всегда использует эвристику."""
    return choose_move_heuristic(game_state)


# ---------------------------------------------------------------------
# A* и вспомогательные функции для поиска пути к еде
# ---------------------------------------------------------------------

def a_star(start: Point,
           goal: Point,
           width: int,
           height: int,
           obstacles: Set[Point]) -> Optional[List[Point]]:
    """
    Классический A* от start до goal на сетке.
    obstacles – запрещённые клетки.
    Возвращает путь (включая начальную и конечную точку) или None.
    """
    def heuristic(a: Point) -> int:
        return abs(a[0] - goal[0]) + abs(a[1] - goal[1])

    open_set = [(0, start)]
    came_from: Dict[Point, Point] = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            # Восстановление пути
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        x, y = current
        for dx, dy in DIRECTIONS.values():
            neighbor = (x + dx, y + dy)
            if not (0 <= neighbor[0] < width and 0 <= neighbor[1] < height):
                continue
            if neighbor in obstacles:
                continue

            tentative_g = g_score[current] + 1
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor)
                heapq.heappush(open_set, (f, neighbor))

    return None


def direction_from_path(path: List[Point]) -> str:
    """Преобразует первые две точки пути в строку направления."""
    if len(path) < 2:
        return "up"
    hx, hy = path[0]
    nx, ny = path[1]
    dx, dy = nx - hx, ny - hy
    for move, (mdx, mdy) in DIRECTIONS.items():
        if (dx, dy) == (mdx, mdy):
            return move
    return "up"


# ---------------------------------------------------------------------
# Основная эвристика
# ---------------------------------------------------------------------

def choose_move_heuristic(game_state: Dict) -> str:
    board = game_state["board"]
    you = game_state["you"]
    width, height = board["width"], board["height"]
    head = (you["head"]["x"], you["head"]["y"])
    length = you["length"]
    health = you["health"]  # не используется, но оставлено для возможных модификаций

    # Все занятые клетки (тела всех змей)
    occupied = _occupied_cells(board["snakes"])

    # Свой хвост считаем проходимым (он освободится на следующем ходу)
    my_tail = (you["body"][-1]["x"], you["body"][-1]["y"])
    if my_tail in occupied and length > 1:
        occupied.remove(my_tail)

    # --- Если змея короткая (длина ≤ 3): ищем путь к еде через A* ---
    if length <= 3:
        foods = [(f["x"], f["y"]) for f in board["food"]]
        if foods:
            # Сортируем по манхэттенскому расстоянию и пробуем A* к каждой
            foods.sort(key=lambda f: abs(f[0] - head[0]) + abs(f[1] - head[1]))
            for target in foods:
                path = a_star(head, target, width, height, occupied)
                if path and len(path) >= 2:
                    # Дополнительно проверяем, что после хода не загоним себя в тупик
                    next_cell = path[1]
                    # Временно добавляем next_cell в препятствия, чтобы flood fill считал область после хода
                    space = _flood_fill(next_cell, occupied | {next_cell}, width, height, limit=width * height)
                    if space >= length + 2:  # достаточно места для манёвра
                        return direction_from_path(path)
        # Если еды нет или ни одна не подходит, делаем безопасный ход к центру (чтобы не застрять)
        return _safe_move_towards_center(head, occupied, width, height)

    # --- Если змея длинная (длина > 3): уклоняемся в центре ---
    center = (width // 2, height // 2)

    legal_moves = []
    for move, (dx, dy) in DIRECTIONS.items():
        nx, ny = head[0] + dx, head[1] + dy
        if _in_bounds((nx, ny), width, height) and (nx, ny) not in occupied:
            legal_moves.append((move, (nx, ny)))

    if not legal_moves:
        return "up"

    best_move = None
    best_score = float("-inf")

    for move, nxt in legal_moves:
        # Размер доступной области после хода
        space = _flood_fill(nxt, occupied, width, height, limit=width * height)
        if space < length + 2:
            space_score = -1000
        else:
            space_score = space

        # Близость к центру: чем меньше расстояние, тем выше score
        dist_to_center = _manhattan(nxt, center)
        center_score = (width + height - dist_to_center) * 2  # вес умеренный

        score = space_score + center_score
        if score > best_score:
            best_score = score
            best_move = move

    return best_move or "up"


def _safe_move_towards_center(head: Point, obstacles: Set[Point],
                              width: int, height: int) -> str:
    """
    Возвращает безопасный ход, который максимально приближает к центру.
    Используется, если короткая змея не может найти путь к еде.
    """
    center = (width // 2, height // 2)
    best_move = None
    best_dist = float("inf")

    for move, (dx, dy) in DIRECTIONS.items():
        nx, ny = head[0] + dx, head[1] + dy
        if _in_bounds((nx, ny), width, height) and (nx, ny) not in obstacles:
            dist = _manhattan((nx, ny), center)
            if dist < best_dist:
                best_dist = dist
                best_move = move

    return best_move or "up"


# ---------------------------------------------------------------------
# Утилитарные функции
# ---------------------------------------------------------------------

def _occupied_cells(snakes: List[Dict]) -> Set[Point]:
    """Все клетки, занятые телами змей."""
    occupied: Set[Point] = set()
    for snake in snakes:
        for seg in snake["body"]:
            occupied.add((seg["x"], seg["y"]))
    return occupied


def _in_bounds(p: Point, width: int, height: int) -> bool:
    return 0 <= p[0] < width and 0 <= p[1] < height


def _manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _flood_fill(start: Point, occupied: Set[Point],
                width: int, height: int, limit: int) -> int:
    """
    Количество свободных клеток, достижимых из start
    (не более limit, для производительности).
    """
    seen: Set[Point] = {start}
    stack: List[Point] = [start]
    count = 0

    while stack:
        x, y = stack.pop()
        count += 1
        if count >= limit:
            break
        for dx, dy in DIRECTIONS.values():
            nb = (x + dx, y + dy)
            if nb in seen:
                continue
            if not _in_bounds(nb, width, height):
                continue
            if nb in occupied:
                continue
            seen.add(nb)
            stack.append(nb)

    return count
