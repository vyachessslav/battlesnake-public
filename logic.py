"""Heuristic move-selection logic for the Battlesnake.

This is a "scaredy-snake" policy that prioritizes staying far from enemies
and only eats when very hungry to maintain minimal length.

Board coordinates: ``(0, 0)`` is the bottom-left corner.
  up    -> y + 1
  down  -> y - 1
  left  -> x - 1
  right -> x + 1

Game-state schema reference: https://docs.battlesnake.com/api
"""

from typing import Dict, List, Set, Tuple
import math

Point = Tuple[int, int]

DIRECTIONS: Dict[str, Point] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}

# How far we try to stay from enemy heads (in cells).
SAFE_DISTANCE = 3
# Only eat when health drops below this threshold.
STARVING_THRESHOLD = 30
# How much we value distance from enemies vs food.
ENEMY_DISTANCE_WEIGHT = 10
FOOD_DISTANCE_WEIGHT = 2
# Penalty applied to a move that could lose a head-to-head collision.
HEAD_TO_HEAD_PENALTY = 100_000
# Bonus for open space (to avoid being cornered).
SPACE_BONUS = 1


def get_info() -> Dict[str, str]:
    """Appearance + metadata returned from ``GET /``."""
    return {
        "apiversion": "1",
        "author": "scaredy-snake",
        "color": "#4a90d9",
        "head": "silly",
        "tail": "bolt",
        "version": "1.0.0",
    }


def choose_move(game_state: Dict) -> str:
    """Return the next move using the scaredy-snake heuristic."""
    return choose_move_scaredy(game_state)


def choose_move_scaredy(game_state: Dict) -> str:
    """Return the next move for the current turn.

    Strategy:
    1. Stay as far as possible from enemy heads.
    2. Avoid head-to-head collisions with equal/larger snakes.
    3. Only go for food when health is below STARVING_THRESHOLD.
    4. Prefer moves that lead to more open space.
    """
    board = game_state["board"]
    you = game_state["you"]
    width: int = board["width"]
    height: int = board["height"]

    head: Point = (you["head"]["x"], you["head"]["y"])
    my_length: int = you["length"]
    health: int = you["health"]

    occupied = _occupied_cells(board["snakes"])
    enemy_heads = _get_enemy_heads(board["snakes"], you["id"])
    danger = _head_to_head_cells(board["snakes"], you["id"], my_length)
    foods = [(f["x"], f["y"]) for f in board["food"]]

    best_move = None
    best_score = float("-inf")

    for move, (dx, dy) in DIRECTIONS.items():
        nxt = (head[0] + dx, head[1] + dy)

        # Don't move out of bounds or into occupied cells.
        if not _in_bounds(nxt, width, height):
            continue
        if nxt in occupied:
            continue

        score = 0.0

        # Primary goal: maximize minimum distance to enemy heads.
        if enemy_heads:
            min_enemy_dist = min(_manhattan(nxt, eh) for eh in enemy_heads)
            score += min_enemy_dist * ENEMY_DISTANCE_WEIGHT
            
            # Extra bonus for being far from all enemies (use sum of inverse distances).
            total_enemy_repulsion = sum(1.0 / max(1, _manhattan(nxt, eh)) for eh in enemy_heads)
            score += total_enemy_repulsion * ENEMY_DISTANCE_WEIGHT * 2

        # Secondary goal: only chase food when starving.
        if foods and health < STARVING_THRESHOLD:
            # Find the nearest food to this move position.
            nearest_food_dist = min(_manhattan(nxt, f) for f in foods)
            # Bonus inversely proportional to distance (closer = better).
            score += (1.0 / max(1, nearest_food_dist)) * FOOD_DISTANCE_WEIGHT * (STARVING_THRESHOLD - health)
        elif foods and health >= STARVING_THRESHOLD:
            # When not hungry, actively avoid food to keep length small.
            nearest_food_dist = min(_manhattan(nxt, f) for f in foods)
            # Slight penalty for being near food (don't eat accidentally).
            score -= (1.0 / max(1, nearest_food_dist)) * 2

        # Check reachable space to avoid being cornered.
        space = _flood_fill(nxt, occupied, width, height, limit=my_length * 2)
        score += space * SPACE_BONUS

        # Heavy penalty for dangerous head-to-head positions.
        if nxt in danger:
            score -= HEAD_TO_HEAD_PENALTY

        if score > best_score:
            best_score = score
            best_move = move

    # No safe move found -> move up and hope for the best.
    return best_move or "up"


def _occupied_cells(snakes: List[Dict]) -> Set[Point]:
    """All cells currently filled by any snake's body."""
    occupied: Set[Point] = set()
    for snake in snakes:
        for seg in snake["body"]:
            occupied.add((seg["x"], seg["y"]))
    return occupied


def _get_enemy_heads(snakes: List[Dict], my_id: str) -> List[Point]:
    """Get positions of all enemy snake heads."""
    heads = []
    for snake in snakes:
        if snake["id"] != my_id:
            heads.append((snake["head"]["x"], snake["head"]["y"]))
    return heads


def _head_to_head_cells(snakes: List[Dict], my_id: str, my_length: int) -> Set[Point]:
    """Cells adjacent to enemy heads that could result in losing head-to-head.

    A cell is dangerous if an enemy of equal or greater length could move there
    next turn. We heavily penalize these positions because our scaredy-snake
    wants to avoid any confrontation.
    """
    danger: Set[Point] = set()
    for snake in snakes:
        if snake["id"] == my_id:
            continue
        if snake["length"] < my_length:
            continue  # We can win against shorter snakes
        ehead = (snake["head"]["x"], snake["head"]["y"])
        for dx, dy in DIRECTIONS.values():
            danger.add((ehead[0] + dx, ehead[1] + dy))
    return danger


def _flood_fill(
    start: Point, occupied: Set[Point], width: int, height: int, limit: int
) -> int:
    """Count open cells reachable from ``start`` (capped at ``limit``)."""
    seen: Set[Point] = {start}
    stack: List[Point] = [start]
    count = 0
    while stack:
        x, y = stack.pop()
        count += 1
        if count >= limit:
            break
        for dx, dy in DIRECTIONS.values():
            nbr = (x + dx, y + dy)
            if nbr in seen:
                continue
            if not _in_bounds(nbr, width, height):
                continue
            if nbr in occupied:
                continue
            seen.add(nbr)
            stack.append(nbr)
    return count


def _in_bounds(p: Point, width: int, height: int) -> bool:
    return 0 <= p[0] < width and 0 <= p[1] < height


def _manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])