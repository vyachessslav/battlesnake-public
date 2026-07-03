"""Heuristic move-selection logic for the Battlesnake.

This is a "scaredy-snake" policy that prioritizes staying far from enemies
and only eats when very hungry to maintain minimal length.
When well-fed, it gravitates toward the center of the board.

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
SAFE_DISTANCE = 5  # Increased from 3
# Only eat when health drops below this threshold.
STARVING_THRESHOLD = 40  # Changed from 30
# How much we value distance from enemies vs food.
ENEMY_DISTANCE_WEIGHT = 15  # Increased from 10
FOOD_DISTANCE_WEIGHT = 2
# Penalty applied to a move that could lose a head-to-head collision.
HEAD_TO_HEAD_PENALTY = 500_000  # Increased from 100_000
# Bonus for open space (to avoid being cornered).
SPACE_BONUS = 3  # Increased from 1
# Weight for center-gravity when well-fed.
CENTER_GRAVITY_WEIGHT = 5  # Increased from 3
# Extra penalty for being near walls when well-fed.
WALL_PENALTY = 50
# Minimum safe distance to maintain from any enemy body.
MIN_BODY_DISTANCE = 2
# Penalty for being too close to enemy bodies.
BODY_PROXIMITY_PENALTY = 200


def get_info() -> Dict[str, str]:
    """Appearance + metadata returned from ``GET /``."""
    return {
        "apiversion": "1",
        "author": "scaredy-snake",
        "color": "#4a90d9",
        "head": "silly",
        "tail": "bolt",
        "version": "3.0.0",
    }


def choose_move(game_state: Dict) -> str:
    """Return the next move using the scaredy-snake heuristic."""
    return choose_move_scaredy(game_state)


def choose_move_scaredy(game_state: Dict) -> str:
    """Return the next move for the current turn.

    Strategy:
    1. Stay as far as possible from enemy heads AND bodies.
    2. Avoid head-to-head collisions with equal/larger snakes.
    3. Only go for food when health is below STARVING_THRESHOLD.
    4. When well-fed, gravitate toward the center of the board.
    5. Prefer moves that lead to more open space.
    6. Account for moving tails (tails become free next turn).
    7. Actively avoid walls and corners when possible.
    """
    board = game_state["board"]
    you = game_state["you"]
    width: int = board["width"]
    height: int = board["height"]

    head: Point = (you["head"]["x"], you["head"]["y"])
    my_length: int = you["length"]
    health: int = you["health"]

    # Calculate occupied cells considering moving tails
    occupied = _occupied_cells_accounting_tails(board["snakes"])
    enemy_heads = _get_enemy_heads(board["snakes"], you["id"])
    enemy_bodies = _get_enemy_bodies(board["snakes"], you["id"])
    danger = _head_to_head_cells(board["snakes"], you["id"], my_length)
    foods = [(f["x"], f["y"]) for f in board["food"]]

    # Calculate center of the board
    center = (width / 2.0, height / 2.0)

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

        # CRITICAL: Heavy penalty for dangerous head-to-head positions.
        if nxt in danger:
            score -= HEAD_TO_HEAD_PENALTY
            # If it's a guaranteed loss, just skip this move entirely
            if _is_losing_head_to_head(nxt, board["snakes"], you["id"], my_length):
                continue

        # Primary goal: maximize distance to ALL enemies (heads AND bodies).
        all_enemy_cells = enemy_heads + enemy_bodies
        
        if all_enemy_cells:
            # Minimum distance to nearest enemy (head or body).
            min_enemy_dist = min(_manhattan(nxt, ec) for ec in all_enemy_cells)
            
            # Heavy penalty for being too close to enemies.
            if min_enemy_dist < MIN_BODY_DISTANCE:
                score -= BODY_PROXIMITY_PENALTY * (MIN_BODY_DISTANCE - min_enemy_dist + 1)
            
            # Base score from distance to enemies.
            score += min_enemy_dist * ENEMY_DISTANCE_WEIGHT
            
            # Extra bonus for being far from all enemies (use sum of inverse distances).
            total_enemy_repulsion = sum(1.0 / max(1, _manhattan(nxt, ec)) for ec in all_enemy_cells)
            score += total_enemy_repulsion * ENEMY_DISTANCE_WEIGHT * 2

        # Secondary goal: only chase food when starving.
        if foods and health < STARVING_THRESHOLD:
            # Find the nearest food to this move position.
            nearest_food_dist = min(_manhattan(nxt, f) for f in foods)
            # Bonus inversely proportional to distance (closer = better).
            # More aggressive food seeking when health is very low.
            hunger_factor = (STARVING_THRESHOLD - health) / STARVING_THRESHOLD
            score += (1.0 / max(1, nearest_food_dist)) * FOOD_DISTANCE_WEIGHT * hunger_factor * 10
        elif foods and health >= STARVING_THRESHOLD:
            # When not hungry, actively avoid food to keep length small.
            nearest_food_dist = min(_manhattan(nxt, f) for f in foods)
            # Stronger penalty for being near food when well-fed.
            score -= (1.0 / max(1, nearest_food_dist)) * 5

        # When well-fed, strongly gravitate toward the center of the board.
        if health >= STARVING_THRESHOLD:
            # Calculate distance to center (Euclidean for smooth gradient).
            dist_to_center = math.sqrt((nxt[0] - center[0])**2 + (nxt[1] - center[1])**2)
            # Bonus inversely proportional to distance from center.
            center_bonus = (1.0 / max(0.1, dist_to_center)) * CENTER_GRAVITY_WEIGHT
            score += center_bonus
            
            # Penalty for being near walls.
            dist_to_wall = min(
                nxt[0],  # distance to left wall
                width - 1 - nxt[0],  # distance to right wall
                nxt[1],  # distance to bottom wall
                height - 1 - nxt[1]  # distance to top wall
            )
            if dist_to_wall <= 1:
                score -= WALL_PENALTY * (2 - dist_to_wall)

        # Check reachable space to avoid being cornered.
        # More thorough flood fill with higher limit.
        space = _flood_fill(nxt, occupied, width, height, limit=my_length * 4)
        score += space * SPACE_BONUS

        # Extra safety: avoid moves that lead to dead ends.
        if space < my_length:
            score -= 1000 * (my_length - space)

        if score > best_score:
            best_score = score
            best_move = move

    # No safe move found -> move up and hope for the best.
    return best_move or "up"


def _is_losing_head_to_head(
    nxt: Point, snakes: List[Dict], my_id: str, my_length: int
) -> bool:
    """Check if moving to nxt would result in a guaranteed loss in head-to-head."""
    for snake in snakes:
        if snake["id"] == my_id:
            continue
        if snake["length"] >= my_length:
            ehead = (snake["head"]["x"], snake["head"]["y"])
            # Check if enemy could also move to the same cell
            for dx, dy in DIRECTIONS.values():
                enemy_move = (ehead[0] + dx, ehead[1] + dy)
                if enemy_move == nxt:
                    return True
    return False


def _occupied_cells_accounting_tails(snakes: List[Dict]) -> Set[Point]:
    """Calculate occupied cells, considering that tails will move next turn.
    
    A snake's tail cell will become free on the next turn UNLESS the snake
    has just eaten (indicated by the body segments). We check if the snake
    is growing by comparing the last two body segments - if they're the same,
    the snake just ate and the tail won't move.
    """
    occupied: Set[Point] = set()
    
    for snake in snakes:
        body = snake["body"]
        
        # Check if snake just ate (tail won't move)
        # If last two segments are identical, snake is growing
        tail_will_move = True
        if len(body) >= 2:
            last = (body[-1]["x"], body[-1]["y"])
            second_last = (body[-2]["x"], body[-2]["y"])
            if last == second_last:
                tail_will_move = False
        
        # Add all body segments
        for i, seg in enumerate(body):
            # Skip the tail if it will move (become free next turn)
            if i == len(body) - 1 and tail_will_move:
                continue
            occupied.add((seg["x"], seg["y"]))
    
    return occupied


def _get_enemy_heads(snakes: List[Dict], my_id: str) -> List[Point]:
    """Get positions of all enemy snake heads."""
    heads = []
    for snake in snakes:
        if snake["id"] != my_id:
            heads.append((snake["head"]["x"], snake["head"]["y"]))
    return heads


def _get_enemy_bodies(snakes: List[Dict], my_id: str) -> List[Point]:
    """Get all body segments of enemy snakes (excluding heads and tails that will move)."""
    bodies = []
    for snake in snakes:
        if snake["id"] == my_id:
            continue
        
        body = snake["body"]
        
        # Determine if tail will move
        tail_will_move = True
        if len(body) >= 2:
            last = (body[-1]["x"], body[-1]["y"])
            second_last = (body[-2]["x"], body[-2]["y"])
            if last == second_last:
                tail_will_move = False
        
        # Add all body segments except head and possibly tail
        for i, seg in enumerate(body):
            # Skip head (already tracked separately)
            if i == 0:
                continue
            # Skip tail if it will move
            if i == len(body) - 1 and tail_will_move:
                continue
            bodies.append((seg["x"], seg["y"]))
    
    return bodies


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
