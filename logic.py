"""Heuristic move-selection logic for the Battlesnake.

This is an "aggressive-chonker" policy that actively hunts smaller snakes,
eats constantly to grow massive, and dominates the center of the board.

Board coordinates: ``(0, 0)`` is the bottom-left corner.
  up    -> y + 1
  down  -> y - 1
  left  -> x - 1
  right -> x + 1

Game-state schema reference: https://docs.battlesnake.com/api
"""

from typing import Dict, List, Set, Tuple, Optional
import math

Point = Tuple[int, int]

DIRECTIONS: Dict[str, Point] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}

# Aggression settings.
HUNT_LENGTH_ADVANTAGE = 2  # Hunt snakes at least 2 segments smaller.
# Always eat, but even more aggressive when hungry.
HUNGER_THRESHOLD = 50  # Start getting anxious about food.
STARVING_THRESHOLD = 25  # Desperate for food.
# Scoring weights.
KILL_BONUS = 10000  # Massive bonus for potential kills.
FOOD_WEIGHT = 8  # High food priority.
CENTER_WEIGHT = 3  # Stay near center.
SPACE_WEIGHT = 2  # Avoid corners.
AGGRESSION_WEIGHT = 5  # Chase smaller snakes.
HEAD_TO_HEAD_WIN_BONUS = 5000  # Bonus for winning head-to-head.
SURVIVAL_PENALTY = 100000  # Avoid losing head-to-head.
# How far we look for prey.
HUNT_RADIUS = 8


def get_info() -> Dict[str, str]:
    """Appearance + metadata returned from ``GET /``."""
    return {
        "apiversion": "1",
        "author": "aggressive-chonker",
        "color": "#ff4444",
        "head": "evil",
        "tail": "sharp",
        "version": "1.0.0",
    }


def choose_move(game_state: Dict) -> str:
    """Return the next move using the aggressive-chonker heuristic."""
    return choose_move_aggressive(game_state)


def choose_move_aggressive(game_state: Dict) -> str:
    """Return the next move for the current turn.

    Strategy:
    1. Hunt and kill snakes that are at least 2 segments smaller.
    2. Win head-to-head collisions against smaller snakes.
    3. Eat food aggressively to maintain/grow length advantage.
    4. Control the center of the board.
    5. Avoid being cornered.
    6. Avoid head-to-head with equal or larger snakes.
    """
    board = game_state["board"]
    you = game_state["you"]
    width: int = board["width"]
    height: int = board["height"]

    head: Point = (you["head"]["x"], you["head"]["y"])
    my_length: int = you["length"]
    health: int = you["health"]

    # Calculate occupied cells considering moving tails.
    occupied = _occupied_cells_accounting_tails(board["snakes"])
    enemy_heads = _get_enemy_heads(board["snakes"], you["id"])
    
    # Separate enemies by size.
    prey, threats, equals = _categorize_enemies(board["snakes"], you["id"], my_length, HUNT_LENGTH_ADVANTAGE)
    
    # Cells where we would lose head-to-head.
    losing_head_to_head = _get_losing_head_to_head_cells(board["snakes"], you["id"], my_length)
    
    # Cells where we would WIN head-to-head.
    winning_head_to_head = _get_winning_head_to_head_cells(board["snakes"], you["id"], my_length, HUNT_LENGTH_ADVANTAGE)
    
    foods = [(f["x"], f["y"]) for f in board["food"]]
    
    # Calculate center of the board.
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

        # CRITICAL: Never move into a losing head-to-head.
        if nxt in losing_head_to_head:
            continue  # Suicide prevention.

        # HUGE bonus for winning head-to-head collisions.
        if nxt in winning_head_to_head:
            score += HEAD_TO_HEAD_WIN_BONUS
            # Extra bonus if this kills a snake we're hunting.
            target = _get_snake_at_head(board["snakes"], nxt, you["id"])
            if target and target["length"] <= my_length - HUNT_LENGTH_ADVANTAGE:
                score += KILL_BONUS

        # Hunt smaller snakes aggressively.
        if prey:
            # Find nearest prey.
            nearest_prey_dist = float("inf")
            nearest_prey_head = None
            for p in prey:
                dist = _manhattan(nxt, p)
                if dist < nearest_prey_dist:
                    nearest_prey_dist = dist
                    nearest_prey_head = p
            
            # Chase prey - more bonus the closer we are.
            if nearest_prey_head:
                # Predict where prey might move and try to cut them off.
                prey_moves = _get_possible_moves(nearest_prey_head, occupied, width, height)
                if prey_moves:
                    # Score based on how many of prey's escape routes we block.
                    blocked_routes = 0
                    for prey_move_pos in prey_moves:
                        if _manhattan(nxt, prey_move_pos) <= 1:
                            blocked_routes += 1
                    score += blocked_routes * KILL_BONUS // 2
                
                # Basic chase bonus.
                score += max(0, HUNT_RADIUS - nearest_prey_dist) * AGGRESSION_WEIGHT * 10

        # Avoid threats (equal or larger snakes) - keep distance.
        if threats:
            for threat_head in threats:
                dist = _manhattan(nxt, threat_head)
                if dist < 3:
                    score -= SURVIVAL_PENALTY // (dist + 1)
        
        # Also avoid equals unless we have no choice.
        if equals:
            for equal_head in equals:
                dist = _manhattan(nxt, equal_head)
                if dist < 2:
                    score -= SURVIVAL_PENALTY // (dist + 1)

        # Food is always good - helps maintain size advantage.
        if foods:
            nearest_food_dist = min(_manhattan(nxt, f) for f in foods)
            hunger_multiplier = 1.0
            if health < STARVING_THRESHOLD:
                hunger_multiplier = 5.0
            elif health < HUNGER_THRESHOLD:
                hunger_multiplier = 2.0
            score += (1.0 / max(1, nearest_food_dist)) * FOOD_WEIGHT * hunger_multiplier

        # Stay near center for board control.
        dist_to_center = math.sqrt((nxt[0] - center[0])**2 + (nxt[1] - center[1])**2)
        center_bonus = (1.0 / max(0.1, dist_to_center)) * CENTER_WEIGHT
        score += center_bonus

        # Check reachable space to avoid being cornered.
        space = _flood_fill(nxt, occupied, width, height, limit=my_length * 3)
        score += space * SPACE_WEIGHT
        
        # Avoid moves that trap us.
        if space < my_length:
            score -= 500

        if score > best_score:
            best_score = score
            best_move = move

    # No safe move found -> fallback.
    return best_move or "up"


def _categorize_enemies(
    snakes: List[Dict], my_id: str, my_length: int, advantage: int
) -> Tuple[List[Point], List[Point], List[Point]]:
    """Split enemy heads into prey, threats, and equals.
    
    Returns:
        prey: Snakes we can hunt (at least 'advantage' segments smaller).
        threats: Snakes larger than us (dangerous).
        equals: Snakes within 1 segment of our length.
    """
    prey = []
    threats = []
    equals = []
    
    for snake in snakes:
        if snake["id"] == my_id:
            continue
        head = (snake["head"]["x"], snake["head"]["y"])
        length = snake["length"]
        
        if length <= my_length - advantage:
            prey.append(head)
        elif length > my_length:
            threats.append(head)
        else:
            equals.append(head)
    
    return prey, threats, equals


def _get_losing_head_to_head_cells(
    snakes: List[Dict], my_id: str, my_length: int
) -> Set[Point]:
    """Cells where we would lose or tie a head-to-head collision.
    
    We lose/tie against snakes of equal or greater length.
    """
    danger: Set[Point] = set()
    for snake in snakes:
        if snake["id"] == my_id:
            continue
        # We lose or tie against equal or larger snakes.
        if snake["length"] >= my_length:
            ehead = (snake["head"]["x"], snake["head"]["y"])
            for dx, dy in DIRECTIONS.values():
                danger.add((ehead[0] + dx, ehead[1] + dy))
    return danger


def _get_winning_head_to_head_cells(
    snakes: List[Dict], my_id: str, my_length: int, advantage: int
) -> Set[Point]:
    """Cells where we would WIN a head-to-head collision.
    
    We win against snakes that are at least 'advantage' segments smaller.
    """
    winning: Set[Point] = set()
    for snake in snakes:
        if snake["id"] == my_id:
            continue
        # We win against snakes at least 'advantage' smaller.
        if snake["length"] <= my_length - advantage:
            ehead = (snake["head"]["x"], snake["head"]["y"])
            for dx, dy in DIRECTIONS.values():
                winning.add((ehead[0] + dx, ehead[1] + dy))
    return winning


def _get_snake_at_head(snakes: List[Dict], point: Point, my_id: str) -> Optional[Dict]:
    """Find a snake whose head is at the given point."""
    for snake in snakes:
        if snake["id"] != my_id:
            head = (snake["head"]["x"], snake["head"]["y"])
            if head == point:
                return snake
    return None


def _get_possible_moves(
    head: Point, occupied: Set[Point], width: int, height: int
) -> List[Point]:
    """Get possible moves for a snake head (excluding its current position)."""
    moves = []
    for dx, dy in DIRECTIONS.values():
        nxt = (head[0] + dx, head[1] + dy)
        if _in_bounds(nxt, width, height) and nxt not in occupied:
            moves.append(nxt)
    return moves


def _occupied_cells_accounting_tails(snakes: List[Dict]) -> Set[Point]:
    """Calculate occupied cells, considering that tails will move next turn.
    
    A snake's tail cell will become free on the next turn UNLESS the snake
    has just eaten.
    """
    occupied: Set[Point] = set()
    
    for snake in snakes:
        body = snake["body"]
        
        # Check if snake just ate (tail won't move).
        tail_will_move = True
        if len(body) >= 2:
            last = (body[-1]["x"], body[-1]["y"])
            second_last = (body[-2]["x"], body[-2]["y"])
            if last == second_last:
                tail_will_move = False
        
        # Add all body segments.
        for i, seg in enumerate(body):
            # Skip the tail if it will move.
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

