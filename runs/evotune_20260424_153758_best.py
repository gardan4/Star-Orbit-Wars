
# Hand-written baseline: mimics HeuristicAgent._score_target (simplified).
def score(f, w):
    mult = 1.0
    if f.is_enemy: mult = w.get('mult_enemy', 1.8)
    elif f.is_neutral: mult = w.get('mult_neutral', 1.0)
    elif f.is_ally: mult = w.get('mult_reinforce_ally', 0.0)
    if f.is_comet: mult *= w.get('mult_comet', 1.5)
    denom = (w.get('w_ships_cost', 0.02) * max(1.0, f.ships_to_send) +
             w.get('w_travel_cost', 0.3) * f.travel_turns +
             w.get('w_distance_cost', 0.05) * f.distance +
             1e-6)
    return mult * w.get('w_production', 5.0) * f.target_production / denom
