
def score(f, w):
    mult = (w.get('mult_enemy', 1.8) if f.is_enemy
            else (w.get('mult_neutral', 1.0) if f.is_neutral
                  else w.get('mult_reinforce_ally', 0.0)))
    if f.is_comet:
        mult *= w.get('mult_comet', 1.5)
    denom = (w.get('w_ships_cost', 0.02) * max(1.0, f.ships_to_send) +
             w.get('w_travel_cost', 0.6) * f.travel_turns + 1e-6)
    return mult * (f.target_production ** 2) / denom
