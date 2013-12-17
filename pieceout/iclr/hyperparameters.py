def get_hyperparameters(rng):
    BATCH_SIZE = [5, 10, 20, 25][rng.random_integers(0, 3)]
    DECAY_FACTOR = 1 + 10 ** rng.random_integers(-6, -3)
    LEARNING_RATE = 10 ** rng.uniform(-3, -0.7)
    INIT_MOMENTUM = rng.uniform(0.5, 0.7)
    FINAL_MOMENTUM = INIT_MOMENTUM + rng.uniform(0.0, 0.25)
    MOMENTUM_SATURATE = 25 * rng.random_integers(1,  8)
    return locals()
