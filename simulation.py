import taichi as ti

ti.init(arch=ti.gpu)

# Simulation parameters
WIDTH, HEIGHT = 1024, 768
MAX_CREATURES = 500
MAX_FOOD = 200
INITIAL_CREATURES = 25
INITIAL_FOOD = 100
FOOD_SPAWN_RATE = 0.2
MUTATION_RATE = 0.2
MUTATION_STRENGTH = 0.2


# Creature properties
@ti.dataclass
class Creature:
    pos: ti.math.vec2
    vel: ti.math.vec2
    energy: ti.f32
    age: ti.f32
    alive: ti.i32
    # Genetic traits
    speed: ti.f32
    size: ti.f32
    vision_range: ti.f32
    efficiency: ti.f32
    color: ti.math.vec3
    generation: ti.i32
    # Wandering behavior
    wander_dir: ti.f32  # Angle that creature directs itself towards when wandering
    wander_steps: ti.i32  # Steps remaining in current wander direction


# Food properties
@ti.dataclass
class Food:
    pos: ti.math.vec2
    energy: ti.f32
    active: ti.i32


# Fields
creatures = Creature.field(shape=MAX_CREATURES)
food = Food.field(shape=MAX_FOOD)
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

total_alive = ti.field(dtype=ti.i32, shape=())
max_generation = ti.field(dtype=ti.i32, shape=())


@ti.kernel
def init_simulation():
    # Initialize creatures
    for i in range(INITIAL_CREATURES):
        set_creature(i, alive=1)

    # Initialize food
    for i in range(INITIAL_FOOD):
        food[i].pos = ti.math.vec2(
            ti.random() * WIDTH,
            ti.random() * HEIGHT
        )
        food[i].energy = 10.0 + ti.random() * 20.0
        food[i].active = 1


@ti.func
def set_creature(i: int, alive: ti.i32):
    creatures[i].pos = ti.math.vec2(
        ti.random() * WIDTH,
        ti.random() * HEIGHT
    )
    creatures[i].vel = ti.math.vec2(0, 0)
    creatures[i].energy = 50.0
    creatures[i].age = 0.0
    creatures[i].alive = alive
    creatures[i].speed = 1.0 + ti.random() * 0.4
    creatures[i].size = 3.0 + ti.random() * 4.0
    creatures[i].vision_range = 35.0 + ti.random() * 40.0
    creatures[i].efficiency = 0.8 + ti.random() * 0.4
    creatures[i].color = ti.math.vec3(
        0.2 + ti.random() * 0.8,
        0.2 + ti.random() * 0.8,
        0.2 + ti.random() * 0.8
    )
    creatures[i].generation = 0
    creatures[i].wander_dir = ti.random() * 2.0 * 3.14159
    creatures[i].wander_steps = 0


@ti.kernel
def reset_simulation():
    # Clear and reset all creatures
    for i in range(MAX_CREATURES):
        set_creature(i, alive=0)

    # Clear and reset all food
    for i in range(MAX_FOOD):
        food[i].active = 0

    # Reset stats
    total_alive[None] = 0
    max_generation[None] = 0


@ti.kernel
def update_creatures(dt: ti.f32):
    total_alive[None] = 0
    sum_speed = 0.0
    sum_size = 0.0
    sum_vision = 0.0
    max_gen = 0

    for i in range(MAX_CREATURES):
        if creatures[i].alive == 1:
            total_alive[None] += 1
            sum_speed += creatures[i].speed
            sum_size += creatures[i].size
            sum_vision += creatures[i].vision_range
            max_gen = ti.max(max_gen, creatures[i].generation)

            # Age and energy consumption
            creatures[i].age += dt
            energy_cost = dt * (0.5 + creatures[i].speed * 0.8 +
                                creatures[i].size * 0.8 +
                                creatures[i].vision_range * 0.01) / creatures[i].efficiency
            creatures[i].energy -= energy_cost

            # Find nearest food
            nearest_food = -1
            min_dist = creatures[i].vision_range

            for j in range(MAX_FOOD):
                if food[j].active == 1:
                    dist = (creatures[i].pos - food[j].pos).norm()
                    if dist < min_dist:
                        min_dist = dist
                        nearest_food = j

            # Move towards food or wander
            if nearest_food >= 0:
                # Food found - move towards it
                direction = (food[nearest_food].pos - creatures[i].pos).normalized()
                creatures[i].vel = direction * creatures[i].speed * 50.0
                creatures[i].wander_steps = 0  # Reset wander when food is found
            else:
                # No food in sight - wander
                if creatures[i].wander_steps <= 0:
                    # Pick new random direction
                    creatures[i].wander_dir = ti.random() * 2.0 * 3.14159
                    creatures[i].wander_steps = ti.cast(10 + ti.random() * 20, ti.i32)  # Wander for 10-30 steps

                # Continue in wander direction with small random variation
                angle_variation = (ti.random() - 0.5) * 0.2  # Add slight randomness
                current_angle = creatures[i].wander_dir + angle_variation
                creatures[i].vel = ti.math.vec2(
                    ti.cos(current_angle) * creatures[i].speed * 30.0,
                    ti.sin(current_angle) * creatures[i].speed * 30.0
                )
                creatures[i].wander_steps -= 1

            # Update position
            creatures[i].pos += creatures[i].vel * dt

            # Wrap around edges
            if creatures[i].pos.x < 0:
                creatures[i].pos.x = WIDTH
            elif creatures[i].pos.x > WIDTH:
                creatures[i].pos.x = 0
            if creatures[i].pos.y < 0:
                creatures[i].pos.y = HEIGHT
            elif creatures[i].pos.y > HEIGHT:
                creatures[i].pos.y = 0

            # Eat food
            for j in range(MAX_FOOD):
                if food[j].active == 1:
                    dist = (creatures[i].pos - food[j].pos).norm()
                    if dist < creatures[i].size + 5:
                        creatures[i].energy += food[j].energy
                        food[j].active = 0

            # Death from starvation or old age
            if creatures[i].energy <= 0 or creatures[i].age > 80:
                creatures[i].alive = 0
                total_alive[None] -= 1

    # Update statistics
    max_generation[None] = max_gen


@ti.kernel
def reproduce():
    for i in range(MAX_CREATURES):
        if creatures[i].alive == 1 and creatures[i].energy > 80:
            # Find empty slot for offspring
            for j in range(MAX_CREATURES):
                if creatures[j].alive == 0:
                    # Create offspring with mutations
                    creatures[j].pos = creatures[i].pos + ti.math.vec2(
                        ti.random() * 20 - 10,
                        ti.random() * 20 - 10
                    )
                    creatures[j].vel = ti.math.vec2(0, 0)
                    creatures[j].energy = 30.0
                    creatures[j].age = 0.0
                    creatures[j].alive = 1
                    creatures[j].generation = creatures[i].generation + 1
                    creatures[j].wander_dir = ti.random() * 2.0 * 3.14159
                    creatures[j].wander_steps = 0

                    # Inherit traits with mutations
                    mutation = 1.0 + (ti.random() - 0.5) * MUTATION_STRENGTH
                    creatures[j].speed = creatures[i].speed * mutation
                    creatures[j].speed = ti.max(0.1, ti.min(3.0, creatures[j].speed))

                    mutation = 1.0 + (ti.random() - 0.5) * MUTATION_STRENGTH
                    creatures[j].size = creatures[i].size * mutation
                    creatures[j].size = ti.max(1.0, ti.min(15.0, creatures[j].size))

                    mutation = 1.0 + (ti.random() - 0.5) * MUTATION_STRENGTH
                    creatures[j].vision_range = creatures[i].vision_range * mutation
                    creatures[j].vision_range = ti.max(10.0, ti.min(150.0, creatures[j].vision_range))

                    mutation = 1.0 + (ti.random() - 0.5) * MUTATION_STRENGTH
                    creatures[j].efficiency = creatures[i].efficiency * mutation
                    creatures[j].efficiency = ti.max(0.2, ti.min(2.0, creatures[j].efficiency))

                    # Slightly mutate color
                    creatures[j].color = creatures[i].color + ti.math.vec3(
                        (ti.random() - 0.5) * 0.1,
                        (ti.random() - 0.5) * 0.1,
                        (ti.random() - 0.5) * 0.1
                    )
                    creatures[j].color = ti.max(0.0, ti.min(1.0, creatures[j].color))

                    # Parent loses energy
                    creatures[i].energy -= 40.0
                    break


@ti.kernel
def spawn_food():
    for i in range(MAX_FOOD):
        if food[i].active == 0:
            if ti.random() < FOOD_SPAWN_RATE:
                food[i].pos = ti.math.vec2(
                    ti.random() * WIDTH,
                    ti.random() * HEIGHT
                )
                food[i].energy = 10.0 + ti.random() * 20.0
                food[i].active = 1


@ti.kernel
def render():
    # Clear screen with white background
    for i, j in pixels:
        pixels[i, j] = ti.math.vec3(1, 1, 1)

    # Draw food
    for idx in range(MAX_FOOD):
        if food[idx].active == 1:
            x = ti.cast(food[idx].pos.x, ti.i32)
            y = ti.cast(food[idx].pos.y, ti.i32)
            size = ti.cast(food[idx].energy / 5, ti.i32)
            for i in range(ti.max(0, x - size), ti.min(WIDTH, x + size + 1)):
                for j in range(ti.max(0, y - size), ti.min(HEIGHT, y + size + 1)):
                    if (i - x) ** 2 + (j - y) ** 2 <= size ** 2:
                        pixels[i, j] = ti.math.vec3(0.2, 0.8, 0.2)

    # Draw creatures
    for idx in range(MAX_CREATURES):
        if creatures[idx].alive == 1:
            x = ti.cast(creatures[idx].pos.x, ti.i32)
            y = ti.cast(creatures[idx].pos.y, ti.i32)
            size = ti.cast(creatures[idx].size, ti.i32)

            # Draw body
            for i in range(ti.max(0, x - size), ti.min(WIDTH, x + size + 1)):
                for j in range(ti.max(0, y - size), ti.min(HEIGHT, y + size + 1)):
                    if (i - x) ** 2 + (j - y) ** 2 <= size ** 2:
                        # Color intensity based on energy
                        intensity = ti.min(1.0, creatures[idx].energy / 100.0)
                        pixels[i, j] = creatures[idx].color * intensity

            # Draw vision range indicator (faint circle)
            vision = ti.cast(creatures[idx].vision_range, ti.i32)
            num_points = 36  # Draw 36 points around the circle
            for i in range(num_points):
                angle = i * 10  # 0, 10, 20, ... 350 degrees
                rad = angle * 3.14159 / 180.0
                px = x + ti.cast(vision * ti.cos(rad), ti.i32)
                py = y + ti.cast(vision * ti.sin(rad), ti.i32)
                if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                    pixels[px, py] = ti.math.vec3(0.3, 0.3, 0.3)


@ti.kernel
def render_paused():
    # Render current state but with a slightly dimmed overlay
    for i, j in pixels:
        pixels[i, j] *= 0.8  # Dim the image to show it's paused


@ti.kernel
def render_start_screen():
    # Show start screen
    for i, j in pixels:
        pixels[i, j] = ti.math.vec3(0.9, 0.9, 0.9)

    # Add "PRESS START" text effect (simple pattern)
    center_x, center_y = WIDTH // 2, HEIGHT // 2
    for i in range(center_x - 150, center_x + 150):
        for j in range(center_y - 15, center_y + 15):
            if 0 <= i < WIDTH and 0 <= j < HEIGHT:
                pixels[i, j] = ti.math.vec3(0.5, 0.5, 0.5)


def main():
    window = ti.ui.Window("Evolution Simulator", (WIDTH, HEIGHT), show_window=True)
    canvas = window.get_canvas()
    gui = window.get_gui()

    # Initialize simulation data but don't start yet
    init_simulation()

    # Simulation state control
    simulation_running = False
    simulation_initialized = False

    frame = 0
    dt = 0.016  # 60 FPS

    while window.running:
        with gui.sub_window(name='Controls', x=0, y=0, width=0.3, height=0.15):
            if gui.button(text='Start'):
                if not simulation_initialized:
                    init_simulation()
                    simulation_initialized = True
                simulation_running = True

            if gui.button(text='Pause'):
                simulation_running = False

            if gui.button(text='Reset'):
                reset_simulation()
                init_simulation()
                simulation_initialized = True
                simulation_running = False
                frame = 0

            gui.text(f"Status: {'Running' if simulation_running else 'Paused'}")
            gui.text(f"Generation: {max_generation[None] if simulation_initialized else 0}")
            gui.text(f"Alive: {total_alive[None] if simulation_initialized else 0}")

            # Parameter controls
            mutation = gui.slider_float(text='Mutation Rate', old_value=MUTATION_RATE, minimum=0.0, maximum=1.0)
            max_food = gui.slider_int(text='Max Food', old_value=MAX_FOOD, minimum=1, maximum=500)
            food_spawn_rate = gui.slider_float(text='Food Spawn Rate', old_value=FOOD_SPAWN_RATE, minimum=0.0,
                                               maximum=1.0)
            mutation_strength = gui.slider_float(text='Mutation Strength', old_value=MUTATION_STRENGTH, minimum=0.0,
                                                 maximum=1.0)

        # Only update simulation if running
        if simulation_running and simulation_initialized:
            # Update simulation
            update_creatures(dt)

            # Reproduction every 30 frames
            if frame % 30 == 0:
                reproduce()

            # Spawn food
            spawn_food()

            # Render simulation
            render()

            frame += 1

        elif simulation_initialized and not simulation_running:
            # Render paused state
            render()
            render_paused()

        else:
            # Show start screen
            render_start_screen()

        # Display
        canvas.set_image(pixels)
        window.show()

    gui.end()


if __name__ == "__main__":
    main()
