import taichi as ti
from core.entities import Creature, Food
from core.config import (
    WIDTH, HEIGHT, MAX_CREATURES, MAX_FOOD,
    INITIAL_CREATURES, INITIAL_FOOD, FOOD_SPAWN_RATE,
    MUTATION_STRENGTH, GRID_WIDTH, GRID_HEIGHT,
    GRID_CELL_SIZE, MAX_ITEMS_PER_CELL
)
from core.events import EventManager, Event, EventType

@ti.data_oriented
class Simulation:
    def __init__(self, event_manager: EventManager):
        self.event_manager = event_manager
        self.creatures = Creature.field(shape=MAX_CREATURES)
        self.food = Food.field(shape=MAX_FOOD)
        self.total_alive = ti.field(dtype=ti.i32, shape=())
        self.max_generation = ti.field(dtype=ti.i32, shape=())

        self.food_grid = ti.field(dtype=ti.i32, shape=(GRID_WIDTH, GRID_HEIGHT, MAX_ITEMS_PER_CELL))
        self.food_grid_count = ti.field(dtype=ti.i32, shape=(GRID_WIDTH, GRID_HEIGHT))

    def initialize(self):
        self._init_simulation()
        self.event_manager.emit(Event(EventType.SIMULATION_START))

    def reset(self):
        self._reset_simulation()
        self.event_manager.emit(Event(EventType.SIMULATION_RESET))

    def update(self, dt: float, should_reproduce: bool):
        self._update_all(dt, should_reproduce)

    @ti.kernel
    def _init_simulation(self):
        for i in range(INITIAL_CREATURES):
            self._set_creature(i, alive=1)

        for i in range(INITIAL_FOOD):
            self.food[i].pos = ti.math.vec2(
                ti.random() * WIDTH,
                ti.random() * HEIGHT
            )
            self.food[i].energy = 10.0 + ti.random() * 20.0
            self.food[i].active = 1

    @ti.func
    def _world_to_grid(self, pos: ti.math.vec2) -> ti.math.ivec2:
        grid_x = ti.cast(pos.x / GRID_CELL_SIZE, ti.i32)
        grid_y = ti.cast(pos.y / GRID_CELL_SIZE, ti.i32)
        grid_x = ti.math.clamp(grid_x, 0, GRID_WIDTH - 1)
        grid_y = ti.math.clamp(grid_y, 0, GRID_HEIGHT - 1)
        return ti.math.ivec2(grid_x, grid_y)

    @ti.func
    def _wrap_position(self, pos: ti.math.vec2) -> ti.math.vec2:
        result = pos
        if result.x < 0:
            result.x = WIDTH
        elif result.x > WIDTH:
            result.x = 0
        if result.y < 0:
            result.y = HEIGHT
        elif result.y > HEIGHT:
            result.y = 0
        return result

    @ti.func
    def _mutate_value(self, value: ti.f32, min_val: ti.f32, max_val: ti.f32) -> ti.f32:
        mutation = 1.0 + (ti.random() - 0.5) * MUTATION_STRENGTH
        return ti.math.clamp(value * mutation, min_val, max_val)

    @ti.func
    def _find_nearest_food(self, creature_pos: ti.math.vec2, vision_range: ti.f32) -> ti.i32:
        nearest_food = -1
        min_dist_sq = vision_range * vision_range

        creature_grid_pos = self._world_to_grid(creature_pos)
        search_radius = ti.cast(vision_range / GRID_CELL_SIZE, ti.i32) + 1

        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                check_x = creature_grid_pos.x + dx
                check_y = creature_grid_pos.y + dy

                if 0 <= check_x < GRID_WIDTH and 0 <= check_y < GRID_HEIGHT:
                    cell_count = self.food_grid_count[check_x, check_y]
                    for idx in range(cell_count):
                        j = self.food_grid[check_x, check_y, idx]
                        if self.food[j].active == 1:
                            dist_sq = ti.math.distance(creature_pos, self.food[j].pos)
                            dist_sq = dist_sq * dist_sq

                            if dist_sq < min_dist_sq:
                                min_dist_sq = dist_sq
                                nearest_food = j

        return nearest_food

    @ti.func
    def _set_creature(self, i: int, alive: ti.i32):
        self.creatures[i].pos = ti.math.vec2(
            ti.random() * WIDTH,
            ti.random() * HEIGHT
        )
        self.creatures[i].vel = ti.math.vec2(0, 0)
        self.creatures[i].energy = 50.0
        self.creatures[i].age = 0.0
        self.creatures[i].alive = alive
        self.creatures[i].speed = 1.0 + ti.random() * 0.4
        self.creatures[i].size = 3.0 + ti.random() * 4.0
        self.creatures[i].vision_range = 35.0 + ti.random() * 40.0
        self.creatures[i].efficiency = 0.8 + ti.random() * 0.4
        self.creatures[i].color = ti.math.vec3(
            0.2 + ti.random() * 0.8,
            0.2 + ti.random() * 0.8,
            0.2 + ti.random() * 0.8
        )
        self.creatures[i].generation = 0
        self.creatures[i].wander_dir = ti.random() * 2.0 * 3.14159
        self.creatures[i].wander_steps = 0

    @ti.kernel
    def _reset_simulation(self):
        for i in range(MAX_CREATURES):
            self._set_creature(i, alive=0)

        for i in range(MAX_FOOD):
            self.food[i].active = 0

        self.total_alive[None] = 0
        self.max_generation[None] = 0

    @ti.kernel
    def _update_all(self, dt: ti.f32, should_reproduce: ti.i32):
        for i, j in ti.ndrange(GRID_WIDTH, GRID_HEIGHT):
            self.food_grid_count[i, j] = 0

        for i in range(MAX_FOOD):
            if self.food[i].active == 0:
                if ti.random() < FOOD_SPAWN_RATE:
                    self.food[i].pos = ti.math.vec2(
                        ti.random() * WIDTH,
                        ti.random() * HEIGHT
                    )
                    self.food[i].energy = 10.0 + ti.random() * 20.0
                    self.food[i].active = 1

            if self.food[i].active == 1:
                grid_pos = self._world_to_grid(self.food[i].pos)
                count = ti.atomic_add(self.food_grid_count[grid_pos.x, grid_pos.y], 1)
                if count < MAX_ITEMS_PER_CELL:
                    self.food_grid[grid_pos.x, grid_pos.y, count] = i

        self.total_alive[None] = 0
        max_gen = 0

        for i in range(MAX_CREATURES):
            if self.creatures[i].alive == 1:
                self.total_alive[None] += 1
                max_gen = ti.max(max_gen, self.creatures[i].generation)

                self.creatures[i].age += dt
                energy_cost = dt * (0.5 + self.creatures[i].speed * 0.8 +
                                    self.creatures[i].size * 0.8 +
                                    self.creatures[i].vision_range * 0.01) / self.creatures[i].efficiency
                self.creatures[i].energy -= energy_cost

                nearest_food = self._find_nearest_food(self.creatures[i].pos, self.creatures[i].vision_range)
                eating_range_sq = (self.creatures[i].size + 5) * (self.creatures[i].size + 5)

                if nearest_food >= 0:
                    direction = ti.math.normalize(self.food[nearest_food].pos - self.creatures[i].pos)
                    self.creatures[i].vel = direction * self.creatures[i].speed * 50.0
                    self.creatures[i].wander_steps = 0
                else:
                    if self.creatures[i].wander_steps <= 0:
                        self.creatures[i].wander_dir = ti.random() * 2.0 * 3.14159
                        self.creatures[i].wander_steps = ti.cast(10 + ti.random() * 20, ti.i32)

                    angle_variation = (ti.random() - 0.5) * 0.2
                    current_angle = self.creatures[i].wander_dir + angle_variation
                    self.creatures[i].vel = ti.math.vec2(
                        ti.cos(current_angle) * self.creatures[i].speed * 30.0,
                        ti.sin(current_angle) * self.creatures[i].speed * 30.0
                    )
                    self.creatures[i].wander_steps -= 1

                self.creatures[i].pos = self._wrap_position(self.creatures[i].pos + self.creatures[i].vel * dt)

                if nearest_food >= 0:
                    dist_sq = ti.math.distance(self.creatures[i].pos, self.food[nearest_food].pos)
                    dist_sq = dist_sq * dist_sq
                    if dist_sq < eating_range_sq:
                        self.creatures[i].energy += self.food[nearest_food].energy
                        self.food[nearest_food].active = 0

                if self.creatures[i].energy <= 0 or self.creatures[i].age > 80:
                    self.creatures[i].alive = 0
                    self.total_alive[None] -= 1

        self.max_generation[None] = max_gen

        if should_reproduce == 1:
            for i in range(MAX_CREATURES):
                if self.creatures[i].alive == 1 and self.creatures[i].energy > 80:
                    for j in range(MAX_CREATURES):
                        if self.creatures[j].alive == 0:
                            self.creatures[j].pos = self.creatures[i].pos + ti.math.vec2(
                                ti.random() * 20 - 10,
                                ti.random() * 20 - 10
                            )
                            self.creatures[j].vel = ti.math.vec2(0, 0)
                            self.creatures[j].energy = 30.0
                            self.creatures[j].age = 0.0
                            self.creatures[j].alive = 1
                            self.creatures[j].generation = self.creatures[i].generation + 1
                            self.creatures[j].wander_dir = ti.random() * 2.0 * 3.14159
                            self.creatures[j].wander_steps = 0

                            self.creatures[j].speed = self._mutate_value(self.creatures[i].speed, 0.1, 3.0)
                            self.creatures[j].size = self._mutate_value(self.creatures[i].size, 1.0, 15.0)
                            self.creatures[j].vision_range = self._mutate_value(self.creatures[i].vision_range, 10.0, 150.0)
                            self.creatures[j].efficiency = self._mutate_value(self.creatures[i].efficiency, 0.2, 2.0)

                            self.creatures[j].color = ti.math.clamp(
                                self.creatures[i].color + ti.math.vec3(
                                    (ti.random() - 0.5) * 0.1,
                                    (ti.random() - 0.5) * 0.1,
                                    (ti.random() - 0.5) * 0.1
                                ),
                                0.0, 1.0
                            )

                            self.creatures[i].energy -= 40.0
                            break

