import taichi as ti
from core.entities import Creature, Food
from core.config import (
    WIDTH, HEIGHT, MAX_CREATURES, MAX_FOOD,
    INITIAL_CREATURES, INITIAL_FOOD, FOOD_SPAWN_RATE,
    MUTATION_STRENGTH
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

    def initialize(self):
        self._init_simulation()
        self.event_manager.emit(Event(EventType.SIMULATION_START))

    def reset(self):
        self._reset_simulation()
        self.event_manager.emit(Event(EventType.SIMULATION_RESET))

    def update(self, dt: float):
        self._update_creatures(dt)

    def reproduce_creatures(self):
        self._reproduce()

    def spawn_food_items(self):
        self._spawn_food()

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
    def _update_creatures(self, dt: ti.f32):
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

                nearest_food = -1
                min_dist = self.creatures[i].vision_range

                for j in range(MAX_FOOD):
                    if self.food[j].active == 1:
                        dist = (self.creatures[i].pos - self.food[j].pos).norm()
                        if dist < min_dist:
                            min_dist = dist
                            nearest_food = j

                if nearest_food >= 0:
                    direction = (self.food[nearest_food].pos - self.creatures[i].pos).normalized()
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

                self.creatures[i].pos += self.creatures[i].vel * dt

                if self.creatures[i].pos.x < 0:
                    self.creatures[i].pos.x = WIDTH
                elif self.creatures[i].pos.x > WIDTH:
                    self.creatures[i].pos.x = 0
                if self.creatures[i].pos.y < 0:
                    self.creatures[i].pos.y = HEIGHT
                elif self.creatures[i].pos.y > HEIGHT:
                    self.creatures[i].pos.y = 0

                for j in range(MAX_FOOD):
                    if self.food[j].active == 1:
                        dist = (self.creatures[i].pos - self.food[j].pos).norm()
                        if dist < self.creatures[i].size + 5:
                            self.creatures[i].energy += self.food[j].energy
                            self.food[j].active = 0

                if self.creatures[i].energy <= 0 or self.creatures[i].age > 80:
                    self.creatures[i].alive = 0
                    self.total_alive[None] -= 1

        self.max_generation[None] = max_gen

    @ti.kernel
    def _reproduce(self):
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

                        mutation = 1.0 + (ti.random() - 0.5) * MUTATION_STRENGTH
                        self.creatures[j].speed = self.creatures[i].speed * mutation
                        self.creatures[j].speed = ti.max(0.1, ti.min(3.0, self.creatures[j].speed))

                        mutation = 1.0 + (ti.random() - 0.5) * MUTATION_STRENGTH
                        self.creatures[j].size = self.creatures[i].size * mutation
                        self.creatures[j].size = ti.max(1.0, ti.min(15.0, self.creatures[j].size))

                        mutation = 1.0 + (ti.random() - 0.5) * MUTATION_STRENGTH
                        self.creatures[j].vision_range = self.creatures[i].vision_range * mutation
                        self.creatures[j].vision_range = ti.max(10.0, ti.min(150.0, self.creatures[j].vision_range))

                        mutation = 1.0 + (ti.random() - 0.5) * MUTATION_STRENGTH
                        self.creatures[j].efficiency = self.creatures[i].efficiency * mutation
                        self.creatures[j].efficiency = ti.max(0.2, ti.min(2.0, self.creatures[j].efficiency))

                        self.creatures[j].color = self.creatures[i].color + ti.math.vec3(
                            (ti.random() - 0.5) * 0.1,
                            (ti.random() - 0.5) * 0.1,
                            (ti.random() - 0.5) * 0.1
                        )
                        self.creatures[j].color = ti.max(0.0, ti.min(1.0, self.creatures[j].color))

                        self.creatures[i].energy -= 40.0
                        break

    @ti.kernel
    def _spawn_food(self):
        for i in range(MAX_FOOD):
            if self.food[i].active == 0:
                if ti.random() < FOOD_SPAWN_RATE:
                    self.food[i].pos = ti.math.vec2(
                        ti.random() * WIDTH,
                        ti.random() * HEIGHT
                    )
                    self.food[i].energy = 10.0 + ti.random() * 20.0
                    self.food[i].active = 1
