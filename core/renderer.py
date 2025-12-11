
import taichi as ti
from core.entities import Creature, Food
from core.config import WIDTH, HEIGHT, MAX_CREATURES, MAX_FOOD


@ti.data_oriented
class Renderer:
    def __init__(self, creatures: ti.template(), food: ti.template()):
        self.creatures = creatures
        self.food = food
        self.pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

    def render_scene(self):
        self._render()

    def render_paused_overlay(self):
        self._render_paused()

    def render_start_screen(self):
        self._render_start_screen()

    def get_pixels(self):
        return self.pixels

    @ti.kernel
    def _render(self):
        for i, j in self.pixels:
            self.pixels[i, j] = ti.math.vec3(1, 1, 1)

        for idx in range(MAX_FOOD):
            if self.food[idx].active == 1:
                x = ti.cast(self.food[idx].pos.x, ti.i32)
                y = ti.cast(self.food[idx].pos.y, ti.i32)
                size = ti.cast(self.food[idx].energy / 5, ti.i32)
                size_sq = size * size
                for i in range(ti.max(0, x - size), ti.min(WIDTH, x + size + 1)):
                    dx = i - x
                    dx_sq = dx * dx
                    for j in range(ti.max(0, y - size), ti.min(HEIGHT, y + size + 1)):
                        dy = j - y
                        if dx_sq + dy * dy <= size_sq:
                            self.pixels[i, j] = ti.math.vec3(0.2, 0.8, 0.2)

        for idx in range(MAX_CREATURES):
            if self.creatures[idx].alive == 1:
                x = ti.cast(self.creatures[idx].pos.x, ti.i32)
                y = ti.cast(self.creatures[idx].pos.y, ti.i32)
                size = ti.cast(self.creatures[idx].size, ti.i32)
                size_sq = size * size
                intensity = ti.math.min(1.0, self.creatures[idx].energy / 100.0)
                color = self.creatures[idx].color * intensity

                for i in range(ti.max(0, x - size), ti.min(WIDTH, x + size + 1)):
                    dx = i - x
                    dx_sq = dx * dx
                    for j in range(ti.max(0, y - size), ti.min(HEIGHT, y + size + 1)):
                        dy = j - y
                        if dx_sq + dy * dy <= size_sq:
                            self.pixels[i, j] = color

    @ti.kernel
    def _render_paused(self):
        for i, j in self.pixels:
            self.pixels[i, j] *= 0.8

    @ti.kernel
    def _render_start_screen(self):
        for i, j in self.pixels:
            self.pixels[i, j] = ti.math.vec3(0.9, 0.9, 0.9)

        center_x, center_y = WIDTH // 2, HEIGHT // 2

        for i in range(center_x - 150, center_x + 150):
            for j in range(center_y - 15, center_y + 15):
                if 0 <= i < WIDTH and 0 <= j < HEIGHT:
                    self.pixels[i, j] = ti.math.vec3(0.5, 0.5, 0.5)
