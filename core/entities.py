import taichi as ti

@ti.dataclass
class Creature:
    pos: ti.math.vec2
    vel: ti.math.vec2
    energy: ti.f32
    age: ti.f32
    alive: ti.i32
    speed: ti.f32
    size: ti.f32
    vision_range: ti.f32
    efficiency: ti.f32
    color: ti.math.vec3
    generation: ti.i32
    wander_dir: ti.f32
    wander_steps: ti.i32

@ti.dataclass
class Food:
    pos: ti.math.vec2
    energy: ti.f32
    active: ti.i32
