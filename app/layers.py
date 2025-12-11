import taichi as ti
from core.config import MUTATION_RATE, MAX_FOOD, FOOD_SPAWN_RATE, MUTATION_STRENGTH
from core.events import EventManager, Event, EventType

class Layer:
    def on_attach(self):
        pass

    def on_detach(self):
        pass

    def on_update(self, dt: float):
        pass

    def on_render(self):
        pass

    def on_gui_render(self, gui):
        pass

class SimulationLayer(Layer):
    def __init__(self, simulation, renderer, event_manager: EventManager):
        self.simulation = simulation
        self.renderer = renderer
        self.event_manager = event_manager
        self.running = False
        self.initialized = False
        self.frame = 0

    def on_attach(self):
        self.simulation.initialize()
        self.initialized = True

    def start(self):
        if not self.initialized:
            self.simulation.initialize()
            self.initialized = True
        self.running = True
        self.event_manager.emit(Event(EventType.SIMULATION_START))

    def pause(self):
        self.running = False
        self.event_manager.emit(Event(EventType.SIMULATION_PAUSE))

    def reset(self):
        self.simulation.reset()
        self.simulation.initialize()
        self.initialized = True
        self.running = False
        self.frame = 0

    def on_update(self, dt: float):
        if self.running and self.initialized:
            self.simulation.update(dt)

            if self.frame % 30 == 0:
                self.simulation.reproduce_creatures()

            self.simulation.spawn_food_items()
            self.frame += 1

    def on_render(self):
        if self.initialized and self.running:
            self.renderer.render_scene()
        elif self.initialized and not self.running:
            self.renderer.render_scene()
            self.renderer.render_paused_overlay()
        else:
            self.renderer.render_start_screen()

class UILayer(Layer):
    def __init__(self, simulation_layer: SimulationLayer, simulation):
        self.simulation_layer = simulation_layer
        self.simulation = simulation
        self.mutation_rate = MUTATION_RATE
        self.max_food = MAX_FOOD
        self.food_spawn_rate = FOOD_SPAWN_RATE
        self.mutation_strength = MUTATION_STRENGTH

    def on_gui_render(self, gui):
        with gui.sub_window(name='Controls', x=0, y=0, width=0.3, height=0.15):
            if gui.button(text='Start'):
                self.simulation_layer.start()

            if gui.button(text='Pause'):
                self.simulation_layer.pause()

            if gui.button(text='Reset'):
                self.simulation_layer.reset()

            status = 'Running' if self.simulation_layer.running else 'Paused'
            gui.text(f"Status: {status}")

            generation = self.simulation.max_generation[None] if self.simulation_layer.initialized else 0
            gui.text(f"Generation: {generation}")

            alive = self.simulation.total_alive[None] if self.simulation_layer.initialized else 0
            gui.text(f"Alive: {alive}")

            self.mutation_rate = gui.slider_float(
                text='Mutation Rate',
                old_value=self.mutation_rate,
                minimum=0.0,
                maximum=1.0
            )
            self.max_food = gui.slider_int(
                text='Max Food',
                old_value=self.max_food,
                minimum=1,
                maximum=500
            )
            self.food_spawn_rate = gui.slider_float(
                text='Food Spawn Rate',
                old_value=self.food_spawn_rate,
                minimum=0.0,
                maximum=1.0
            )
            self.mutation_strength = gui.slider_float(
                text='Mutation Strength',
                old_value=self.mutation_strength,
                minimum=0.0,
                maximum=1.0
            )
