import taichi as ti
from core.config import WIDTH, HEIGHT
from core.simulation import Simulation
from core.renderer import Renderer
from core.events import EventManager
from app.layers import SimulationLayer, UILayer

class Application:
    def __init__(self):
        ti.init(arch=ti.gpu)

        self.event_manager = EventManager()
        self.simulation = Simulation(self.event_manager)
        self.renderer = Renderer(self.simulation.creatures, self.simulation.food)

        self.layers = []
        self.dt = 0.016

    def add_layer(self, layer):
        self.layers.append(layer)
        layer.on_attach()

    def run(self):
        window = ti.ui.Window("Evolution Simulator", (WIDTH, HEIGHT), show_window=True)
        canvas = window.get_canvas()
        gui = window.get_gui()

        while window.running:
            for layer in self.layers:
                layer.on_update(self.dt)

            for layer in self.layers:
                layer.on_render()

            for layer in self.layers:
                layer.on_gui_render(gui)

            canvas.set_image(self.renderer.get_pixels())
            window.show()

        gui.end()

def main():
    app = Application()

    simulation_layer = SimulationLayer(app.simulation, app.renderer, app.event_manager)
    ui_layer = UILayer(simulation_layer, app.simulation)

    app.add_layer(simulation_layer)
    app.add_layer(ui_layer)

    app.run()

if __name__ == "__main__":
    main()
