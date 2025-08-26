# Evolution Simulator  

An interactive simulation of evolving creatures built with [Taichi](https://taichi-lang.org/).  
Creatures wander, search for food, reproduce, mutate, and die — all while evolving traits such as speed, size, vision range, and efficiency.  

---

##  Features  
- **Evolving population**: Creatures inherit traits with random mutations.  
- **Genetic traits**:  
  - Speed  
  - Size  
  - Vision range  
  - Energy efficiency  
  - Body color (slight variation with each generation)  
- **Natural selection**: Energy costs, old age, and starvation remove weaker creatures.  
- **GUI Controls**:  
  - Start, Pause, Reset 
  - Adjust mutation rate, mutation strength, max food, and food spawn rate in real time.  
- **Visualization**:  
  - Creatures shown as colored circles (size scales with body size).  
  - Energy levels affect brightness.  
  - Vision range displayed as a faint circle around each creature.  
  - Food appears as green dots.  

---

## Getting Started  

### Prerequisites  
- Python 3.8+  
- [Taichi](https://docs.taichi-lang.org/)  

Install dependencies:  
```bash
pip install taichi
