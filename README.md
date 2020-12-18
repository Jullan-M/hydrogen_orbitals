# Hydrogen orbital visualization
Create 3D/2D density plot animations of hydrogen atom probability densities.
## Example
https://twitter.com/i/status/1339631942933897219

## How to use
```python
from hydrogen_orbitals import Orbital_3D, Transition_3D

orb = Orbital_3D(1,0,0) # Initial orbital to be transitioned from with quantum numbers n,l,m=1,0,0
orb.snapshot() # A still-view snapshot of the (1,0,0) 3D plot
tr_3d = Transition_3D(orb, fps=30) # Initialize Transition object that handles the animation from state to state.
tr_3d.wait(2) # Wait for 2 seconds in the animation in (1,0,0)
tr_3d.transition((2,1,0), duration=1) # Transition from (1,0,0) to (2,1,0) in the duration of 1 second
tr_3d.wait(1) # Wait for 1 second in (2,1,0)
tr_3d.transition((2,1,1), duration=0.5)
tr_3d.wait(2)
tr_3d.save() # Actually save the animation after every operation is done
```
