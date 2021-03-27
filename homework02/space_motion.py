"""
Write a program that simulates the motion of objects in space. (stars/planets/satellites etc.)
Any two objects are attracted to each other by the force of gravitation which can be calculated using the following formula:
(The formula is written in TeX, you can use free online tools such as latexbase.com to view it)

$$\vec{F_{ij}} = \frac{G m_i m_j}{r_{ij}^2} \frac{\vec{r_{ij}}}{\|r_{ij}\|}.$$

The force that acts on an object $i$ can be calculated as the vector sum of all forces acting on the object.
In our case, we only need to account for the gravitational forces caused by other objects in the simulation.

$$\vec{F_i} = \sum_{j \neq i} \vec{F_{ij}}$$

You are required to write four functions:

- "calculate_force" will calculate the force acting on an object based on other objects' current state in the simulation.
- "update_space_object" will calculate new coordinates and speed for an object based on the return value of calculate_force, the timestep and the object's current state.
- "update_motion" will simulate the motion of all objects for a single timestep (the size of the timestep is passed in)
- "simulate_motion" is a generator function that yields a dictionary with object names as keys and tuples of (x,y) coordinates as values. Each iteration progresses the simulation by the defined timestep.

More precise descriptions of each function's required functionality can be found in the code below.
For examples of function calls, check the test script `test_space_motion.py`.

You also have to write a parametrized decorator "logging" that will measure how many times a function
has been called and how long it ran. The information should be printed to standard output in this form:
"function_name - number_of_calls - time units\n".
The decorator should have an optional "units" parameter for specifying the output format (default is 'ms').
The decorator should accept 'ns', 'us', 'ms', 's', 'min', 'h' and 'days' as values for the "units" parameter.
The time should be printed as a float number with exactly 3 decimal places (eg. 0.042).

A couple more things:

- Use scientific notation (SI units).
- Do not bother with optimisation; just write something that works! (That is a nice thing to hear for once, isn't it?)
- Any function that takes multiple space objects should take them as separately named tuples, not as a list of them.

    f(5, earth, moon, mars, sun) - GOOD
    f(5, (earth, moon, mars, sun)) - MUCH BAD

Good luck!
"""
import time  # measuring time
import datetime
from collections import namedtuple
from math import sqrt, floor

# Define universal gravitation constant

G = 6.67408e-11  # N-m2/kg2
SpaceObject = namedtuple('SpaceObject', 'name mass x y vx vy color')
Force = namedtuple('Force', 'fx fy')

# time_stamp_ns = {'ns': 1, 'us': 10 ** 3, 'ms': 10 ** 6}
# time_stamp_s = {'s': 1, 'min': 60, 'h': 3600, 'days': 3600 * 24}
time_func_unit = {'ns': (time.time_ns, 1), 'us': (time.time_ns, 10 ** 3), 'ms': (time.time_ns, 10 ** 6),
                  's': (time.time, 1), 'min': (time.time, 60), 'h': (time.time, 3600), 'days': (time.time, 3600 * 24)}


def truncate(number, ndigints):
    return floor(number * 10 ** ndigints) / 10 ** ndigints


def logging(unit='ms'):
    time_func, time_unit_multiplier = time_func_unit[unit]

    def decorator(func):
        num_of_calls = 0

        def logger(*args, **kwargs):
            nonlocal num_of_calls
            nonlocal time_func
            nonlocal time_unit_multiplier
            num_of_calls += 1
            # calling passed function
            start_time = time_func()
            returned = func(*args, **kwargs)
            run_time = time_func() - start_time
            print(f"{func.__name__} - {num_of_calls} - {truncate(run_time.real / time_unit_multiplier, 3)} {unit}")
            # print(f"{func.__name__} - {num_of_calls} - {round(run_time.real / time_unit_multiplier, 3)} {unit}")
            return returned

        return logger

    return decorator


@logging(unit='ms')
def calculate_force(i: SpaceObject, *other_objects: SpaceObject):
    # input: one of the space objects (indexed as i in below formulas), other space objects (indexed as j, may be any number of them)
    # returns named tuple (see above) that represents x and y components of the gravitational force
    # calculate force (vector) for each pair (space_object, other_space_object):
    # |F_ij| = G*m_i*m_j/distance^2
    # F_x = |F_ij| * (other_object.x-space_object.x)/distance
    # analogous for F_y
    # for each coordinate (x, y) it sums force from all other space objects

    total_f_x = 0
    total_f_y = 0

    # iterate over all passed space objects
    for j in other_objects:
        # does nothing when i == j
        if i == j:
            continue

        distance_square = (j.x - i.x) ** 2 + (j.y - i.y) ** 2
        f_ij = (G * i.mass * j.mass) / distance_square
        total_f_x += f_ij * (j.x - i.x) / sqrt(distance_square)
        total_f_y += f_ij * (j.y - i.y) / sqrt(distance_square)

    return Force(fx=total_f_x, fy=total_f_y)


@logging(unit='s')
def update_space_object(o: SpaceObject, force: Force, timestep: int):
    # here we update coordinates and speed of the object based on the force that acts on it
    # input: space_object we want to update (evolve in time), force (from all other objects) that acts on it, size of timestep
    # returns: named tuple (see above) that contains updated coordinates and speed for given space_object
    # hint:
    # acceleration_x = force_x / mass

    speed_change_x = (force.fx / o.mass) * timestep
    speed_change_y = (force.fy / o.mass) * timestep

    speed_new_x = o.vx + speed_change_x
    speed_new_y = o.vy + speed_change_y

    x_final = o.x + speed_new_x * timestep
    y_final = o.y + speed_new_y * timestep

    return SpaceObject(name=o.name, mass=o.mass, x=x_final, y=y_final, vx=speed_new_x, vy=speed_new_y, color=o.color)


@logging(unit='ms')
def update_motion(timestep_size: int, *objects: SpaceObject):
    # input: timestep and space objects we want to simulate (as named tuples above)
    # returns: list or tuple with updated objects
    # hint:
    # iterate over space objects, for given space object calculate_force with function above, update
    updated_space_objects = []

    for o in objects:
        updated_space_objects += [update_space_object(o, calculate_force(o, *objects), timestep_size)]

    return updated_space_objects  # (named tuple with x and y)


@logging()
def simulate_motion(timestep: int, timestep_amount: int, *objects: SpaceObject):
    # generator that in every iteration yields dictionary with the name of the objects as a key and tuple of coordinates (x first, y second) as values
    # input size of the timestep, number of timesteps (integer), space objects (any number of them)
    for _ in range(timestep_amount):
        objects = update_motion(timestep, *objects)
        yield {o.name: (o.x, o.y) for o in objects}
