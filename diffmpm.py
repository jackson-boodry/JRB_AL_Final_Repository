import taichi as ti
import argparse
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import random


real = ti.f32
ti.init(default_fp=real, arch=ti.gpu, flatten_if=True)

dim = 2
n_particles = 8192
n_solid_particles = 0
n_actuators = 0
n_grid = 128
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 1e-3
learning_rate = 0.1  # Learning rate for gradient descent
p_vol = 1
E = 10
# TODO: update
mu = E
la = E
max_steps = 2048
steps = 1024
gravity = 3.8
target = [0.8, 0.2]

scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(dim, dtype=real)
mat = lambda: ti.Matrix.field(dim, dim, dtype=real)

actuator_id = ti.field(ti.i32)
particle_type = ti.field(ti.i32)
x, v = vec(), vec()
grid_v_in, grid_m_in = vec(), scalar()
grid_v_out = vec()
C, F = mat(), mat()

loss = scalar()

n_sin_waves = 4
weights = scalar()
bias = scalar()
x_avg = vec()

actuation = scalar()
actuation_omega = 40
act_strength = 4

# Fields for orientation detection
adv_factor = 10000
init_omega = 10000
min_idx = ti.field(ti.i32, shape=())
max_idx = ti.field(ti.i32, shape=())
initial_min_idx = ti.field(ti.i32, shape=())
initial_max_idx = ti.field(ti.i32, shape=())
vector_current = ti.Vector.field(2, dtype=real, shape=())
check_var = ti.field(ti.i32, shape=())
mult_var = ti.field(ti.f32, shape=())
omega_choice = ti.field(ti.f32, shape=())


def allocate_fields():
    ti.root.dense(ti.ij, (n_actuators, n_sin_waves)).place(weights)
    ti.root.dense(ti.i, n_actuators).place(bias)

    ti.root.dense(ti.ij, (max_steps, n_actuators)).place(actuation)
    ti.root.dense(ti.i, n_particles).place(actuator_id, particle_type)
    ti.root.dense(ti.k, max_steps).dense(ti.l, n_particles).place(x, v, C, F)
    ti.root.dense(ti.ij, n_grid).place(grid_v_in, grid_m_in, grid_v_out)
    ti.root.place(loss, x_avg)

    ti.root.lazy_grad()


@ti.kernel
def clear_grid():
    for i, j in grid_m_in:
        grid_v_in[i, j] = [0, 0]
        grid_m_in[i, j] = 0
        grid_v_in.grad[i, j] = [0, 0]
        grid_m_in.grad[i, j] = 0
        grid_v_out.grad[i, j] = [0, 0]


@ti.kernel
def clear_particle_grad():
    # for all time steps and all particles
    for f, i in x:
        x.grad[f, i] = [0, 0]
        v.grad[f, i] = [0, 0]
        C.grad[f, i] = [[0, 0], [0, 0]]
        F.grad[f, i] = [[0, 0], [0, 0]]


@ti.kernel
def clear_actuation_grad():
    for t, i in actuation:
        actuation[t, i] = 0.0


@ti.kernel
def p2g(f: ti.i32):
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_F = (ti.Matrix.diag(dim=2, val=1) + dt * C[f, p]) @ F[f, p]
        J = (new_F).determinant()
        if particle_type[p] == 0:  # fluid
            sqrtJ = ti.sqrt(J)
            new_F = ti.Matrix([[sqrtJ, 0], [0, sqrtJ]])

        F[f + 1, p] = new_F
        r, s = ti.polar_decompose(new_F)

        act_id = actuator_id[p]

        act = actuation[f, ti.max(0, act_id)] * act_strength
        if act_id == -1:
            act = 0.0
        # ti.print(act)

        A = ti.Matrix([[0.0, 0.0], [0.0, 1.0]]) * act
        cauchy = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        mass = 0.0
        if particle_type[p] == 0:
            mass = 4
            cauchy = ti.Matrix([[1.0, 0.0], [0.0, 0.1]]) * (J - 1) * E
        else:
            mass = 1
            cauchy = 2 * mu * (new_F - r) @ new_F.transpose() + \
                     ti.Matrix.diag(2, la * (J - 1) * J)
        cauchy += new_F @ A @ new_F.transpose()
        stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
        affine = stress + mass * C[f, p]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos = (ti.cast(ti.Vector([i, j]), real) - fx) * dx
                weight = w[i][0] * w[j][1]
                grid_v_in[base + offset] += weight * (mass * v[f, p] + affine @ dpos)
                grid_m_in[base + offset] += weight * mass


bound = 3
coeff = 0.5


@ti.kernel
def grid_op():
    for i, j in grid_m_in:
        inv_m = 1 / (grid_m_in[i, j] + 1e-10)
        v_out = inv_m * grid_v_in[i, j]
        v_out[1] -= dt * gravity
        if i < bound and v_out[0] < 0:
            v_out[0] = 0
            v_out[1] = 0
        if i > n_grid - bound and v_out[0] > 0:
            v_out[0] = 0
            v_out[1] = 0
        if j < bound and v_out[1] < 0:
            v_out[0] = 0
            v_out[1] = 0
            normal = ti.Vector([0.0, 1.0])
            lsq = (normal**2).sum()
            if lsq > 0.5:
                if ti.static(coeff < 0):
                    v_out[0] = 0
                    v_out[1] = 0
                else:
                    lin = v_out.dot(normal)
                    if lin < 0:
                        vit = v_out - lin * normal
                        lit = vit.norm() + 1e-10
                        if lit + coeff * lin <= 0:
                            v_out[0] = 0
                            v_out[1] = 0
                        else:
                            v_out = (1 + coeff * lin / lit) * vit
        if j > n_grid - bound and v_out[1] > 0:
            v_out[0] = 0
            v_out[1] = 0

        grid_v_out[i, j] = v_out


@ti.kernel
def g2p(f: ti.i32):
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, real)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector([0.0, 0.0])
        new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                dpos = ti.cast(ti.Vector([i, j]), real) - fx
                g_v = grid_v_out[base[0] + i, base[1] + j]
                weight = w[i][0] * w[j][1]
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx

        v[f + 1, p] = new_v
        x[f + 1, p] = x[f, p] + dt * v[f + 1, p]
        C[f + 1, p] = new_C


@ti.kernel
def compute_initial_indices():
    min_y = ti.f32(1e10)
    max_y = ti.f32(-1e10)
    initial_min_idx[None] = 0
    initial_max_idx[None] = 0

    for i in range(n_particles):
        if x[0, i].y < min_y:
            min_y = x[0, i].y
            initial_min_idx[None] = i
        if x[0, i].y > max_y:
            max_y = x[0, i].y
            initial_max_idx[None] = i


@ti.kernel
def compute_indices_and_vector(t: ti.i32):
    min_idx[None] = initial_min_idx[None]
    max_idx[None] = initial_max_idx[None]
    vector_current[None] = (x[t, max_idx[None]] - x[t, min_idx[None]]).normalized()

@ti.kernel
def compute_non_looping_statements():
    vector_base = ti.Vector([0.0, 1.0])
    check_num = vector_base.dot(vector_current[None])
    check_var[None] = ti.select(check_num < ti.sqrt(2) / 2, 1, 0)
    mult_var[None] = ti.select(check_var[None], adv_factor, 1)
    omega_choice[None] = ti.select(check_var[None], init_omega, actuation_omega)

@ti.kernel
def compute_orientation() -> ti.i32:
    vector_base = ti.Vector([0.0, 1.0])
    orientation_dot = vector_base.dot(vector_current[None])
    return ti.select(orientation_dot > ti.sqrt(2) / 2, 1, 0)  # 1 if upright, 0 if not

@ti.kernel
def compute_actuation(t: ti.i32, is_upright: ti.i32):
    for i in range(n_actuators):
        act = 0.0
        if is_upright == 1:
            # Jumping behavior: Strong, synchronized pulses
            jump_freq = actuation_omega * 0.25
            jump_phase = 2.0 * 3.14159 * i / n_actuators
            
            # Compute weighted sum of sine waves for jumping
            for j in ti.static(range(n_sin_waves)):
                phase = jump_freq * t * dt + jump_phase + 2.0 * 3.14159 * j / n_sin_waves
                base_signal = ti.sin(phase)
                # Reduced coefficients for gentler movement
                act += weights[i, j] * (1.8 * base_signal * base_signal + 0.2 * ti.sin(0.5 * phase))
        else:
            # Galloping behavior: Wave-like motion
            gallop_freq = actuation_omega * 0.75
            gallop_phase = 4.0 * 3.14159 * i / n_actuators
            
            # Compute weighted sum of sine waves for galloping
            for j in ti.static(range(n_sin_waves)):
                phase = gallop_freq * t * dt + gallop_phase + 2.0 * 3.14159 * j / n_sin_waves
                # Reduced coefficients for gentler movement
                act += weights[i, j] * (1.0 * ti.sin(phase) + 
                                      0.3 * ti.sin(2.0 * phase) +
                                      0.2 * ti.cos(0.5 * phase))
        
        # Add learnable bias
        act += bias[i]
        # Apply tanh activation to keep actuation bounded
        actuation[t, i] = ti.tanh(act)

@ti.kernel
def apply_weight_grad():
    # Update weights based on their gradients
    for i, j in ti.ndrange(n_actuators, n_sin_waves):
        weights[i, j] -= learning_rate * weights.grad[i, j]
    
    # Update biases based on their gradients
    for i in range(n_actuators):
        bias[i] -= learning_rate * bias.grad[i]

@ti.kernel
def compute_x_avg():
    for i in range(n_particles):
        contrib = 0.0
        if particle_type[i] == 1:
            contrib = 1.0 / n_solid_particles
        ti.atomic_add(x_avg[None], contrib * x[steps - 1, i])


@ti.kernel
def compute_loss():
    dist = x_avg[None][0]
    loss[None] = -dist


@ti.ad.grad_replaced
def advance(s):
    clear_grid()
    compute_indices_and_vector(s)
    compute_non_looping_statements()
    is_upright = compute_orientation()
    compute_actuation(s, is_upright)
    p2g(s)
    grid_op()
    g2p(s)


@ti.ad.grad_for(advance)
def advance_grad(s):
    clear_grid()
    p2g(s)
    grid_op()
    g2p.grad(s)
    grid_op.grad()
    p2g.grad(s)
    is_upright = compute_orientation()
    compute_actuation.grad(s, is_upright)


def forward(total_steps=steps):
    # simulation
    for s in range(total_steps - 1):
        advance(s)
    x_avg[None] = [0, 0]
    compute_x_avg()
    compute_loss()
    # Apply gradients to update weights and biases
    apply_weight_grad()


class Scene:
    def __init__(self):
        self.n_particles = 0
        self.n_solid_particles = 0
        self.x = []
        self.actuator_id = []
        self.particle_type = []
        self.offset_x = 0
        self.offset_y = 0

    def add_rect(self, x, y, w, h, actuation, ptype=1):
        if ptype == 0:
            assert actuation == -1
        global n_particles
        w_count = int(w / dx) * 2
        h_count = int(h / dx) * 2
        real_dx = w / w_count
        real_dy = h / h_count
        for i in range(w_count):
            for j in range(h_count):
                self.x.append([
                    x + (i + 0.5) * real_dx + self.offset_x,
                    y + (j + 0.5) * real_dy + self.offset_y
                ])
                self.actuator_id.append(actuation)
                self.particle_type.append(ptype)
                self.n_particles += 1
                self.n_solid_particles += int(ptype == 1)
    
    def add_circle(self, x, y, r, actuation, ptype=1, direction = 'up'):
        if ptype == 0:
            assert actuation == -1
            
        global n_particles

        r_count = int(2 * r / dx)
        real_dr = r/r_count
        n_theta_steps = int(2 * np.pi * r / dx)
        theta_step = (2 * np.pi/ n_theta_steps)


        for i in range(r_count):
            for j in range(n_theta_steps):
                radius = (i + 0.5) * real_dr
                theta = j * theta_step
                px = x + radius * np.cos(theta)
                py = y + radius * np.sin(theta)

                if (direction == 'up' and py >= y) or (direction == 'down' and py <= y) or (direction == 'left' and px <= x) or (direction == 'right' and px >= x):
                    self.x.append([px + self.offset_x, py + self.offset_y])
                    self.actuator_id.append(actuation)
                    self.particle_type.append(ptype)
                    self.n_particles += 1
                    self.n_solid_particles += int(ptype == 1)


    def set_offset(self, x, y):
        self.offset_x = x
        self.offset_y = y

    def finalize(self):
        global n_particles, n_solid_particles
        n_particles = self.n_particles
        n_solid_particles = self.n_solid_particles
        print('n_particles', n_particles)
        print('n_solid', n_solid_particles)

    def set_n_actuators(self, n_act):
        global n_actuators
        n_actuators = n_act


def fish(scene):
    scene.add_rect(0.025, 0.025, 0.95, 0.1, -1, ptype=0)
    scene.add_rect(0.1, 0.2, 0.15, 0.05, -1)
    scene.add_rect(0.1, 0.15, 0.025, 0.05, 0)
    scene.add_rect(0.125, 0.15, 0.025, 0.05, 1)
    scene.add_rect(0.2, 0.15, 0.025, 0.05, 2)
    scene.add_rect(0.225, 0.15, 0.025, 0.05, 3)
    scene.set_n_actuators(4)

def generate_robot_segments(scene, optimization_params, constant_params):
    num_segments = constant_params[0]
    connectivity = constant_params[5]
    connection_params = [optimization_params[-2], optimization_params[-1]]
    branching_factor = constant_params[4]

    branch_ID = constant_params[3]


    if branch_ID == 0:
        branch_wid = optimization_params[2]
        branch_height = optimization_params[3]
        branch_relative = optimization_params[4]
    else:
        branch_rad = optimization_params[2]
        branch_relative = optimization_params[3]

    rec_wid = optimization_params[0]
    rec_height = optimization_params[1]
    x = constant_params[1]
    y = constant_params[2]

    if connectivity == 0 or connectivity == 1:
        branch_connection = 1
    else:
        branch_connection = 0
    
    for i in range(num_segments):
        actuation = i % 4
        connector_wid = connection_params[0]
        connector_height = connection_params[1]
        scene.add_rect(x, y, rec_wid, rec_height, actuation)
        for j in range(branching_factor):
            branch_actuation = actuation
            side_check = j % 2 
            if branch_connection == 0:
                if branch_ID == 0:
                    scene.add_rect(x + side_check*rec_wid + branch_wid*(side_check - 1), y + rec_height*branch_relative - branch_height/2, branch_wid, branch_height, branch_actuation)
                else:
                    if side_check == 0:
                        branch_dir = 'left'
                    else:
                        branch_dir = 'right'
                    scene.add_circle(x + side_check*rec_wid, y + rec_height*branch_relative, branch_rad, branch_actuation, direction = branch_dir)
            else:
                if branch_ID == 0:
                    scene.add_rect(x + rec_wid*branch_relative - branch_wid/2, y + rec_height*(side_check) + branch_height*(side_check - 1), branch_wid, branch_height, branch_actuation)
                else:
                    if side_check == 0:
                        branch_dir = 'down'
                    else:
                        branch_dir = 'up'
                    scene.add_circle(x + rec_wid*branch_relative, y + rec_height*(side_check), branch_rad, branch_actuation, direction = branch_dir)

        if i != num_segments - 1:
            #connectors are type -1 (eg. no movement) by default
            if connectivity == 0:  # Moving left
                # Place connector at current position first
                scene.add_rect(x - connector_wid, y + rec_height/2 - connector_height/2, connector_wid, connector_height, -1)
                # Then update position for next segment
                x = x - (connector_wid + rec_wid)
            elif connectivity == 1:  # Moving right
                # Place connector at current position first
                scene.add_rect(x + rec_wid, y + rec_height/2 - connector_height/2, connector_wid, connector_height, -1)
                # Then update position for next segment
                x = x + rec_wid + connector_wid
            elif connectivity == 2:  # Moving up
                # Place connector at current position first
                scene.add_rect(x + rec_wid/2 - connector_wid/2, y + rec_height, connector_wid, connector_height, -1)
                # Then update position for next segment
                y = y + rec_height + connector_height

def robot(scene, optimization_params, constant_params):
    scene.set_offset(0.15, 0.1)
    generate_robot_segments(scene, optimization_params, constant_params)
    scene.set_n_actuators(constant_params[0])


gui = ti.GUI("Differentiable MPM", (640, 640), background_color=0xFFFFFF)


def visualize(s, folder, frame_counter=[0]):  # Using list for mutable counter
    aid = actuator_id.to_numpy()
    colors = np.empty(shape=n_particles, dtype=np.uint32)
    particles = x.to_numpy()[s]
    actuation_ = actuation.to_numpy()
    for i in range(n_particles):
        color = 0x111111
        if aid[i] != -1:
            act = actuation_[s - 1, int(aid[i])]
            color = ti.rgb_to_hex((0.5 - act, 0.5 - abs(act), 0.5 + act))
        colors[i] = color
    gui.circles(pos=particles, color=0x111111, radius=1.5)
    gui.line((0.05, 0.02), (0.95, 0.02), radius=3, color=0x0)

    os.makedirs(folder, exist_ok=True)
    frame_counter[0] += 1  # Increment counter
    gui.show(f'{folder}/{frame_counter[0]:03d}.png')  # Use counter for sequential naming

@ti.kernel
def clear_particle_states():
    # Clear particle state arrays
    for f, i in x:
        x[f, i] = [0.0, 0.0]
        v[f, i] = [0.0, 0.0]
        C[f, i] = [[0.0, 0.0], [0.0, 0.0]]
        F[f, i] = [[1.0, 0.0], [0.0, 1.0]]  # Initialize F to identity matrix

    # Clear particle IDs and types
    for i in range(n_particles):
        actuator_id[i] = -1  # Set to -1 as default (no actuation)
        particle_type[i] = 0  # Set to 0 as default (fluid particle)

    # Clear actuation values
    for t, i in actuation:
        actuation[t, i] = 0.0

    # Clear weights and biases
    for i, j in weights:
        weights[i, j] = 0.0
    
    # Clear bias separately
    for i in range(n_actuators):
        bias[i] = 0.0

    # Clear scalar fields
    loss[None] = 0.0
    x_avg[None] = [0.0, 0.0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=30)
    options = parser.parse_args()

    # Open log file in append mode
    with open('restart.txt', 'a') as log_file:
        log_file.write("\n" + "="*80 + "\n")
        log_file.write("NEW EVOLUTION RUN STARTED\n")
        log_file.write("="*80 + "\n\n")

    # Initial optimization parameters for geometry
    # optimization_params_init = np.array([0.05, 0.05, 0.03, 0.5, 0.03, 0.07])
    # Initial constant parameters
    # constant_params_init = np.array([2, 0, 0, 1, 1, 2])

    
    #after restart
    # optimization_params_init = np.array([0.05, 0.03863009, 0.03449276, 0.63662021, 0.03727828, 0.07])

    #after 1 generation:
    # optimization_params_init = np.array([0.05, 0.03863009, 0.03, 0.52765774, 0.04690867, 0.0542206])
    
    #after 8 generations:
    # optimization_params_init = np.array([0.05, 0.04311731, 0.03, 0.62489846, 0.03312806, 0.0542206])
    # constant_params_init = np.array([2, 0, 0, 1, 1, 2])

    #after 18 generations:
    optimization_params_init = np.array([0.03851529, 0.035, 0.0336775, 0.38426973, 0.04430574, 0.04278268])
    constant_params_init = np.array([2, 0, 0, 1, 1, 2])


    
    # Combine both parameter sets into a single evolution vector
    combined_params_init = np.concatenate([optimization_params_init, constant_params_init])
    
    # Define parameter types and constraints
    param_constraints = {
        'continuous': list(range(6)),  # First 6 params are continuous
        'discrete': list(range(6, 12)),  # Last 6 params are discrete
        'min_values': [0.035, 0.035, 0.03, 0.1, 0.025, 0.025, 2, 0, 0, 1, 0, 2],  # Minimum values
        'max_values': [None, None, None, None, None, None, 4, 0, 0, 1, 2, 2]    # No max for optimization params
    }

    N_combined = len(combined_params_init)
    num_generations = 10
    robos_per_gen = 3
    num_mutations = 6  # Increased to account for more parameters
    mutation_factor = 0.1
    MAX_PARTICLES = 555  # Maximum allowed particles

    # Initialize global variables
    global n_particles, n_actuators
    n_particles = 0
    n_actuators = 0

    # Create initial scene to set n_actuators and n_particles
    scene = Scene()
    robot(scene, optimization_params_init, constant_params_init.astype(int))
    scene.finalize()
    n_particles = scene.n_particles
    n_actuators = constant_params_init[0] + 2
    
    # Set MAX_PARTICLES based on initial robot configuration
    MAX_PARTICLES = n_particles
    print(f'Setting MAX_PARTICLES to initial robot particle count: {MAX_PARTICLES}')
    
    # Log the particle count
    with open('restart.txt', 'a') as log_file:
        log_file.write(f"Initial robot particle count (MAX_PARTICLES): {MAX_PARTICLES}\n\n")
    
    allocate_fields()

    # Initialize momentum tracking arrays
    prev_weights = np.zeros((n_actuators, n_sin_waves))
    prev_bias = np.zeros(n_actuators)

    def initialize_weights():
        # Initialize weights with smaller variance for more controlled movement
        for i in range(n_actuators):
            for j in range(n_sin_waves):
                weights[i, j] = np.random.randn() * 0.05  # Reduced initial scale
                prev_weights[i, j] = weights[i, j]
            bias[i] = np.random.randn() * 0.01
            prev_bias[i] = bias[i]

    def mutate(params):
        temp = params.copy()
        # Increase probability of mutation for more variation
        mutation_indices = random.sample(range(N_combined), random.randint(num_mutations, N_combined))
        
        for i in mutation_indices:
            if i in param_constraints['continuous']:
                # More aggressive continuous parameter mutation
                if random.random() < 0.3:  # 30% chance of large mutation
                    # Large mutation: completely new value within reasonable bounds
                    base_value = param_constraints['min_values'][i]
                    temp[i] = base_value * (1 + random.uniform(0.5, 5.0))
                else:
                    # Regular mutation with wider range
                    mutation_dir = random.choice([-1, 1])
                    mutation_size = random.uniform(mutation_factor, mutation_factor * 3)
                    new_val = temp[i] * (1 + (mutation_dir * mutation_size))
                    if new_val < param_constraints['min_values'][i]:
                        new_val = param_constraints['min_values'][i]
                    temp[i] = new_val
            else:
                # Discrete parameter mutation with more variation
                min_val = int(param_constraints['min_values'][i])
                max_val = int(param_constraints['max_values'][i])
                
                if i == 6:  # num_segments
                    # Allow bigger changes in number of segments
                    current = int(temp[i])
                    change = random.choice([-2, -1, 1, 2])
                    new_val = np.clip(current + change, min_val, max_val)
                elif i == 9:  # branch_ID
                    # Random choice between 0 and 1
                    new_val = random.randint(0, 1)
                elif i == 10:  # branch_factor
                    # Allow 1 or 2 branches
                    new_val = random.randint(0, 2)
                elif i == 11:  # connectivity
                    # Random connectivity type
                    new_val = 2
                else:
                    # Other discrete parameters: completely random within bounds
                    new_val = random.randint(min_val, max_val)
                
                temp[i] = new_val
        
        return temp

    def create_valid_robot(params):
        # Create a robot with the given parameters and check if it's valid
        scene = Scene()
        optimization_params = params[:6]
        constant_params = params[6:].astype(int)
        robot(scene, optimization_params, constant_params)
        scene.finalize()
        
        # Return None if invalid, otherwise return the params and scene
        if scene.n_particles <= MAX_PARTICLES:
            # Reallocate fields if number of particles has changed
            global n_particles, n_actuators
            if scene.n_particles != n_particles:
                n_particles = scene.n_particles
                n_actuators = constant_params[0]
                allocate_fields()
            return params, scene
        return None, None

    # Get initial valid robot configuration
    initial_params, initial_scene = create_valid_robot(combined_params_init)
    if initial_params is None:
        # If initial parameters create an invalid robot, keep trying with mutations until we get a valid one
        while True:
            params = mutate(combined_params_init)
            initial_params, initial_scene = create_valid_robot(params)
            if initial_params is not None:
                break

    combined_params_itr = initial_params
    all_losses = []

    for m in range(num_generations):
        with open('restart.txt', 'a') as log_file:
            log_file.write("\n" + "-"*40 + f" Generation {m} " + "-"*40 + "\n\n")
        
        optimization_params_tot = np.zeros((robos_per_gen, N_combined))
        optimization_params_tot[0] = combined_params_itr
        
        # Generate mutations
        for n in range(robos_per_gen - 1):
            # Keep mutating and trying until we get a valid robot
            while True:
                mutated_params = mutate(combined_params_itr)
                new_params, new_scene = create_valid_robot(mutated_params)
                if new_params is not None:
                    optimization_params_tot[n+1] = new_params
                    break

        robos_losses = []
        for p in range(robos_per_gen):
            combined_params = optimization_params_tot[p]
            optimization_params = combined_params[:6]
            constant_params = combined_params[6:].astype(int)
            
            # Log starting parameters
            with open('restart.txt', 'a') as log_file:
                log_file.write(f"Robot {p} Starting Parameters:\n")
                log_file.write(f"Optimization params: {optimization_params}\n")
                log_file.write(f"Configuration params: {constant_params}\n")
            
            print("Testing robot with params:", optimization_params)
            print("and configuration:", constant_params)
            
            # Clear all particle states before creating new robot
            clear_grid()
            clear_particle_grad()
            clear_actuation_grad()
            clear_particle_states()
            
            scene = Scene()
            robot(scene, optimization_params, constant_params)
            scene.finalize()

            # Initialize fresh weights and biases
            for i in range(n_actuators):
                for j in range(n_sin_waves):
                    weights[i, j] = np.random.randn() * 0.05  # Increased from 0.01
                bias[i] = np.random.randn() * 0.05  # Added random initialization for bias

            # Set up initial particle states
            for i in range(scene.n_particles):
                x[0, i] = scene.x[i]
                F[0, i] = [[1, 0], [0, 1]]
                actuator_id[i] = scene.actuator_id[i]
                particle_type[i] = scene.particle_type[i]

            losses = []
            best_loss = float('inf')
            plateau_count = 0
            
            # Dynamic learning rate parameters
            learning_rate = 0.1  # Reduced initial learning rate for more stable optimization
            min_learning_rate = 0.01
            lr_decay = 0.98  # Slower decay
            momentum_coeff = 0.8  # Reduced momentum coefficient
            
            initialize_weights()

            for iter in range(options.iters):
                clear_grid()
                clear_particle_grad()
                clear_actuation_grad()
                
                with ti.ad.Tape(loss):
                    forward()
                
                l = loss[None]
                losses.append(l)
                
                # More aggressive learning rate adjustment
                if l < best_loss:
                    best_loss = l
                    plateau_count = 0
                    learning_rate = min(learning_rate * 1.1, 0.2)  # Reward progress with higher learning rate
                else:
                    plateau_count += 1
                
                if plateau_count > 3:  # Reduced patience
                    learning_rate = max(learning_rate * lr_decay, min_learning_rate)
                    plateau_count = 0
                
                # Apply gradients with modified momentum
                for i in range(n_actuators):
                    for j in range(n_sin_waves):
                        if iter > 0:
                            momentum = momentum_coeff * (weights[i, j] - prev_weights[i, j])
                            grad_update = learning_rate * weights.grad[i, j]
                            weights[i, j] -= grad_update + momentum
                        else:
                            weights[i, j] -= learning_rate * weights.grad[i, j]
                        prev_weights[i, j] = weights[i, j]
                    
                    if iter > 0:
                        momentum = momentum_coeff * (bias[i] - prev_bias[i])
                        grad_update = learning_rate * bias.grad[i]
                        bias[i] -= grad_update + momentum
                    else:
                        bias[i] -= learning_rate * bias.grad[i]
                    prev_bias[i] = bias[i]
                
                print('i=', iter, 'loss=', l, 'lr=', learning_rate)

                # Visualization code remains the same
                if iter % 10 == 0:
                    clear_grid()
                    clear_particle_grad()
                    clear_actuation_grad()
                    forward(1500)
                    frame_counter = [0]
                    robot_dir = f'diffmpm/gen{m:02d}_robot{p:02d}/iter{iter:03d}/'
                    for s in range(15, 1500, 16):
                        visualize(s, robot_dir, frame_counter)

            robos_losses.append(losses[-1])
            if p == robos_per_gen-1:
                print('Generation ', m, 'losses: ', robos_losses)
                best_robo = np.argmin(robos_losses)
                best_combined_params = optimization_params_tot[best_robo]
                combined_params_itr = best_combined_params
                all_losses.append(robos_losses[best_robo])
                # Log the best robot of this generation
                with open('restart.txt', 'a') as log_file:
                    log_file.write(f"\nBest Robot of Generation {m}:\n")
                    log_file.write(f"Robot index: {best_robo}\n")
                    log_file.write(f"Best loss: {robos_losses[best_robo]}\n")
                    log_file.write(f"Best parameters: {best_combined_params}\n\n")
                print("Best combined parameters:", best_combined_params)
    
    all_losses = np.array(all_losses)
    print('BEST ROBOT PARAMS:', best_combined_params[:6])
    print('BEST ROBOT CONFIG:', best_combined_params[6:])
    print('ALL LOSSES:', all_losses)
    plt.title('Cost Optimization over Generations')
    plt.ylabel('Cost')
    plt.xlabel('Generations')
    plt.plot(range(num_generations), all_losses)
    plt.show()

    plt.title("Optimization of Initial Velocity")
    plt.ylabel("Loss")
    plt.xlabel("Gradient Descent Iterations")
    plt.plot(losses)
    plt.show()


if __name__ == '__main__':
    main()
