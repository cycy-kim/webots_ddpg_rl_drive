from controller import Robot, Supervisor
import numpy as np
import os
import json
import pickle
from DDPG import DDPG
from util import minmax_norm

supervisor = Supervisor()
pioneer = supervisor.getFromDef("Pioneer")

timestep = int(supervisor.getBasicTimeStep())
sensorTimeStep = int(supervisor.getBasicTimeStep())


sensorMax = 1200
maxSpeed = 6.4

motors = []
distance_sensors=[]

motor_names = [
    'front left wheel',
    'front right wheel',
    'back left wheel',
    'back right wheel',
]
distance_sensor_names=["so0", "so1", "so2", "so3", "so4", "so5", "so6", "so7", "so8", "so9", "so10", "so11", "so12", "so13", "so14", "so15"]

for motor_name in motor_names:
    motor = supervisor.getDevice(motor_name)
    motor.setPosition(float("inf"))
    motors.append(motor)
for distance_sensor_name in distance_sensor_names:
    distance_sensor = supervisor.getDevice(distance_sensor_name)
    distance_sensor.enable(sensorTimeStep)
    distance_sensors.append(distance_sensor)

translationField = pioneer.getField("translation")
rotationField = pioneer.getField("rotation")
initial_translation = translationField.getSFVec3f()
initial_rotation = [0,1,0,0]
initial_orientation = pioneer.getOrientation()

# 왼쪽, 오른쪽 바퀴 속도
motor_space = 2
distance_sensor_space = len(distance_sensor_names)

velocity_l = 0
velocity_r = velocity_l + motor_space
distance_l = velocity_r
distance_r = distance_l + distance_sensor_space

observation_space = motor_space + distance_sensor_space
# observation_space = distance_sensor_space
action_space = 2 # 각각 왼쪽바퀴 오른쪽바퀴 + or -

args = {
    'seed': 1234, 
    'hidden1': 300,
    'hidden2': 300,
    'init_w': 0.00003,          # 가중치 초기화 매개변수
    'prate': 0.0001,            # Actor 네트워크의 학습률 1e-4
    'rate': 0.001,              # Critic 네트워크의 학습률 1e-3
    'rmsize': 100000,           # 리플레이 버퍼 크기
    'window_length': 1,
    'load_sigma': False,        # True: 저장된 시그마 사용 / False: ou_sigma 사용
    'ou_theta': 0.15,           # Ornstein-Uhlenbeck 노이즈의 theta 값
    'ou_mu': 0,                 # Ornstein-Uhlenbeck 노이즈의 mu 값
    'ou_sigma': 0.15,           # Ornstein-Uhlenbeck 노이즈의 sigma 값
    'bsize': 128,
    'tau': 0.001,
    'discount': 0.99,
    'epsilon': 50000,           # epsilon-greedy 알고리즘 사용 시
    'epsilon_decay': 1.0,       # epsilon-greedy 알고리즘 사용 시
    'cur_episode': 0,           # 현재 episode
    'total_episode' : 1000000,  
    'mode' : 'train'            # 'test' or 'train' 
}
agent = DDPG(observation_space, action_space,args)
try:
    agent.load_weights('.')
    if agent.mode =='train':
        print("저장되어 있는 모델이 있습니다. 이어서 학습을 시작합니다.")
    elif agent.mode =='test':
        print("저장되어 있는 모델이 있습니다. test를 시작합니다.")
    else:
        print("올바른 모드를 입력해주세요. test / train")
except:
    if agent.mode =='test':
        print("저장되어 있는 모델이 없습니다. 학습을 먼저 진행해주세요.")
        exit()
    print("처음부터 학습을 시작합니다.")

def init_world():
    for motor in motors:
        motor.setPosition(float("inf"))
    
    for motor in motors:
        motor.setVelocity(0)

    for _ in range(int(1000 / timestep)):
        supervisor.step(timestep)
    translationField.setSFVec3f(initial_translation)
    rotationField.setSFRotation(initial_rotation)
    for motor in motors:
        motor.setVelocity(0)

    for _ in range(int(1000 / timestep)):
        supervisor.step(timestep)

    return

def get_state():
    cur_state = []

    motor_state = []
    distance_sensor_state = []

    motor_state.append(motors[0].getVelocity())
    motor_state.append(motors[1].getVelocity())
    for distance_sensor in distance_sensors:
        distance_sensor_state.append(distance_sensor.getValue())


    norm_motor_state = minmax_norm(motor_state, -maxSpeed, maxSpeed)
    norm_distance_sensor_state = minmax_norm(distance_sensor_state, 0, sensorMax)

    cur_state = norm_motor_state + norm_distance_sensor_state
    # cur_state = norm_distance_sensor_state
    return cur_state

def get_reward(state):
    reward = 0
    if max(state[distance_l:distance_r]) > 0.65:
        reward = -1
    # else:
    #     reward = 0.1
    elif state[0] > 0.7 and state[1] > 0.7:
        reward = 0.2
    elif state[0] > 0.1 and state[1] > 0.1:
        reward = 0.1
    else:
        reward = -0.1
    return reward

action_scale = 0.55
def set_action(actions):
    cur_leftmotor_value = motors[0].getVelocity()
    cur_rightmotor_value = motors[1].getVelocity()
    new_leftmotor_value = max(-1, min(maxSpeed , cur_leftmotor_value + actions[0]))
    new_rightmotor_value = max(-1, min(maxSpeed , cur_rightmotor_value + actions[1]))
    
    motors[0].setVelocity(new_leftmotor_value)
    motors[2].setVelocity(new_leftmotor_value)
    
    motors[1].setVelocity(new_rightmotor_value)
    motors[3].setVelocity(new_rightmotor_value)

    # motor.setPosition하는데 걸리는 시간
    for _ in range(int(400 / timestep)):
        supervisor.step(timestep)
    
def is_done(state):
    if max(state[distance_l:distance_r]) > 0.65:
        return True
    
    return False

# train mode
step = 0
if args['mode'] == 'train':
    for episode in range(agent.cur_episode, agent.total_episode):
        init_world()
        total_reward = 0
        step = 0

        # while supervisor.step(timestep) != -1:
        while True:
            ###
            state = get_state()
            action = agent.select_action(state)
            set_action(action)
            ###
            next_state = get_state()
            
            
            reward = get_reward(next_state)

            total_reward += reward
            done = is_done(next_state)
            agent.replay_buffer.append(state, action, reward, next_state, done)

            step += 1
            if done:
                agent.update_policy()
                break
          

        # agent.actor.ou_noise.decrease_sigma()
        print("---episode finished---")
        print("episode : " + str(episode))
        print("total_reward : " + str(total_reward))
        print("sigma : " + str(agent.random_process.sigma))
        agent.cur_episode += 1
        if episode % 100 == 0:
        # 저장하고 리셋
            agent.save_model('.')
            pioneer.resetPhysics()


# test mode
if args['mode'] == 'test':
    while True:
        init_world()
        total_reward = 0
        while supervisor.step(timestep) != -1:
            state = get_state()
            action = agent.select_action(state, decay_epsilon=False, noise=False)
            print(action)
            set_action(action)
            next_state = get_state()
            reward = get_reward(next_state)
            total_reward += reward

            done = is_done(next_state)

            if done:
                break

        print("---episode finished---")
        print("total_reward : " + str(total_reward))