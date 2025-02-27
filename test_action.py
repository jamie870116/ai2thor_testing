import copy
import glob
import json
import os
import argparse
from pathlib import Path
from datetime import datetime
import random
import subprocess
import ai2thor.controller
import tqdm
import numpy as np
from scipy.spatial import distance
import re

from openai import OpenAI
llama_client = OpenAI(api_key=Path('llama_api_key.txt').read_text(), base_url="https://api.llama-api.com")
gpt_client = OpenAI(api_key=Path('gpt_api_key.txt').read_text())
import sys
sys.path.append(".")

import resources.actions as actions
import resources.robots as robots

# 全域變數
action_queue = []
# c 會在 initialize_env 之後重新指派
c = ai2thor.controller.Controller()
reachable_positions = []

def distance_pts(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[2] - p2[2]) ** 2) ** 0.5

def closest_node(node, nodes, no_robot, clost_node_location):
    distances = distance.cdist([node], nodes)[0]
    dist_indices = np.argsort(np.array(distances))
    return [nodes[dist_indices[i * 5 + clost_node_location[i]]] for i in range(no_robot)]

def execute_actions():
    while action_queue:
        action = action_queue.pop(0)
        c.step(action)

def GoToObject(robot, dest_obj):
    """讓機器人移動到指定物件位置"""
    agent_id = int(robot['name'][-1]) - 1
    objs = [obj for obj in c.last_event.metadata['objects']]
    for obj in objs:
        if re.match(dest_obj, obj['objectId']):
            dest_obj_id, dest_obj_center = obj['objectId'], obj['axisAlignedBoundingBox']['center']
            break
    dest_obj_pos = [dest_obj_center['x'], dest_obj_center['y'], dest_obj_center['z']]
    crp = closest_node(dest_obj_pos, reachable_positions, 1, [0])
    action_queue.append({'action': 'MoveAhead', 'position': {'x': crp[0][0], 'y': crp[0][1], 'z': crp[0][2]}, 'agentId': agent_id})
    execute_actions()

def PickupObject(robot, pick_obj):
    """讓機器人撿起指定物件"""
    agent_id = int(robot['name'][-1]) - 1
    for obj in c.last_event.metadata['objects']:
        if re.match(pick_obj, obj['objectId']):
            pick_obj_id = obj['objectId']
            break
    action_queue.append({'action': 'PickupObject', 'objectId': pick_obj_id, 'agentId': agent_id})
    execute_actions()

def PutObject(robot, recp):
    """讓機器人放下物件到指定容器"""
    agent_id = int(robot['name'][-1]) - 1
    action_queue.append({'action': 'PutObject', 'objectId': recp, 'agentId': agent_id})
    execute_actions()

def SwitchOn(robot, obj):
    """開啟指定裝置"""
    agent_id = int(robot['name'][-1]) - 1
    action_queue.append({'action': 'ToggleObjectOn', 'objectId': obj, 'agentId': agent_id})
    execute_actions()

def SwitchOff(robot, obj):
    """關閉指定裝置"""
    agent_id = int(robot['name'][-1]) - 1
    action_queue.append({'action': 'ToggleObjectOff', 'objectId': obj, 'agentId': agent_id})
    execute_actions()

def OpenObject(robot, obj):
    """打開指定物件"""
    agent_id = int(robot['name'][-1]) - 1
    action_queue.append({'action': 'OpenObject', 'objectId': obj, 'agentId': agent_id})
    execute_actions()

def CloseObject(robot, obj):
    """關閉指定物件"""
    agent_id = int(robot['name'][-1]) - 1
    action_queue.append({'action': 'CloseObject', 'objectId': obj, 'agentId': agent_id})
    execute_actions()

def BreakObject(robot, obj):
    """破壞指定物件"""
    agent_id = int(robot['name'][-1]) - 1
    action_queue.append({'action': 'BreakObject', 'objectId': obj, 'agentId': agent_id})
    execute_actions()

def SliceObject(robot, obj):
    """切割指定物件"""
    agent_id = int(robot['name'][-1]) - 1
    action_queue.append({'action': 'SliceObject', 'objectId': obj, 'agentId': agent_id})
    execute_actions()

def ThrowObject(robot, obj):
    """投擲指定物件"""
    agent_id = int(robot['name'][-1]) - 1
    action_queue.append({'action': 'ThrowObject', 'objectId': obj, 'agentId': agent_id})
    execute_actions()

def CleanObject(robot, obj):
    """清潔指定物件"""
    agent_id = int(robot['name'][-1]) - 1
    action_queue.append({'action': 'CleanObject', 'objectId': obj, 'agentId': agent_id})
    execute_actions()

def get_args():
    '''
    Get the arguments from the command line
    --floor-plan: int: the floor plan number
    --llama-version: str: the version of the llama model
    --prompt-decompse-set: str:file name of the set of prompts for task decomposition
    --prompt-allocation-set: str:file name of the set of prompts for task allocation
    --test-set: str:file name of the set of test tasks
    --log-results: boolean whether to log the results
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--floor-plan", type=int, required=True)
    parser.add_argument("--model", default="gpt", choices=['llama', 'gpt'])
    parser.add_argument("--gpt-version", default="gpt-4", choices=['gpt-4', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k'])
    parser.add_argument("--llama-version", type=str, default="llama3.3-70b", 
                        choices=['llama3.1-405b', 'llama3.3-70b', 'llama3.1-70b', 'llama3.1-8b', 'deepseek-r1', 'deepseek-v3', 'mixtral-8x22b-instruct', 'gemma2-27b'])
    
    parser.add_argument("--prompt-decompse-set", type=str, default="train_task_decompose", 
                        choices=['train_task_decompose', 'train_task_decompose_llama'])
    
    parser.add_argument("--prompt-allocation-set", type=str, default="train_task_allocation", 
                        choices=['train_task_allocation', 'train_task_allocation_llama'])
    
    parser.add_argument("--test-set", type=str, default="test", 
                        choices=['final_test', 'test'])
    
    parser.add_argument("--log-results", type=bool, default=True)
    
    args = parser.parse_args()
    return args

def initialize_env(floor_no, agent_count=2, width=1024, height=1024, grid_size=0.25):
    '''
    Initialize the positions of the agents randomly in the scene
    '''
    c = ai2thor.controller.Controller(height=height, width=width)
    c.reset("FloorPlan" + str(floor_no)) 

    # initialize n agents into the scene
    multi_agent_event = c.step(dict(action='Initialize', agentMode="default", snapGrid=False, gridSize=grid_size, rotateStepDegrees=20, visibilityDistance=100, fieldOfView=90, agentCount=agent_count))

    # add a top view camera
    event = c.step(action="GetMapViewCameraProperties")
    event = c.step(action="AddThirdPartyCamera", **event.metadata["actionReturn"])

    # get reachabel positions
    reachable_positions_ = c.step(action="GetReachablePositions").metadata["actionReturn"]
    reachable_positions = [(p["x"], p["y"], p["z"]) for p in reachable_positions_]

    # randomize postions of the agents
    agent_positions = []
    for i in range(agent_count):
        init_pos = random.choice(reachable_positions_)
        c.step(dict(action="Teleport", position=init_pos, agentId=i))
        agent_positions.append(init_pos)
    return c, reachable_positions, agent_positions

def LM(prompt, model, version, max_tokens=128, temperature=0, stop=None, logprobs=1, frequency_penalty=0, isAllocate=False):
    '''
    Use the LLM to generate a response to the prompt
    args:
        prompt: str: the prompt to generate a response to
        model: str: the model to use: llama or gpt
        version: str: the version of the llama model
        max_tokens: int: the maximum number of tokens to generate
        temperature: float: the temperature of the response
        stop: str: the stop sequence
        logprobs: int: the log probabilities of the response
        frequency_penalty: float: the frequency penalty of the response
        isAllocate: boolean: whether the response is for task allocation
    returns:
        response: str: the full response from the LLM
        content: str: the content of the response
    '''
    if model == 'llama':
        response = llama_client.chat.completions.create(model=version, 
                                                messages=prompt, 
                                                max_tokens=max_tokens, 
                                                temperature=temperature, 
                                                frequency_penalty = frequency_penalty,
                                                stream=False)
        content = response.choices[0].message.content.strip()

        if not isAllocate and '```python' in content:
            content = content.split('```python')[1].split('```')[0]
    elif model == 'gpt':
        response = gpt_client.chat.completions.create(model=version, 
                                                prompt=prompt, 
                                                max_tokens=max_tokens, 
                                                temperature=temperature, 
                                                stop=stop, 
                                                logprobs=logprobs, 
                                                frequency_penalty = frequency_penalty,
                                                stream=False)
        content = response.choices[0].message.content.strip()
    return response, content.strip()

def get_task_from_file(test_set, floor_plan):
    # read the tasks        
    task = {}
    # task = {
    #     "task": "Wash the lettuce and place lettuce on the Countertop",
    #     "robot list": [1, 2],
    #     "object_states": [{"name": "CounterTop", "contains": ["Lettuce"], "state": "None"}],
    #     "trans": 0,
    #     "max_trans": 0,
    #     "coverages": ["Lettuce", "Sink", "CounterTop"],
    #     "interact_objs": ["Lettuce"],
    #     "interact_recep": ["Sink", "CounterTop"]
    # }  
    with open (f"./data/{test_set}/FloorPlan{floor_plan}.json", "r") as f:
        for line in f.readlines():
            task["task"] = list(json.loads(line).values())[0]
            task["robot_list"] = list(json.loads(line).values())[1]
            task["object_states"] = list(json.loads(line).values())[2]
            task["trans"] = list(json.loads(line).values())[3]
            task["max_trans"] = list(json.loads(line).values())[4]
            task["coverages"] = list(json.loads(line).values())[5]
            task["interact_objs"] = list(json.loads(line).values())[6]
            task["interact_recep"] = list(json.loads(line).values())[7]
    return task

def get_available_robots(robots_test_tasks):
    available_robots = []
    for robots_list in robots_test_tasks:
        task_robots = []
        for i, r_id in enumerate(robots_list):
            rob = robots.robots[r_id-1]
            # rename the robot
            rob['name'] = 'robot' + str(i+1)
            task_robots.append(rob)
        available_robots.append(task_robots)
    return available_robots

# Function returns object list with name and properties.
def convert_to_dict_objprop(objs, obj_mass, obj_position):
    objs_dict = []
    for i, obj in enumerate(objs):
        obj_dict = {'name': obj , 'mass' : obj_mass[i], 'position' : obj_position[i]}
        objs_dict.append(obj_dict)
    return objs_dict

def get_ai2_thor_objects(floor_plan_id):
    # connector to ai2thor to get object list
    controller = ai2thor.controller.Controller(scene="FloorPlan"+str(floor_plan_id))
    obj = list([obj["objectType"] for obj in controller.last_event.metadata["objects"]])
    obj_mass = list([obj["mass"] for obj in controller.last_event.metadata["objects"]])
    obj_position = list([obj["position"] for obj in controller.last_event.metadata["objects"]])
    controller.stop()
    obj = convert_to_dict_objprop(obj, obj_mass, obj_position)
    return obj

def get_decompose_prompt(scene, ai2thor_objects, task, num_robots=None, open_subtasks=None, completed_subtasks=None):
    """
    Get the prompt for task decomposition
    """
    prompt = f"""
        You are an excellent task planner whose task is to help {num_robots} robots to complete the task of {task} in {scene}:
        In current step, you will have the information of what task needs to be done, what objects are available in the scene and what skills each robot has.
        And you will need to plan the task for the robots to complete the task.
        You are given a high-level task in {scene} and a list of available robots.
        The following is the skills that a robot can perform, which is shown in the format of a list containing 
        "skill_name <robot_number><object_name>" or "skill_name <robot_number><object_name><receptacle_object_name>":
        ### {actions.ai2thor_actions} ###
        The following is the list of objects in {scene} that you have access to:
        ### {ai2thor_objects} ###
        So, along with the object list and the skills that each robot has, you will get the following information as the input in the following format:
        ### INPUT INFORMATION ###
        {{"task": {task} - a string decribing the task in high level,
         "Robots open subtasks": {open_subtasks} - list of subtasks the robots are supposed to carry out to finish the task. If no plan has been already created, this will be None.,
         "completed subtasks": {completed_subtasks} -list of subtasks that have been completed. If no subtasks have been completed, this will be None}}
    
        Reason over the task, objects, skills, Robots open subtasks, and completed subtasks, and then output the following:
        * Reason: The reason for why new subtasks need to be added.
        * Subtasks: A list of open subtasks the robots are supposed to take to complete the task. Remember, as you get new information about the environment, 
        you can modify this list. You can keep the same plan if you think it is still valid. Do not include the subtasks that have already been completed.
        The "Plan" should be in a list format where the actions are listed sequentially.
        For example:
            ["locate the apple", "transport the apple to the fridge", "transport the book to the table"]
            ["locate the cup", "go to cup", "clean cup"]
        When possible do not perform additional steps when one is sufficient (e.g. CleanObject is sufficient to clean an object, no other actions need to be done)
        Your output should be in the form of a python dictionary as shown below. The "reason" should be a string and the "plan" should be a list of dictionaries (each dictionary's key is the action and the value is the list of objects involved in the action).
        Example output: {{"reason": "since no subtasks have been planned or completed, the robot needs to wash the lettuce and place it on the countertop.", 
        ,"plan": ["transport the lettece to the sink", "wash the lettece", "transport the lettece to the countertop"]}}
    
        Ensure that the subtasks are not generic statements like "explore the environment" or "do the task". They should be specific to the task at hand.
        Do not assign subtasks to any particular robot. Try not to modify the subtasks that already exist in the open subtasks list. Rather add new subtasks to the list.
        Since there are {num_robots} agents moving around in one room, make sure to plan subtasks and actions so that they don't block or bump into each other as much as possible. 
        This is especially important because there are so many agents in one room.
    
        * NOTE: DO NOT OUTPUT ANYTHING EXTRA OTHER THAN WHAT HAS BEEN SPECIFIED
        Let's work this out in a step by step way to be sure we have the right answer.
        """
    return prompt

def get_allocate_prompt(scene, ai2thor_objects, task, num_robots, reachable_positions, initial_positions, open_subtasks, completed_subtasks=None):
    """
    Get the prompt for task allocation
    """ 
    allocate_prompt = f"""
        You are an excellent task planner whose task is to help {num_robots} robots to complete the final task of {task} in {scene}:
        In current step, you will have the information of what task needs to be done, what objects are available in the scene (including the position of the objects)
        and what skills each robot has, and the reachable position of robots, and the initial position of eachrobot.
    
        And now, you will need to allocate the subtasks to the robots, and you will need to plan the task for the robots to complete the task.
    
        You are given a high-level task in {scene} and a list of available robots.
        The following is the skills that a robot can perform, which is shown in the format of a list containing 
        "skill_name <robot_number><object_name>" or "skill_name <robot_number><object_name><receptacle_object_name>":
        ### {actions.ai2thor_actions} ###
        The following is the list of objects in {scene} that you have access to, this includes the name, mass, and the position of the object:
        ### {ai2thor_objects} ###
        The following is the reachable position of each robot, which means the position that the robot can move to:
        ### {reachable_positions} ###
        The following is the initial position of each robot:
        ### {initial_positions} ###
        So, along with the above information, you will also get the following information as the input in the following format:
        ### INPUT INFORMATION ###
        {{
         "Robots open subtasks": {open_subtasks} - list of subtasks the robots are supposed to carry out to finish the task. If no plan has been already created, this will be None.,
         "completed subtasks": {completed_subtasks} -list of subtasks that have been completed. If no subtasks have been completed, this will be None}}
        and this is an example of the input:
        {{
         "Robots open subtasks": [
             "transport the spatula to the garbage can",
             "wash the lettuce",
         ]
         "completed subtasks": []
        }}
    
        You should reason over the information above, and allocate the subtasks(which is the *Robots open subtasks*) to the suitable robots and plan how the subtask, and then output the following:
        Also you should consider the reachable positions of the robots and the initial positions of the robots, and the skills of the robots, to best allocate the subtasks to the right robots. 
        It is usually best to have the closet robot to complete the subtask.
        * Subtasks: A list of open subtasks the robots are supposed to take to complete the task. Remember, as you get new information about the environment, 
        you can modify this list. You can keep the same plan if you think it is still valid. Do not include the subtasks that have already been completed.
        The "Plan" should be in a list format where the actions are listed sequentially.
        For example:
            ["locate the apple", "transport the apple to the fridge", "transport the book to the table"]
            ["locate the cup", "go to cup", "clean cup"]
        When possible do not perform additional steps when one is sufficient (e.g. CleanObject is sufficient to clean an object, no other actions need to be done)
        Your output should be in the form of a python dictionary as shown below. The "reason" should be a string and the "plan" should be a list of dictionaries (each dictionary's key is the action and the value is the list of objects involved in the action).
        The "plan" should be in the format of "robot_number, [action1, action2, action3, ...]", the action should be the action that the robot can do and along with the receptacle object if it is needed. Make sure the object is in the given scene. No need to include which robot is doing the action in this part.
        
        Example output: {{
         "transport the spatula to the garbage can": ["robot 1", 
             [
                 "GoToObject Spatula", 
                 "PickupObject Spatula", 
                 "GoToObject GarbageCan", 
                 "ThrowObject Spatula"
             ]
         }}
        * NOTE: DO NOT OUTPUT ANYTHING EXTRA OTHER THAN WHAT HAS BEEN SPECIFIED
        Let's work this out in a step by step way to be sure we have the right answer.
        """
    return allocate_prompt

# ===== 新增：環境狀態讀取與任務檢查函數 =====
def get_environment_state():
    """
    從最新事件中提取環境中所有物件的資訊，並以字典形式返回。
    每個物件包含 position, isPickedUp, isOpen, isToggled 等資訊，
    以及 receptacle 中的物件（此處假設存在 "receptacleObjectIds" 欄位）。
    """
    state = {}
    objects = c.last_event.metadata["objects"]
    for obj in objects:
        state[obj["objectId"]] = {
            "position": obj["position"],
            "isPickedUp": obj.get("isPickedUp", False),
            "isOpen": obj.get("isOpen", False),
            "isToggled": obj.get("isToggled", False),
            "contains": obj.get("receptacleObjectIds", [])
        }
    return state

def check_task_progress(task, env_state):
    """
    根據 task 中定義的 coverages 與 object_states 檢查當前環境是否達到目標。

    task: dict, 例如：
        {
            "task": "Wash the lettuce and place lettuce on the Countertop",
            "robot list": [1,2],
            "object_states": [{"name": "CounterTop", "contains": ["Lettuce"], "state": "None"}],
            "trans": 0,
            "max_trans": 0,
            "coverages": ["Lettuce", "Sink", "CounterTop"],
            "interact_objs": ["Lettuce"],
            "interact_recep": ["Sink", "CounterTop"]
        }
    env_state: 由 get_environment_state() 獲得的字典

    返回:
        {
          "coverage_success": bool,
          "object_states_success": bool
        }
    """
    # 檢查 coverages：所有指定的物件名稱必須至少出現在某個 objectId 中
    coverage_success = True
    for cov_obj in task.get("coverages", []):
        found = any(cov_obj.lower() in obj_id.lower() for obj_id in env_state.keys())
        if not found:
            coverage_success = False
            break

    # 檢查 object_states：例如要求 CounterTop 包含 Lettuce
    object_states_success = True
    for state_req in task.get("object_states", []):
        target_name = state_req.get("name")
        target_contains = state_req.get("contains", [])
        target_state = state_req.get("state", "None")
        # 找出物件名稱中包含 target_name 的物件
        found_obj = None
        for obj_id, obj_info in env_state.items():
            if target_name.lower() in obj_id.lower():
                found_obj = obj_info
                break
        if not found_obj:
            object_states_success = False
            break
        # 檢查 contains 要求
        if target_contains:
            contains_list = [item.lower() for item in found_obj.get("contains", [])]
            if not all(item.lower() in contains_list for item in target_contains):
                object_states_success = False
                break
        # 檢查其他狀態（例如 "open"），此處 target_state 為 "None" 則不檢查
        if target_state != "None":
            if target_state.lower() == "open" and not found_obj.get("isOpen", False):
                object_states_success = False
                break
            # 可根據需求添加其他狀態判斷
    return {"coverage_success": coverage_success, "object_states_success": object_states_success}

# ===== 主程式 =====
def main():
    args = get_args()
    print(args)

    print('Getting task from file...')
    task = get_task_from_file(args.test_set, args.floor_plan)
    # task = {
    #     "task": "Wash the lettuce and place lettuce on the Countertop",
    #     "robot list": [1, 2],
    #     "object_states": [{"name": "CounterTop", "contains": ["Lettuce"], "state": "None"}],
    #     "trans": 0,
    #     "max_trans": 0,
    #     "coverages": ["Lettuce", "Sink", "CounterTop"],
    #     "interact_objs": ["Lettuce"],
    #     "interact_recep": ["Sink", "CounterTop"]
    # }
    print(f"\n----Test tasks----\n{task}\n")

    print('Getting available robots...')
    available_robots = get_available_robots(task['robot list'])
    print('Getting ai2thor objects...')
    ai2thor_objects = get_ai2_thor_objects(args.floor_plan)

    # 根據 floor-plan 決定場景名稱
    scene = ''
    if args.floor_plan < 30:
        scene = "kitchen"
    elif args.floor_plan < 230:
        scene = "living room"
    elif args.floor_plan < 330:
        scene = "bedroom"
    else:
        scene = "bathroom"

    # 初始化環境
    width = 1024
    height = 1024
    grid_size = 0.25
    controller, reachable_positions_local, agent_positions = initialize_env(args.floor_plan, len(available_robots), width, height, grid_size)
    global c, reachable_positions
    c = controller
    reachable_positions = reachable_positions_local

    objs = [obj["objectId"] for obj in controller.last_event.metadata["objects"]]

    # 加入 top view camera
    event = controller.step(action="GetMapViewCameraProperties")
    event = controller.step(action="AddThirdPartyCamera", **event.metadata["actionReturn"])

    # 初始化一般 prompt（保持不變）
    general_prompt = f"""
    You are given a high-level task in {scene} and a list of available robots.
    The following is the skills that a robot can perform, which is shown in the format of a list containing 
    "skill_name <robot_number><object_name>" or "skill_name <robot_number><object_name><receptacle_object_name>":
    ### {actions.ai2thor_actions} ###
    The following is the list of objects in {scene} that you have access to:
    ### {ai2thor_objects} ###
    """

    # 假設我們的 task 如下（或從檔案中讀取）
    task = {
        "task": "Wash the lettuce and place lettuce on the Countertop",
        "robot list": [1, 2],
        "object_states": [{"name": "CounterTop", "contains": ["Lettuce"], "state": "None"}],
        "trans": 0,
        "max_trans": 0,
        "coverages": ["Lettuce", "Sink", "CounterTop"],
        "interact_objs": ["Lettuce"],
        "interact_recep": ["Sink", "CounterTop"]
    }
    print("Initial Task:", task)

    # 假設初始子任務規劃（保持原本設定的 output 格式不變）
    open_subtasks = [
        "pickup Lettuce",
        "clean Lettuce",
        "put Lettuce on CounterTop"
    ]
    completed_subtasks = []

    # 生成任務分解 prompt（保持不變）
    decompose_prompt = general_prompt + get_decompose_prompt(scene, ai2thor_objects, task["task"], len(available_robots))
    print("Decompose Prompt:\n", decompose_prompt)
    # 此處可呼叫 LM() 以取得初始計劃，此處略過

    # 主迴圈：執行動作並檢查環境狀態與任務進度
    task_finished = False
    iteration = 0
    max_iterations = 50  # 防止無限迴圈
    while not task_finished and iteration < max_iterations:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")
        
        # 根據 open_subtasks 分配動作給機器人（保持原本 output 格式）
        if open_subtasks:
            current_action = open_subtasks.pop(0)
            print("Executing action:", current_action)
            if "pickup" in current_action.lower():
                PickupObject(available_robots[0][0], "Lettuce")
            elif "put" in current_action.lower():
                PutObject(available_robots[0][0], "CounterTop")
            elif "clean" in current_action.lower():
                CleanObject(available_robots[0][0], "Lettuce")
        
        # 執行動作後，更新環境狀態並檢查進度
        current_env_state = get_environment_state()
        progress = check_task_progress(task, current_env_state)
        print("Task progress:", progress)
        
        if progress["object_states_success"]:
            print("Task goal achieved!")
            task_finished = True
        else:
            print("Task goal not achieved yet.")
        
        # 若 coverages 完成，則將其加入 completed_subtasks（僅作參考）
        if progress["coverage_success"] and "Coverage complete" not in completed_subtasks:
            completed_subtasks.append("Coverage complete")
        
        # 若 open_subtasks 為空但任務未完成，可重新生成計劃（此處僅示範打印）
        if not open_subtasks and not task_finished:
            print("Re-planning task based on current state...")
            # 可呼叫 LM() 與 get_decompose_prompt() 產生新計劃，此處略過

    print("Final Task State:", task)
    print("Completed Subtasks:", completed_subtasks)
    print("Iteration count:", iteration)

if __name__ == "__main__":
    main()
