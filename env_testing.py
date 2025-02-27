import math
import re
import shutil
import subprocess
import time
import threading
import cv2
import numpy as np
from ai2thor.controller import Controller
from scipy.spatial import distance
from typing import Tuple
import random
import os
from glob import glob
import json
import datetime
import heapq
from pathlib import Path

from openai import OpenAI
llama_client = OpenAI(api_key=Path('llama_api_key.txt').read_text(), base_url="https://api.llama-api.com")
gpt_client = OpenAI(api_key=Path('gpt_api_key.txt').read_text())
import sys
sys.path.append(".")

import resources.actions as actions
import resources.robots as robots

# ---------------------------
# Helper：計算固定初始位置（所有機器人同在一角且排列成一列）
# ---------------------------
def get_same_corner_positions(num_agents, reachable_positions, spacing=1):
    xs = [p[0] for p in reachable_positions]
    zs = [p[2] for p in reachable_positions]
    min_x, max_x = min(xs), max(xs)
    min_z, max_z = min(zs), max(zs)
    default_y = reachable_positions[0][1]
    corners = [
        (min_x, default_y, min_z),  # 左下
        (max_x, default_y, min_z),  # 右下
        (min_x, default_y, max_z),  # 左上
        (max_x, default_y, max_z)   # 右上
    ]
    chosen_corner = random.choice(corners)
    positions = []
    if chosen_corner[0] == min_x:
        for i in range(num_agents):
            positions.append((min_x + i * spacing, default_y, chosen_corner[2]))
    else:
        for i in range(num_agents):
            positions.append((max_x - i * spacing, default_y, chosen_corner[2]))
    return chosen_corner, positions

# ---------------------------
# Helper：Graph 與 A* 搜尋
# ---------------------------
def build_graph(reachable_positions, edge_threshold=0.5):
    graph = {}
    for pos in reachable_positions:
        key = tuple(pos)
        graph[key] = []
    for pos in reachable_positions:
        for pos2 in reachable_positions:
            if np.array_equal(pos, pos2):
                continue
            if np.linalg.norm(np.array(pos) - np.array(pos2)) < edge_threshold:
                graph[tuple(pos)].append(tuple(pos2))
    return graph

def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def astar_search(graph, start, goal):
    queue = []
    heapq.heappush(queue, (0, start))
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while queue:
        current_priority, current = heapq.heappop(queue)
        if current == goal:
            break
        for neighbor in graph[current]:
            new_cost = cost_so_far[current] + np.linalg.norm(np.array(current) - np.array(neighbor))
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal)
                heapq.heappush(queue, (priority, neighbor))
                came_from[neighbor] = current

    if goal not in came_from:
        return []  # 找不到路徑
    current = goal
    path = []
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path

def distance_pts(p1: Tuple[float, float, float], p2: Tuple[float, float, float]):
    return ((p1[0]-p2[0])**2 + (p1[2]-p2[2])**2)**0.5

def generate_video():
    frame_rate = 5
    cur_path = os.path.join(os.path.dirname(__file__), "*")
    for imgs_folder in glob(cur_path, recursive=False):
        view = os.path.basename(os.path.normpath(imgs_folder))
        if not os.path.isdir(imgs_folder):
            print("The input path: {} you specified does not exist.".format(imgs_folder))
        else:
            command_set = ['ffmpeg', '-y', '-i',
                           os.path.join(imgs_folder, "img_%05d.png"),
                           '-framerate', str(frame_rate),
                           '-pix_fmt', 'yuv420p',
                           os.path.join(os.path.dirname(__file__), f"video_{view}.mp4")]
            subprocess.call(command_set)

def get_task_from_file(test_set, floor_plan):
    task = {}
    with open(f"./data/{test_set}/FloorPlan{floor_plan}.json", "r") as f:
        for line in f.readlines():
            data = json.loads(line)
            values = list(data.values())
            task["task"] = values[0]
            task["robot_list"] = values[1]
            task["object_states"] = values[2]
            task["trans"] = values[3]
            task["max_trans"] = values[4]
            task["coverages"] = values[5]
            task["interact_objs"] = values[6]
            task["interact_recep"] = values[7]
            task["open_subtasks"] = data.get("open_subtasks", [])
    return task

# ---------------------------
# Helper：LLM 呼叫函式
# ---------------------------
def LM(prompt, model='gpt', version='gpt-4', max_tokens=128, temperature=0, stop=None, logprobs=1, frequency_penalty=0, isAllocate=False):
    if model == 'llama':
        response = llama_client.chat.completions.create(model=version, 
                                                        messages=prompt, 
                                                        max_tokens=max_tokens, 
                                                        temperature=temperature, 
                                                        frequency_penalty=frequency_penalty,
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
                                                      frequency_penalty=frequency_penalty,
                                                      stream=False)
        content = response.choices[0].message.content.strip()
    return response, content.strip()

# ---------------------------
# AI2ThorEnvironment 定義
# ---------------------------
class AI2ThorEnvironment:
    def __init__(self, floorplan: str, num_agents: int, task: dict):
        self.controller = Controller(width=1024, height=1024)
        self.floorplan = floorplan
        self.num_agents = num_agents
        self.reachable_positions = []
        self.agent_positions = []
        self.top_view_event = None
        self.task_info = task
        self.controller.reset(self.floorplan)
        print("reset environment")
        multi_agent_event = self.controller.step(dict(
            action='Initialize',
            agentMode="default",
            snapGrid=False,
            gridSize=0.25,
            rotateStepDegrees=20,
            visibilityDistance=100,
            fieldOfView=90,
            agentCount=num_agents))
        self.top_view_event = self.controller.step(action="GetMapViewCameraProperties")
        self.top_view_event = self.controller.step(action="AddThirdPartyCamera", **self.top_view_event.metadata["actionReturn"])
        reachable_positions_ = self.controller.step(action="GetReachablePositions").metadata["actionReturn"]
        self.reachable_positions = [(p["x"], p["y"], p["z"]) for p in reachable_positions_]
        # 固定初始位置：選一個隨機角落，所有機器人均位於該角落並排成一列
        chosen_corner, fixed_positions = get_same_corner_positions(self.num_agents, self.reachable_positions, spacing=0.5)
        xs = [p[0] for p in self.reachable_positions]
        zs = [p[2] for p in self.reachable_positions]
        center = ((min(xs)+max(xs))/2, self.reachable_positions[0][1], (min(zs)+max(zs))/2)
        for i, pos in enumerate(fixed_positions):
            delta_x = center[0] - pos[0]
            delta_z = center[2] - pos[2]
            angle = math.degrees(math.atan2(delta_x, delta_z)) % 360
            init_pos = {"x": pos[0], "y": pos[1], "z": pos[2]}
            init_rot = {"y": angle}
            self.controller.step(dict(action="Teleport", position=init_pos, rotation=init_rot, agentId=i))
            self.agent_positions.append((pos[0], pos[1], pos[2]))
            print(f"Agent {i} positioned at {init_pos} facing {angle:.2f}°")

    def get_scene_info(self, isOjectOnly=False):
        print("getting scene info")
        objects = self.controller.last_event.metadata["objects"]
        
        if not isOjectOnly:
            state = {}
            for obj in objects:
                state[obj['name']] = {
                    "objectId": obj["objectId"],
                    "objectType": obj["objectType"],
                    "position": obj["position"],
                    "isPickedUp": obj.get("isPickedUp", False),
                    "isOpen": obj.get("isOpen", False),
                    "isSliced": obj.get("isSliced", False),
                    "isToggled": obj.get("isToggled", False),
                    "isBroken": obj.get("isBroken", False),
                    "isFilledWithLiquid": obj.get("isFilledWithLiquid", False),
                    "mass": obj.get("mass", 0.0),
                    "center": obj.get("axisAlignedBoundingBox", {}).get("center", None),
                    "state": obj.get("state", "None")
                }
            return state
        else:
            state = []
            for obj in objects:
                state.append({
                    "objectId": obj["objectId"],
                    "objectType": obj["objectType"],
                })
            return state


    def get_current_agent_view(self, agent_id):
        event = self.controller.last_event
        return event.events[agent_id].frame

    def stop(self):
        self.controller.stop()

# ---------------------------
# AI2ThorTaskExecutor 定義
# ---------------------------
class AI2ThorTaskExecutor:
    def __init__(self, floorplan: str, num_agents: int, task: dict):
        self.env = AI2ThorEnvironment(floorplan, num_agents, task)
        self.task_info = task
        self.action_queue = []  
        self.completed_subtasks = []  
        self.open_subtasks = task.get("open_subtasks", [])
        self.executor_running = True
        self.frame_counter = 0  
        self.recp_id = None
        task_name = task.get("task", "task").replace(" ", "_")
        dt_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_folder = os.path.join("logs", f"{task_name}_{dt_str}")
        os.makedirs(self.log_folder, exist_ok=True)
        for i in range(self.env.num_agents):
            os.makedirs(os.path.join(self.log_folder, f"agent_{i}"), exist_ok=True)
        os.makedirs(os.path.join(self.log_folder, "top_view"), exist_ok=True)
        self.executor_thread = threading.Thread(target=self._exec_actions)
        self.executor_thread.start()

    def _exec_actions(self):
        while self.executor_running:
            if self.action_queue:
                action = self.action_queue.pop(0)
                try:
                    action_type = action.get("action")
                    target = action.get("target")
                    if action_type == "GoToObject":
                        self.go_to_object(target, 0)
                    elif action_type == "PickupObject":
                        self.pickup_object(target, 0)
                    elif action_type == "PutObject":
                        receptacle = action.get("receptacle", "")
                        self.put_object(target, receptacle, 0)
                    elif action_type == "SwitchOn":
                        self.switch_on(target, 0)
                    elif action_type == "SwitchOff":
                        self.switch_off(target, 0)
                    elif action_type == "OpenObject":
                        self.open_object(target, 0)
                    elif action_type == "CloseObject":
                        self.close_object(target, 0)
                    elif action_type == "BreakObject":
                        self.break_object(target, 0)
                    elif action_type == "SliceObject":
                        self.slice_object(target, 0)
                    elif action_type == "CleanObject":
                        self.clean_object(target, 0)
                    elif action_type == "ThrowObject":
                        self.throw_object(target, 0)
                    elif action_type == "RotateRight":
                        degrees = action.get("degrees", 0)
                        self.env.controller.step(dict(action="RotateRight", degrees=degrees, agentId=0))
                    elif action_type == "RotateLeft":
                        degrees = action.get("degrees", 0)
                        self.env.controller.step(dict(action="RotateLeft", degrees=degrees, agentId=0))
                    time.sleep(1)
                    self._save_frames()
                except Exception as e:
                    print("Error executing action:", e)
            else:
                time.sleep(0.1)

    def _save_frames(self):
        event = self.env.controller.last_event
        agent_folder = os.path.join(self.log_folder, "agent_0")
        filename = os.path.join(agent_folder, f"img_{self.frame_counter:05d}.png")
        cv2.imwrite(filename, event.events[0].frame)
        if hasattr(event.events[0], "third_party_camera_frames") and event.events[0].third_party_camera_frames:
            top_view_img = event.events[0].third_party_camera_frames[-1]
            top_view_img_rgb = cv2.cvtColor(top_view_img, cv2.COLOR_BGR2RGB)
            top_view_folder = os.path.join(self.log_folder, "top_view")
            filename = os.path.join(top_view_folder, f"img_{self.frame_counter:05d}.png")
            cv2.imwrite(filename, top_view_img_rgb)
        self.frame_counter += 1

    # ---------------------------
    # LLM Prompt Methods
    # ---------------------------
    def get_prompt(self, stage):
        if stage == "decompose":
            return self.get_decompose_prompt()
        elif stage == "allocate":
            return self.get_allocate_prompt()
        else:
            return ""
    
    def get_decompose_prompt(self, stage="initial"):
        prompt = f"""
            [Task Decomposition Prompt]
            Task: {self.task_info["task"]}
            Scene: {self.env.scene if hasattr(self.env, 'scene') else "Unknown"}
            Available objects with objectId(object_name|position) and objectType: {self.env.get_scene_info(isOjectOnly=True)}
            Robots open subtasks: {self.open_subtasks}
            Completed subtasks: {self.completed_subtasks}
            Skills available (format: function_name <object> or function_name <object><receptacle_object>): 
            ### {actions.ai2thor_actions} ###
            Please decompose the high-level task into specific subtasks that the robots can perform.
            Output a python dictionary with keys "reason" (a string) and "plan" (a list of subtasks).
            Example output: {{"reason": "No subtasks planned yet, so we need to wash the lettuce and place it on the countertop.", 
            "plan": ["wash the lettuce", "place the lettuce on the countertop"]}}
            Ensure that the subtasks are not generic statements like "explore the environment" or "do the task". They should be specific to the task at hand.
            Do not assign subtasks to any particular robot. Try not to modify the subtasks that already exist in the open subtasks list. Rather add new subtasks to the list.
            Since there are {self.env.num_agents} agents moving around in one room, make sure to plan subtasks and actions so that they don't block or bump into each other as much as possible.
            * NOTE: DO NOT OUTPUT ANYTHING EXTRA OTHER THAN WHAT HAS BEEN SPECIFIED
            Let's work this out in a step by step way to be sure we have the right answer.
        """
        return prompt

    def get_allocate_prompt(self, stage="initial"):
        prompt = f"""
            [Task Allocation Prompt]
            Task: {self.task_info["task"]}
            Scene: {self.env.scene if hasattr(self.env, 'scene') else "Unknown"}
            Available objects with objectId(object_name|position) and objectType: {self.env.get_scene_info(isOjectOnly=True)}
            Reachable positions: {self.env.reachable_positions}
            Initial positions: {self.env.agent_positions}
            Robots open subtasks (this is the subtasks that are not completed yet): {self.open_subtasks}
            Completed subtasks (this is the subtasks that are already completed): {self.completed_subtasks}
            Skills available (format: function_name <object> or function_name <object><receptacle_object>): 
            ### {actions.ai2thor_actions} ###
            Please allocate the open subtasks to the robots and generate a detailed action plan for each subtask. Use only the skills that a robot can perform.
            Output a python dictionary where each key is a subtask and the value is a list of action dictionaries.
            Example output: {{"wash the lettuce": ["agent_id": 0, "actions": [
                {{"action": "GoToObject", "target": "Lettuce"}},
                {{"action": "PickupObject", "target": "Lettuce"}},
                {{"action": "GoToObject", "target": "Sink"}},
                {{"action": "PutObject", "target": "Lettuce", "receptacle": "Sink"}}
            ]}}
            Make sure to allocate the subtasks to the robots so that they don't block or bump into each other as much as possible.
            And it is probably better to allocate only one subtask to one robot at the same time.
            * NOTE: DO NOT OUTPUT ANYTHING EXTRA OTHER THAN WHAT HAS BEEN SPECIFIED
            
            Let's work this out in a step by step way to be sure we have the right answer.
        """
        return prompt

    def LM(self, prompt, model, version, max_tokens=128, temperature=0, stop=None, logprobs=1, frequency_penalty=0, isAllocate=False):
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
            messages = [{"role": "system", "content": "You are a expert in task planning. "},{"role": "user", "content": prompt}]
            response = gpt_client.chat.completions.create(model=version, 
                                                    messages=messages, 
                                                    max_tokens=max_tokens, 
                                                    temperature=temperature, 
                                                    frequency_penalty = frequency_penalty)
            content = response.choices[0].message.content.strip()
        return response, content.strip()

    def convert_llm_response(self, response_text):
        """
        將 LLM 的回應轉換成計劃字典。
        假設 LLM 回傳的內容為 JSON 格式，若包含程式碼塊，則去除之。
        """
        try:
            cleaned = response_text.strip()
            if cleaned.startswith("```") and cleaned.endswith("```"):
                cleaned = cleaned.split("```")[1].strip()
            plan = json.loads(cleaned)
            return plan
        except Exception as e:
            print("Error converting LLM response:", e)
            return {}

    # ---------------------------
    # Task Execution Methods
    # ---------------------------
    def execute_plan(self, plan: dict):
        # 若 plan 為 allocation 格式（字典形式），則將所有 action 合併
        if isinstance(plan, dict) and "actions" not in plan:
            actions_list = []
            for subtask, acts in plan.items():
                actions_list.extend(acts)
            plan = {"actions": actions_list}
        for act in plan.get("actions", []):
            self.action_queue.append(act)

    def verify_execution(self):
        state = self.env.get_scene_info()
        open_subtasks = self.task_info.get("open_subtasks", [])
        for subtask in open_subtasks:
            description = subtask.get("description", "")
            expected = subtask.get("expected", "None")
            if self._check_subtask_with_llm(description, state, expected):
                if description not in self.completed_subtasks:
                    self.completed_subtasks.append(description)
        return state

    def _check_subtask_with_llm(self, description, state, expected):
        for obj_name, obj_info in state.items():
            if description.lower() in obj_name.lower() and obj_info.get("state", "None") == expected:
                return True
        return False

    def _generate_videos(self):
        frame_rate = 5
        folder = os.path.join(self.log_folder, "agent_0")
        video_file = os.path.join(self.log_folder, "agent_0.mp4")
        command_set = ['ffmpeg', '-y', '-framerate', str(frame_rate),
                       '-i', os.path.join(folder, "img_%05d.png"),
                       '-pix_fmt', 'yuv420p', video_file]
        subprocess.call(command_set)
        folder = os.path.join(self.log_folder, "top_view")
        video_file = os.path.join(self.log_folder, "top_view.mp4")
        command_set = ['ffmpeg', '-y', '-framerate', str(frame_rate),
                       '-i', os.path.join(folder, "img_%05d.png"),
                       '-pix_fmt', 'yuv420p', video_file]
        subprocess.call(command_set)

    def stop(self):
        self.executor_running = False
        self.executor_thread.join()
        self.env.stop()
        self._generate_videos()

    # ---------------------------
    # Action Methods
    # ---------------------------
    def go_to_object(self, target, agent_id):
        controller = self.env.controller
        objects = controller.last_event.metadata["objects"]
        target_obj = None
        for obj in objects:
            if re.match(target, obj["objectId"]):
                target_obj = obj
                break
        if target_obj is None:
            print(f"Target {target} not found. Skipping navigation.")
            return

        if "axisAlignedBoundingBox" in target_obj and target_obj["axisAlignedBoundingBox"].get("center"):
            target_pos = target_obj["axisAlignedBoundingBox"]["center"]
        else:
            target_pos = target_obj["position"]
        print("Target object found:", target_obj["objectId"], "at", target_pos)

        goal = min(self.env.reachable_positions, key=lambda pos: np.linalg.norm(np.array(pos) - np.array([target_pos["x"], target_pos["y"], target_pos["z"]])))
        goal_arr = np.array(goal)
        threshold = 0.25

        while True:
            metadata = controller.last_event.events[agent_id].metadata
            current_pos = metadata["agent"]["position"]
            curr_arr = np.array([current_pos["x"], current_pos["y"], current_pos["z"]])
            if np.linalg.norm(curr_arr - goal_arr) < threshold:
                print("Agent is close enough to goal.")
                break
            start = min(self.env.reachable_positions, key=lambda pos: np.linalg.norm(np.array(pos) - curr_arr))
            graph = build_graph(self.env.reachable_positions, edge_threshold=0.5)
            path = astar_search(graph, tuple(start), tuple(goal))
            if not path:
                print("No path found, using Teleport as fallback.")
                controller.step(dict(action="Teleport", position={"x": goal[0], "y": goal[1], "z": goal[2]}, agentId=agent_id))
                break
            print("New planned path:", path)
            for node in path:
                node_arr = np.array(node)
                count_no_progress = 0
                prev_d = np.linalg.norm(curr_arr - node_arr)
                while True:
                    metadata = controller.last_event.events[agent_id].metadata
                    current_pos = metadata["agent"]["position"]
                    curr_arr = np.array([current_pos["x"], current_pos["y"], current_pos["z"]])
                    d = np.linalg.norm(curr_arr - node_arr)
                    print(f"Distance to node {node}: {d:.3f}")
                    if d < threshold:
                        break
                    if abs(d - prev_d) < 0.01:
                        count_no_progress += 1
                    else:
                        count_no_progress = 0
                    prev_d = d
                    if count_no_progress >= 3:
                        print("Stuck on current segment, re-planning route from current position.")
                        break
                    controller.step(dict(action="MoveAhead", agentId=agent_id))
                    time.sleep(0.5)
                if count_no_progress >= 3:
                    break
            if count_no_progress >= 3:
                continue
            else:
                break

        metadata = controller.last_event.events[agent_id].metadata
        current_pos = metadata["agent"]["position"]
        vec_to_target = [target_pos["x"] - current_pos["x"], target_pos["z"] - current_pos["z"]]
        norm = np.linalg.norm(vec_to_target) + 1e-6
        unit_vector = np.array(vec_to_target) / norm
        desired_angle = math.degrees(math.atan2(unit_vector[0], unit_vector[1]))
        current_rot = metadata["agent"]["rotation"]["y"]
        rot_angle = (desired_angle - current_rot + 360) % 360
        if rot_angle > 180:
            rot_angle -= 360
        if rot_angle > 0:
            controller.step(dict(action="RotateRight", degrees=abs(rot_angle), agentId=agent_id))
        else:
            controller.step(dict(action="RotateLeft", degrees=abs(rot_angle), agentId=agent_id))
        print("Reached target object and aligned towards it.")

    def pickup_object(self, pick_obj, agent_id):
        controller = self.env.controller
        objects = controller.last_event.metadata["objects"]
        pick_obj_id = None
        for obj in objects:
            if re.match(pick_obj, obj["objectId"]):
                pick_obj_id = obj["objectId"]
                break
        if pick_obj_id is None:
            print(f"Object {pick_obj} not found for pickup.")
            return
        print("Picking up ", pick_obj_id)
        controller.step(dict(action="PickupObject", objectId=pick_obj_id, agentId=agent_id))
        time.sleep(1)

    def put_object(self, put_obj, receptacle, agent_id):
        controller = self.env.controller
        objects = controller.last_event.metadata["objects"]
        recp_obj_id = None
        metadata = controller.last_event.events[agent_id].metadata
        robot_pos = [metadata["agent"]["position"]["x"], metadata["agent"]["position"]["y"], metadata["agent"]["position"]["z"]]
        closest_dist = float("inf")
        for obj in objects:
            if re.match(receptacle, obj["objectId"]):
                center = obj.get("axisAlignedBoundingBox", {}).get("center", None)
                if center:
                    d = distance_pts(robot_pos, [center["x"], center["y"], center["z"]])
                    if d < closest_dist:
                        closest_dist = d
                        recp_obj_id = obj["objectId"]
        if recp_obj_id is None:
            print(f"Receptacle {receptacle} not found for putting object.")
            return
        time.sleep(1)
        controller.step(dict(action="PutObject", objectId=recp_obj_id, agentId=agent_id))
        print(f"Put {put_obj} into {receptacle}")
        time.sleep(1)

    def switch_on(self, sw_obj, agent_id):
        controller = self.env.controller
        objects = list(set([obj["objectId"] for obj in controller.last_event.metadata["objects"]]))
        if sw_obj == "StoveKnob":
            for obj_id in objects:
                if re.match(sw_obj, obj_id):
                    self.go_to_object(obj_id, agent_id)
                    controller.step(dict(action="ToggleObjectOn", objectId=obj_id, agentId=agent_id))
                    time.sleep(0.1)
            print(f"Switched on all {sw_obj}")
        else:
            sw_obj_id = None
            for obj_id in objects:
                if re.match(sw_obj, obj_id):
                    sw_obj_id = obj_id
                    break
            if sw_obj_id is None:
                print(f"Switch {sw_obj} not found.")
                return
            self.go_to_object(sw_obj_id, agent_id)
            time.sleep(1)
            controller.step(dict(action="ToggleObjectOn", objectId=sw_obj_id, agentId=agent_id))
            print(f"Switched on {sw_obj}")
            time.sleep(1)

    def switch_off(self, sw_obj, agent_id):
        controller = self.env.controller
        objects = list(set([obj["objectId"] for obj in controller.last_event.metadata["objects"]]))
        if sw_obj == "StoveKnob":
            for obj_id in objects:
                if re.match(sw_obj, obj_id):
                    controller.step(dict(action="ToggleObjectOff", objectId=obj_id, agentId=agent_id))
                    time.sleep(0.1)
            print(f"Switched off all {sw_obj}")
        else:
            sw_obj_id = None
            for obj_id in objects:
                if re.match(sw_obj, obj_id):
                    sw_obj_id = obj_id
                    break
            if sw_obj_id is None:
                print(f"Switch {sw_obj} not found.")
                return
            self.go_to_object(sw_obj_id, agent_id)
            time.sleep(1)
            controller.step(dict(action="ToggleObjectOff", objectId=sw_obj_id, agentId=agent_id))
            print(f"Switched off {sw_obj}")
            time.sleep(1)

    def open_object(self, obj_pattern, agent_id):
        controller = self.env.controller
        objects = controller.last_event.metadata["objects"]
        target_id = None
        for obj in objects:
            if re.match(obj_pattern, obj["objectId"]):
                target_id = obj["objectId"]
                break
        if target_id is None:
            print(f"Object {obj_pattern} not found for opening.")
            return
        if self.recp_id is not None:
            target_id = self.recp_id
        self.go_to_object(target_id, agent_id)
        time.sleep(1)
        controller.step(dict(action="OpenObject", objectId=target_id, agentId=agent_id))
        time.sleep(1)
        print(f"Opened object {target_id}")

    def close_object(self, obj_pattern, agent_id):
        controller = self.env.controller
        objects = controller.last_event.metadata["objects"]
        target_id = None
        for obj in objects:
            if re.match(obj_pattern, obj["objectId"]):
                target_id = obj["objectId"]
                break
        if target_id is None:
            print(f"Object {obj_pattern} not found for closing.")
            return
        self.go_to_object(target_id, agent_id)
        time.sleep(1)
        controller.step(dict(action="CloseObject", objectId=target_id, agentId=agent_id))
        time.sleep(1)
        print(f"Closed object {target_id}")

    def break_object(self, obj_pattern, agent_id):
        controller = self.env.controller
        objects = controller.last_event.metadata["objects"]
        target_id = None
        for obj in objects:
            if re.match(obj_pattern, obj["objectId"]):
                target_id = obj["objectId"]
                break
        if target_id is None:
            print(f"Object {obj_pattern} not found for breaking.")
            return
        self.go_to_object(target_id, agent_id)
        time.sleep(1)
        controller.step(dict(action="BreakObject", objectId=target_id, agentId=agent_id))
        time.sleep(1)
        print(f"Broke object {target_id}")

    def slice_object(self, obj_pattern, agent_id):
        controller = self.env.controller
        objects = controller.last_event.metadata["objects"]
        target_id = None
        for obj in objects:
            if re.match(obj_pattern, obj["objectId"]):
                target_id = obj["objectId"]
                break
        if target_id is None:
            print(f"Object {obj_pattern} not found for slicing.")
            return
        self.go_to_object(target_id, agent_id)
        time.sleep(1)
        controller.step(dict(action="SliceObject", objectId=target_id, agentId=agent_id))
        time.sleep(1)
        print(f"Sliced object {target_id}")

    def clean_object(self, obj_pattern, agent_id):
        controller = self.env.controller
        objects = controller.last_event.metadata["objects"]
        target_id = None
        for obj in objects:
            if re.match(obj_pattern, obj["objectId"]):
                target_id = obj["objectId"]
                break
        if target_id is None:
            print(f"Object {obj_pattern} not found for cleaning.")
            return
        self.go_to_object(target_id, agent_id)
        time.sleep(1)
        controller.step(dict(action="CleanObject", objectId=target_id, agentId=agent_id))
        time.sleep(1)
        print(f"Cleaned object {target_id}")

    def throw_object(self, obj_pattern, agent_id):
        controller = self.env.controller
        objects = controller.last_event.metadata["objects"]
        target_id = None
        for obj in objects:
            if re.match(obj_pattern, obj["objectId"]):
                target_id = obj["objectId"]
                break
        if target_id is None:
            print(f"Object {obj_pattern} not found for throwing.")
            return
        self.go_to_object(target_id, agent_id)
        time.sleep(1)
        controller.step(dict(action="ThrowObject", objectId=target_id, agentId=agent_id, moveMagnitude=7))
        time.sleep(1)
        print(f"Threw object {target_id}")

# ---------------------------
# 主程序
# ---------------------------
if __name__ == "__main__":
    task = get_task_from_file("test", "6")
    executor = AI2ThorTaskExecutor(floorplan="FloorPlan6", num_agents=3, task=task)
    
    # 規劃階段：
    # 若您希望自行輸入計劃，請取消下行註解：
    # user_plan = input("請以 JSON 格式輸入計劃: ")
    # plan = json.loads(user_plan)
    # 否則，使用 LLM 規劃：
    decompose_prompt = executor.get_decompose_prompt("initial")
    # print("Decompose prompt:", decompose_prompt)
    # _, decompose_response = executor.LM(decompose_prompt, model="gpt", version="gpt-4", max_tokens=1000, temperature=0.5)
    # print("Decompose response:", decompose_response)
    decompose_response = {"reason": "The lettuce needs to be washed before it can be placed on the countertop.", 
"plan": ["GoToObject <robot><Lettuce|+01.11|+00.83|-01.43>", "PickupObject <robot><Lettuce|+01.11|+00.83|-01.43>", "GoToObject <robot><Faucet|+01.31|+00.90|-01.54>", "CleanObject <robot><Lettuce|+01.11|+00.83|-01.43>", "GoToObject <robot><CounterTop|-00.36|+00.95|+01.09>", "PutObject <robot><Lettuce|+01.11|+00.83|-01.43><CounterTop|-00.36|+00.95|+01.09>"]}
    # print("Decompose response:", decompose_response)
    # decompose_plan = executor.convert_llm_response(decompose_response)
    # print("Decomposed plan:", decompose_plan)
    executor.open_subtasks = decompose_response.get("plan", [])
    # print("Open subtasks:", executor.open_subtasks)
    
    allocate_prompt = executor.get_allocate_prompt("initial")
    # print("Allocate prompt:", allocate_prompt)
    _, allocate_response = executor.LM(allocate_prompt, model="gpt", version="gpt-4", max_tokens=1000, temperature=0.5)
    print("Allocate response:", allocate_response)
  
    allocation_plan = executor.convert_llm_response(allocate_response)
    print("Allocation plan:", allocation_plan)
    
    # 假設最終計劃採用 allocation_plan
    # executor.execute_plan(allocation_plan)
    
    time.sleep(30)
    
    state = executor.verify_execution()
    print("驗證結果：")
    # print(json.dumps(state, indent=2, ensure_ascii=False))
    
    executor.stop()
