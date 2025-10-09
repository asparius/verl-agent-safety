# Copyright 2025 - AI Safety Gridworlds Integration for verl-agent
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import random
import json
import re
from typing import List


def safety_gridworld_projection(text_actions: List[str]):
    """
    Project LLM text outputs to valid Safety Gridworld actions.
    
    Args:
        text_actions: List of text strings from LLM outputs
        env_name: Name of the Safety Gridworld environment
        
    Returns:
        output_indices: List of action indices (0-3 for valid, -1 for invalid)
        valids: List of validity flags (1 for valid, 0 for invalid)
    """
    output_indices = []
    valids = []
    
    # All Safety Gridworlds use the same action space: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
    action_list = ["up", "down", "left", "right"]
    action_aliases = {
        "up": ["up", "north", "u", "n", "move up", "go up", "move north", "go north"],
        "down": ["down", "south", "d", "s", "move down", "go down", "move south", "go south"],
        "left": ["left", "west", "l", "w", "move left", "go left", "move west", "go west"],
        "right": ["right", "east", "r", "e", "move right", "go right", "move east", "go east"],
    }
    
    for string in text_actions:
        if not isinstance(string, str):
            # Directly output -1 if the input is not a string
            output_indices.append(-1)
            valids.append(0)
            continue
        
        string_lower = string.lower()
        extracted_action = None
        
        # Strategy 1: Try to parse as JSON and extract "action" field
        try:
            # Find JSON object in the string
            json_match = re.search(r'\{[^}]+\}', string)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                if "action" in parsed:
                    extracted_action = str(parsed["action"]).lower().strip()
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Strategy 2: Try to extract from XML-style tags <action>...</action>
        if not extracted_action:
            action_tag_match = re.search(r'<action>\s*([^<]+)\s*</action>', string_lower)
            if action_tag_match:
                extracted_action = action_tag_match.group(1).strip()
        
        # Strategy 3: Look for '"action":' keyword and extract value after it
        if not extracted_action:
            action_index = string_lower.find('"action":')
            if action_index != -1:
                # Extract substring after "action":
                after_action = string_lower[action_index + len('"action":'):]
                # Try to find the action value (might be quoted or not)
                value_match = re.search(r'["\']?\s*(\w+(?:\s+\w+)?)\s*["\']?', after_action)
                if value_match:
                    extracted_action = value_match.group(1).strip()
        
        # If we extracted something specific, check if it matches an action
        if extracted_action:
            for action in action_list:
                if extracted_action in action_aliases[action]:
                    output_indices.append(action_list.index(action))
                    valids.append(1)
                    break
            else:
                # Extracted something but it doesn't match any valid action
                output_indices.append(-1)
                valids.append(0)
            continue
        
        # Strategy 4: Fall back to searching for any alias in the entire string
        contained_actions = []
        for action in action_list:
            for alias in action_aliases[action]:
                # Use word boundaries to avoid partial matches
                if re.search(r'\b' + re.escape(alias) + r'\b', string_lower):
                    contained_actions.append(action)
                    break  # Only add each action once
        
        # Remove duplicates
        contained_actions = list(set(contained_actions))
        
        if len(contained_actions) == 1:
            # Only one unique action keyword found
            output_indices.append(action_list.index(contained_actions[0]))
            valids.append(1)
        else:
            # None or multiple keywords found - invalid action
            output_indices.append(-1)
            valids.append(0)
    
    return output_indices, valids