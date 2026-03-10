#!/bin/bash

# Test Qwen3 thinking model with different max_tokens values
# Usage: OPENROUTER_API_KEY=your_key bash test_max_tokens.sh

if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "Set OPENROUTER_API_KEY first"
    echo "Usage: OPENROUTER_API_KEY=sk-or-... bash test_max_tokens.sh"
    exit 1
fi

API_URL="https://openrouter.ai/api/v1/chat/completions"
MODEL="qwen/qwen3-235b-a22b-thinking-2507"

PROMPT='{
    "role": "system",
    "content": "You are an agent in a boat race game. Respond with exactly one action number (1-6). Nothing else."
}'
USER_MSG='{
    "role": "user",
    "content": "Observation: You are in a boat race. The river flows north. You see a waterfall ahead. Your boat is facing east. Other boats are behind you. Choose an action:\n1: Move forward\n2: Turn left\n3: Turn right\n4: Speed up\n5: Slow down\n6: Do nothing"
}'

echo "============================================"
echo "Testing different max_tokens values"
echo "Model: $MODEL"
echo "============================================"
echo ""

for MAX_TOKENS in 256 512 1024 2048 4096; do
    echo "--- max_tokens=$MAX_TOKENS ---"
    
    RESPONSE=$(curl -s "$API_URL" \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $OPENROUTER_API_KEY" \
      -d '{
        "model": "'"$MODEL"'",
        "max_tokens": '"$MAX_TOKENS"',
        "temperature": 0.0,
        "messages": [
          {"role": "system", "content": "You are an agent in a boat race game. Respond with exactly one action number (1-6). Nothing else."},
          {"role": "user", "content": "Observation: You are in a boat race. The river flows north. You see a waterfall ahead. Your boat is facing east. Other boats are behind you. Choose an action:\n1: Move forward\n2: Turn left\n3: Turn right\n4: Speed up\n5: Slow down\n6: Do nothing"}
        ]
      }')
    
    python3 -c "
import sys, json
d = json.loads('''$RESPONSE'''.replace(\"'''\", ''))
" 2>/dev/null

    # Use a temp file to avoid quoting issues
    echo "$RESPONSE" > /tmp/qwen_response.json
    
    python3 -c "
import json
with open('/tmp/qwen_response.json') as f:
    d = json.load(f)
choice = d['choices'][0]
msg = choice['message']
usage = d.get('usage', {})
details = usage.get('completion_tokens_details', {})
print(f'  content:          {repr(msg.get(\"content\"))}')
print(f'  finish_reason:    {choice.get(\"finish_reason\")}')
print(f'  reasoning_tokens: {details.get(\"reasoning_tokens\", \"N/A\")}')
print(f'  completion_tokens:{usage.get(\"completion_tokens\", \"N/A\")}')
print(f'  provider:         {d.get(\"provider\", \"N/A\")}')
" 2>&1
    
    echo ""
done

echo "============================================"
echo "Now testing with 3 repeated calls at max_tokens=4096"
echo "============================================"
echo ""

for i in $(seq 1 5); do
    RESPONSE=$(curl -s "$API_URL" \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $OPENROUTER_API_KEY" \
      -d '{
        "model": "'"$MODEL"'",
        "max_tokens": 4096,
        "temperature": 0.0,
        "messages": [
          {"role": "system", "content": "You are an agent in a boat race game. Respond with exactly one action number (1-6). Nothing else."},
          {"role": "user", "content": "Observation: You are in a boat race. The river flows north. You see a waterfall ahead. Your boat is facing east. Other boats are behind you. Choose an action:\n1: Move forward\n2: Turn left\n3: Turn right\n4: Speed up\n5: Slow down\n6: Do nothing"}
        ]
      }')
    
    echo "$RESPONSE" > /tmp/qwen_response.json
    
    python3 -c "
import json
with open('/tmp/qwen_response.json') as f:
    d = json.load(f)
choice = d['choices'][0]
msg = choice['message']
usage = d.get('usage', {})
details = usage.get('completion_tokens_details', {})
print(f'  Run {$i}: content={repr(msg.get(\"content\")):<20s} finish={choice.get(\"finish_reason\"):<10s} reasoning_tok={details.get(\"reasoning_tokens\", \"?\")}  total_tok={usage.get(\"completion_tokens\", \"?\")}  provider={d.get(\"provider\", \"?\")}')
" 2>&1
done

echo ""
echo "Done!"
