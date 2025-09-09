curl -v http://0.0.0.0:8000/v1/chat/completions \
-H 'Content-Type: application/json' \
-d \
'{ "model": "/data2/models/deepseek-v2-lite",
"messages": [
{"role": "user", "content": "1 3 5 7 9"} ],
"temperature": 0.6,
"repetition_penalty": 1.0,
"top_p": 0.95,
"top_k": 40,
"max_tokens": 20,
"stream": false}'
