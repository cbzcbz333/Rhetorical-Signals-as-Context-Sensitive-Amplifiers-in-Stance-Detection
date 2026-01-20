import requests
import json
url = "https://api.shubiaobiao.com/v1/chat/completions"
payload = json.dumps({
   "messages": [
      {
         "role": "system",
         "content": "你是一个大语言模型机器人"
      },
      {
         "role": "user",
         "content": "你好"
      }
   ],
   "stream": False,
   "model": "gpt-4.1-mini",
   "temperature": 0.5,
   "presence_penalty": 0,
   "frequency_penalty": 0,
   "top_p": 1
})
headers = {
    "Content-Type": "application/json",
    # sk-xxx替换为自己的key
   'Authorization': 'Bearer sk-JhzsitLNi4ztobLxgmbdIBCPXtUPTFwFmkYdAsOILqW1xDEy',
}
response = requests.request("POST", url, headers=headers, data=payload)
print(response.text)