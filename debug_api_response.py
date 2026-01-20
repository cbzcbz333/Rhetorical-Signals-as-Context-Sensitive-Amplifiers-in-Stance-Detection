import http.client

conn = http.client.HTTPSConnection("api.shubiaobiao.com")
payload = ''
headers = {
   'Authorization': 'Bearer sk-JhzsitLNi4ztobLxgmbdIBCPXtUPTFwFmkYdAsOILqW1xDEy'
}
conn.request("GET", "/v1/models", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))