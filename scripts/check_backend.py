import urllib.request

try:
    with urllib.request.urlopen('http://localhost:8000/', timeout=5) as r:
        print('STATUS', r.status)
        print(r.read().decode())
except Exception as e:
    print('ERROR', e)
