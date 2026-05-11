import sys
import subprocess
import base64

with open(r"d:\CTGroup\Mlops\ChatbotCSKHv2\frontend\src\views\ChatView.vue", "rb") as f:
    content = f.read()

encoded = base64.b64encode(content).decode('ascii')

script = f"""
import base64
content = base64.b64decode('{encoded}')
with open('/root/ChatbotCSKHv2/frontend/src/views/ChatView.vue', 'wb') as f:
    f.write(content)
"""

subprocess.run(["ssh", "root@103.186.101.200", "python3", "-c", script])
