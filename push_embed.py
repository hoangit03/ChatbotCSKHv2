import sys
import subprocess

with open(r"d:\CTGroup\Mlops\ChatbotCSKHv2\app\infrastructure\llm\providers\embed_provider.py", "rb") as f:
    content = f.read()

import base64
encoded = base64.b64encode(content).decode('ascii')

script = f"""
import base64
content = base64.b64decode('{encoded}')
with open('/root/ChatbotCSKHv2/app/infrastructure/llm/providers/embed_provider.py', 'wb') as f:
    f.write(content)
"""

subprocess.run(["ssh", "root@103.186.101.200", "python", "-c", script])
