path = '/root/gateway/nginx/conf.d/app.ctpai.vn.conf'
with open(path, 'r') as f:
    content = f.read()

content = content.replace('rewrite ^/api/primer-diamond/(.*) /api/v1/$1 break;', 'rewrite ^/api/[^/]+/(.*) /api/v1/$1 break;')

with open(path, 'w') as f:
    f.write(content)
