import re
from pathlib import Path

h = Path('web/index.html').read_text(encoding='utf-8')
ids = set(re.findall(r'id="([^"]+)"', h))
used = set(re.findall(r"\$\('([^']+)'\)", h))
missing = sorted(used - ids)
print('HTML id 数:', len(ids), '| JS 引用数:', len(used))
print('引用了但不存在的 id:', missing or '无')

secs = set(re.findall(r'<section id="v-([a-z]+)"', h))
navs = set(re.findall(r'data-v="([a-z]+)"', h))
print('视图:', sorted(secs))
print('导航:', sorted(navs), '| 无对应视图:', sorted(navs - secs) or '无')

calls = set(re.findall(r'onclick="([a-zA-Z_]+)\(', h))
defined = set(re.findall(r'function ([a-zA-Z_]+)\(', h))
print('onclick 调用:', sorted(calls))
print('未定义函数:', sorted(c for c in calls if c not in defined) or '无')

for tag in ('section', 'div', 'table', 'tbody', 'header', 'main'):
    o = len(re.findall(r'<' + tag + r'[\s>]', h))
    c = len(re.findall(r'</' + tag + '>', h))
    print('  <%s> %d / </%s> %d  %s' % (tag, o, tag, c, 'OK' if o == c else '★不配对★'))

print('JS 里 fetch 的接口:')
for m in sorted(set(re.findall(r'api\(`([^`]+)`', h))):
    print('   ', m)
