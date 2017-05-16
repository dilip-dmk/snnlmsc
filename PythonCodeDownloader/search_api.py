import json
import timeit
import urllib2
from urllib2 import *

repo_urls = []
start = timeit.default_timer()
for i, j in ((a, b) for a in range(6, 11, 1) for b in range(7, 5, -1)):  # next range for i (6, 11, 1)
    url = 'https://api.github.com/search/repositories?q=python+language:python+pushed:<201%d-12-31&sort=stars&order=desc&per_page=100&page=%d' % (
        j, i)
    print url
    req = urllib2.Request(url, headers={'User-Agent': "Mozilla/5.0"})
    con = urlopen(url)
    data = con.read().decode('utf8')
    text = json.loads(data)
    for x in range(0, 100, 1):  # only 100 results displayed per page
        u = text['items'][x]['svn_url']
        repo_urls.append(u)
        print u
    con.close()

with open("daily_repo.txt", "a") as f:
    for p in repo_urls:
        f.write(p)
        f.write('\n')

print "Number of unique Repositories  : %d" % len(list(set(repo_urls)))
stop = timeit.default_timer()
print "Elapsed time                   : %8.2f seconds" % (stop - start)
