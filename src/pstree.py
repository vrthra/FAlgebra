#!/usr/bin/env python3

import sys
import os
import subprocess
from collections import defaultdict

def psaxo():
    cmd = ['ps', 'axo', 'ppid,pid,comm,etimes,args']
    #pid,comm,lstart,etimes,time,args 
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    proc.stdout.readline()
    for line in proc.stdout:
        fields = line.decode().rstrip().split(None, 4)
        yield fields

def get_cwd(pid):
    try:
        res = subprocess.check_output(['readlink', '-e', '/proc/%d/cwd' % pid]).decode().strip()
        return res
    except:
        return ''

def show_tree(pids, pid, prefix=''):
    p = pids[pid]
    cwd = get_cwd(pid)
    sys.stdout.write('{0}{1}s ({2}) [{3}] {4}{5}'.format(prefix,p['etimes'],pid,cwd,p['args'], os.linesep))
    if pids[pid]['children']:
        prefix = prefix.replace('-', ' ')
        for idx,spid in enumerate(pids[pid]['children']):
            show_tree(pids, spid, prefix+' |- ')

if __name__ == '__main__':
    pids = defaultdict(lambda:{"cmd":"", "children":[], 'ppid':None})
    for ppid,pid,command,etimes,args in psaxo():
        ppid = int(ppid)
        pid  = int(pid)
        pids[pid].update({ "cmd": command, "args": args, "etimes": etimes, "ppid": ppid})
        pids[ppid]['children'].append(pid)

    show_tree(pids, int(sys.argv[1]), '')
