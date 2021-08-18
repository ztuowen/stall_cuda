#!/bin/env python3

import re
import subprocess
import sys

def binSlicer(a, fr, l):
    a = a >> fr
    a = a & ((1 << l) - 1)
    return a

def parse21(a):
    result = ""
    stall = binSlicer(a, 0, 4)
    yi = binSlicer(a, 4, 1)
    write = binSlicer(a, 5, 3) + 1
    if write > 6:
        write = "-"
    else:
        write = str(write)
    read = binSlicer(a, 8, 3) + 1
    if read > 6:
        read = "-"
    else:
        read = str(read)
    barr = binSlicer(a, 11, 6)
    reuse = binSlicer(a, 17, 4)
    return "{0:06b}:{1}:{2}:{3:d}:{4:2d}".format(barr, read, write, yi, stall)

if __name__ == '__main__':
    # argv[1] is the cubin file to disassemble
    r = subprocess.run(["cuobjdump", "--dump-sass", sys.argv[1]],stdout=subprocess.PIPE)
    lines = r.stdout.decode('utf-8').split('\n')
    mode = 0
    major = 0
    minor = 0
    cnt = 0
    width = 15
    pad = width * " "
    ctrl = 0
    for line in lines:
        uline = line.upper()
        ch = re.match(r'^\s+CODE FOR SM_(\d+)', uline)
        if ch:
            major = int(ch.group(1))
            minor = major % 10
            major = major // 10
            if major >= 5:
                mode = 1
                cnt = 0
            else:
                mode = 0
        if mode == 1:
            loc = re.match(r'^\s+/\*([0-9A-F]+)\*/', uline)
            inst = re.search(r'/\* 0X([0-9A-F]+) \*/', uline)
            if major < 7:
                if inst:
                    if cnt % 4 == 0:
                        ctrl = int(inst.group(1), 16)
                    cnt += 1
                if loc:
                    print(parse21(ctrl), end="")
                    ctrl = ctrl >> 21
                else:
                    print(pad, end="")
            if major == 7:
                if inst:
                    if cnt % 2 == 1:
                        ctrl = int(inst.group(1), 16)
                        print(parse21(ctrl >> 41), end="")
                    else:
                        print(pad, end="")
                    cnt += 1
                else:
                    print(pad, end="")
        else:
            print(pad, end="")
        print(line)
