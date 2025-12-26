#!/bin/bash
kill -9 $(ps -ef| grep 'infer.py'| grep python3 | grep -v grep| awk '{print $2}')  #需手动杀死进程
kill -9 $(ps -ef| grep 'ffn.py'| grep python3 | grep -v grep| awk '{print $2}')  #需手动杀死进程