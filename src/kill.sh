#! /bin/bash
############################################
#
# Author: liuanan
# Create time: 2023-04-29 11:30:31
# E-Mail: liuanan123@qq.com
# version 1.1
#
############################################

ps aux | grep 'python3 run.py' | grep -v grep | awk '{print $2}' | xargs kill
