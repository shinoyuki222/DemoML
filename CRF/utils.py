import itertools
import os
import json
import copy
import pickle



def check_dir(path):
    return os.path.exists(path)

def remove_old_file(path):
    if check_dir(path):
        os.remove(path)
def create_dir(path):
    if not check_dir(path):
        os.mkdir(path)
        
def save_obj(obj, filename):
    remove_old_file(filename)
    json.dump(obj, open(filename, 'w', encoding="utf8"), ensure_ascii=False)

def load_obj(filename):
    obj = json.load(open(filename, 'r', encoding="utf8"))
    return obj
