"""
    mail@kaiploeger.net
"""

import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser(description="This skript only prints the values of it's arguments.",
                        epilog="This secteion should appear at the end of --help.")
parser.add_argument('-mfl', '--my_flag', action='store_true', help="The value of my_flag will be printed")
parser.add_argument('-mi', '--my_int', type=int, default=0, help="The value of my_int will be printed")
parser.add_argument('-mf', '--my_float', type=float, default=0, help="The value of my_float will be printed")


def my_func():
    args = parser.parse_args()

    print(f"my_flag = {args.my_flag}")
    print(f"my_int  = {args.my_int}")
    print(f"my_float  = {args.my_float}")

if __name__ == '__main__':
    my_func()

