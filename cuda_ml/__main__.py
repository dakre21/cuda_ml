'''
Author: David Akre
Date: 11/14/17
Description: Main entry point for cuda ml application
'''

import sys
from ml.engine import Engine

def main():
    print("****************************")
    print("Starting CUDA ML Application")
    print("****************************")
    print("")
    print("")

    engine = Engine()
    engine.start()

if __name__ == "__main__":
    main()
