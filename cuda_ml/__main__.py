'''
Author: David Akre
Date: 11/14/17
Description: Main entry point for cuda ml application
'''

import sys
import click
from ml.engine import Engine


@click.command()
@click.option('-p', '--pictures_location', required=True)
@click.option('-s', '--serial', default=False)
def main(pictures_location, serial):
    print("****************************")
    print("Starting CUDA ML Application")
    print("****************************")
    print("")
    print("")

    engine = Engine(pictures_location, serial)
    engine.start()

if __name__ == "__main__":
    main()
