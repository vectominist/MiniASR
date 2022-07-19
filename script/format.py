#!/usr/bin/env python3

import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Only checks the files")
    args = parser.parse_args()

    print("Run flake8")
    # stop the build if there are Python syntax errors or undefined names
    os.system("flake8 ./ --count --select=E9,F63,F7,F82 --show-source --statistics")
    # exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
    os.system(
        "flake8 ./ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics"
    )

    print("Run black")
    if args.check:
        os.system("black --check ./")
    else:
        os.system("black ./")

    print("Run isort")
    if args.check:
        os.system("isort --profile black --check ./")
    else:
        os.system("isort --profile black ./")

    if args.check:
        print("Successfully passed the format check!")


if __name__ == "__main__":
    main()
