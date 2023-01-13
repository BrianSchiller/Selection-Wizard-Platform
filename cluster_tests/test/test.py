#!/usr/bin/env python3
"""Simple program to test reading and writing files works."""

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'number',
        type=int,
        default=2,
        help='Number to multiply with the number in input.txt.')
    parser.add_argument(
        'job_id',
        type=str,
        help='PBS job ID.')
    args = parser.parse_args()

    with open("input.txt") as infile:
        print("Reading...")
        in_num = int(infile.readline())

    out_num = args.number * in_num

    with open("output.txt", "w") as outfile:
        print("Writing...")
        outfile.write(f"Number: {str(out_num)}\n")
        outfile.write(f"Job ID: {args.job_id}")

    print("Done!")
