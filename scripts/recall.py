"""
This script takes in an automatically-generated causal KG of wikidata events,
and a Base KG (available causal knowledge in Wikidata) and measures the "recall".
"""

import time
import argparse
import json
import csv
import logging
import os
import sys
from pytablewriter import MarkdownTableWriter


def evaluate_recall(input_kg, base_kg):
    """
    This function takes in an output of causal extraction in the JSON CausalKG format
    as well as the Base KG (causal knowledge in Wikidata in CausalKG format) and produces
    recall evaluation measures.
    """
    base_src_set = set()
    for pair in base_kg:
        base_src_set.add(pair[0])
    kg_src_set = set()
    for pair in input_kg:
        kg_src_set.add(pair[0])
    base_count = len(base_src_set.intersection(kg_src_set))
    base_coverage = base_count * 1.0 / len(base_src_set)

    hit_count = len(input_kg.intersection(base_kg))
    precision = hit_count * 1.0 / len(input_kg) if len(input_kg) > 0 else 0
    recall = hit_count * 1.0 / len(base_kg)
    rel_count = len(input_kg)
    base_kg_size = len(base_kg)
    return (
        hit_count,
        rel_count,
        base_kg_size,
        precision,
        recall,
        base_count,
        base_coverage,
    )


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--base_kg_file",
        required=False,
        help="Base KG file.",
        default="data/base-kg/base-kg-v1.jsonl",
    )
    parser.add_argument(
        "-i",
        "--input_kg_file",
        required=False,
        help="Input automatically-constructed KG for evaluation.",
        default="data/extracted-kg/causenet-full-linked-v1.jsonl",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        required=False,
        help="Output CSV file.",
        default="results/recall-v1.csv",
    )
    parser.add_argument(
        "-m",
        "--output_md",
        required=False,
        help="Output Markdown file.",
        default="results/recall-v1.md",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logging.info("Starting evaluation with args: %s", args)

    eval_base_kg = set()
    base_types_kg = set()
    base_examples_kg = set()
    with open(args.base_kg_file, "r", encoding="utf-8") as base_kg_file:
        for line in base_kg_file:
            obj = json.loads(line)
            source_id = obj["event"]["id"]
            for c in obj["consequences"]:
                target_id = c["id"]
                eval_base_kg.add((source_id, target_id))
                base_types_kg.add((source_id, target_id))
                if "examples" in c and len(c["examples"]) > 0:
                    for e in c["examples"]:
                        e_cid = e["cause"]["id"]
                        e_eid = e["effect"]["id"]
                        eval_base_kg.add((e_cid, e_eid))
                        base_examples_kg.add((e_cid, e_eid))
    logging.info(
        "Done reading Base KG with %d relations, including %d type relations "
        "and %d examples.",
        len(eval_base_kg),
        len(base_types_kg),
        len(base_examples_kg),
    )
    eval_kg = set()
    eval_types_kg = set()
    eval_examples_kg = set()
    with open(args.input_kg_file, "r", encoding="utf-8") as input_kg_file:
        for line in input_kg_file:
            obj = json.loads(line)
            if "event" not in obj and "cause" not in obj:
                logging.error(
                    "Input KG does not have 'event' or 'cause' field in line: %s", line
                )
                sys.exit(1)
            source_id = obj["event"]["id"] if "event" in obj else obj["cause"]["id"][0]

            if "consequences" not in obj and "effect" not in obj:
                logging.error(
                    "Input KG does not have 'effect' or 'consequences' field in line: %s",
                    line,
                )
                sys.exit(1)
            if "effect" in obj:
                obj["consequences"] = [
                    {"id": obj["effect"]["id"][0], "label": obj["effect"]["label"]}
                ]
            for c in obj["consequences"]:
                target_id = c["id"]
                eval_kg.add((source_id, target_id))
                eval_types_kg.add((source_id, target_id))
                if "examples" in c and len(c["examples"]) > 0:
                    for e in c["examples"]:
                        e_cid = e["cause"]["id"]
                        e_eid = e["effect"]["id"]
                        eval_kg.add((e_cid, e_eid))
                        eval_examples_kg.add((e_cid, e_eid))

    logging.info(
        "Read Evaluation KG %s with %d relations, including %d type relations "
        "and %d examples.",
        args.input_kg_file,
        len(eval_kg),
        len(eval_types_kg),
        len(eval_examples_kg),
    )

    rows = []
    if os.path.exists(args.output_file):
        with open(args.output_file, "r", encoding="utf-8") as reader:
            for row in csv.reader(reader):
                rows.append(row)
    else:
        with open(args.output_file, "w", encoding="utf-8") as results_file:
            results_writer = csv.writer(results_file)
            header_row = [
                "input_kg_file_name",
                "base_kg_file_name",
                "eval_type",
                "recall",
                "hit_count",
                "rel_count",
                "base_kg_size",
                "base_count",
                "base_coverage",
            ]
            rows.append(header_row)
            results_writer.writerow(header_row)

    base_file_name = os.path.basename(args.base_kg_file)
    input_kg_file_name = os.path.basename(args.input_kg_file)

    with open(args.output_file, "a", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        for eval_type in ["full", "classes", "instances"]:
            if eval_type == "full":
                (test_kg, test_base_kg) = eval_kg, eval_base_kg
            elif eval_type == "classes":
                (test_kg, test_base_kg) = (eval_types_kg, base_types_kg)
            elif eval_type == "instances":
                (test_kg, test_base_kg) = (eval_examples_kg, base_examples_kg)
            (
                test_hit_count,
                test_rel_count,
                test_base_kg_size,
                test_precision,
                test_recall,
                test_base_count,
                test_base_coverage,
            ) = evaluate_recall(test_kg, test_base_kg)
            logging.info(
                "%s hit_count, rel_count, base_kg_size, precision, recall, "
                "base_count, base_coverage: (%f, %f, %f, %f, %f, %f, %f)",
                eval_type,
                test_hit_count,
                test_rel_count,
                test_base_kg_size,
                test_precision,
                test_recall,
                test_base_count,
                test_base_coverage,
            )
            new_row = [
                input_kg_file_name,
                base_file_name,
                eval_type,
                f"{test_recall:.4f}",
                f"{test_hit_count:d}",
                f"{test_rel_count:d}",
                f"{test_base_kg_size:d}",
                f"{test_base_count:d}",
                f"{ test_base_coverage:.4f}",
            ]
            if new_row not in rows:
                rows.append(new_row)
                writer.writerow(new_row)

    with open(args.output_md, "w", encoding="utf-8") as md_file:
        md_writer = MarkdownTableWriter(
            table_name="Recall Results",
            headers=rows[0],
            value_matrix=rows[1:],
        )
        md_writer.stream = md_file
        md_writer.write_table(flavor="gfm")

    logging.info("Took: %f seconds.", time.time() - start_time)
