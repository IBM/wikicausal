"""
This script takes in a causal KG of wikidata events, and measures "precision"
through the use of a large language model (LLM) in lieu of manual evaluation.
The script as-is uses smaller publicly available models and runs on a machine 
without a GPU, although it will be much faster with a GPU. You can add a 
custom LLM by modifying or rewriting the util.generate function.
"""

import time
import argparse
import json
import csv
import logging
import os
import sys
from pytablewriter import MarkdownTableWriter
from util.generate import generate_answer_instruct


def generate_instruction_prompt(cause_text, effect_text):
    """
    Generate instruction prompt from cause_text and effect_text
    """
    return (
        "Definition: Answer the question with yes or no. "
        "Now complete the following example - "
        f"Input: Question: Could {cause_text} result in {effect_text}? Output:"
    )


def generate_question_prompt(cause_text, effect_text):
    """
    Generate question prompt from cause_text and effect_text
    """
    return (
        "Answer the question with yes or no.\n"
        f"Question: Does {cause_text} cause {effect_text}? "
        "Answer:"
    )


def validate(cause_text, effect_text, tries=5):
    """
    Validates a causal relation between cause_text and effect_text.
    The more the "tries" the more reliable the outcome, but it takes
    longer to validate.
    """
    yes_count = 0
    for _ in range(tries):
        prompt = generate_instruction_prompt(cause_text, effect_text)
        answer = generate_answer_instruct(prompt)
        logging.debug(prompt)
        logging.debug(answer)
        if answer.startswith("yes"):
            yes_count += 1
    if yes_count >= ((tries // 2) + 1):
        return True
    return False


def precision_score(evaluation_results):
    """
    Returns the percentage of True (correct) results in evaluation_results is as
    precision score.
    """
    if len(evaluation_results) == 0:
        return 0
    return len([1 for pair in evaluation_results if evaluation_results[pair]]) / len(
        evaluation_results
    )


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_kg_file",
        required=False,
        help="Input KG for precision evaluation.",
        default="data/base-kg/base-kg-v1.jsonl",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        required=False,
        help="Output CSV file.",
        default="results/precision-v1.csv",
    )
    parser.add_argument(
        "-m",
        "--output_md",
        required=False,
        help="Output Markdown file.",
        default="results/precision-v1.md",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logging.info("Starting evaluation with args: %s", args)

    eval_all = {}
    eval_types = {}
    eval_instances = {}
    with open(args.input_kg_file, "r", encoding="utf-8") as input_kg_file:
        for line in input_kg_file:
            obj = json.loads(line)
            if "event" not in obj and "cause" not in obj:
                logging.error(
                    "Input KG does not have 'event' or 'cause' field in line: %s", line
                )
                sys.exit(1)
            source_label = (
                obj["event"]["label"] if "event" in obj else obj["cause"]["label"][0]
            )
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
                target_label = c["label"]
                pair = (source_label, target_label)
                if pair not in eval_all:
                    validation_result = validate(pair[0], pair[1])
                    eval_all[pair] = validation_result
                    eval_types[pair] = validation_result
                if "examples" in c and len(c["examples"]) > 0:
                    for e in c["examples"]:
                        cause_instance_label = e["cause"]["label"]
                        effect_instance_label = e["effect"]["label"]
                        instance_pair = (cause_instance_label, effect_instance_label)
                        if instance_pair not in eval_all:
                            instance_validation_result = validate(
                                instance_pair[0], instance_pair[1]
                            )
                            eval_all[instance_pair] = instance_validation_result
                            eval_instances[instance_pair] = instance_validation_result

    logging.info(
        "Done evaluating KG %s with %d relations, including %d type relations "
        "and %d examples.",
        args.input_kg_file,
        len(eval_all),
        len(eval_types),
        len(eval_instances),
    )
    full_precision = precision_score(eval_all)
    types_precision = precision_score(eval_types)
    instances_precision = precision_score(eval_instances)

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
                "full precision",
                "types precision",
                "instances precision",
            ]
            rows.append(header_row)
            results_writer.writerow(header_row)
    input_kg_file_name = os.path.basename(args.input_kg_file)
    with open(args.output_file, "a", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        new_row = [
            input_kg_file_name,
            f"{full_precision:.4f}",
            f"{types_precision:.4f}",
            f"{instances_precision:.4f}",
        ]
        if new_row not in rows:
            rows.append(new_row)
            writer.writerow(new_row)

    with open(args.output_md, "w", encoding="utf-8") as md_file:
        md_writer = MarkdownTableWriter(
            table_name="Precision Results",
            headers=rows[0],
            value_matrix=rows[1:],
        )
        md_writer.stream = md_file
        md_writer.write_table(flavor="gfm")

    logging.info("Took: %f seconds.", time.time() - start_time)
