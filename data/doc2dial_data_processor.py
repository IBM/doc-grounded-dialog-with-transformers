""" Parsing and processing Doc2Dial data, loading into SQUAD style examples.
IBM 2020
License: Apache-2.0
"""
import json
import os
import re
from collections import defaultdict
from tqdm import tqdm

from transformers.utils import logging
from transformers.data.processors.utils import DataProcessor


# Store the tokenizers which insert 2 separators tokens
MULTI_SEP_TOKENS_TOKENIZERS_SET = {"roberta", "camembert", "bart", "mpnet"}
logger = logging.get_logger(__name__)

DD_PREPROCESSING_VERSION = "d2dv1"
IGNORED_UTTERANCES = ['NA', 'null', 'Null', '123', 'N/A', ""]
KEY_DOC_TEXT = "doc_text"
KEY_DIAL_DATA = "dial_data"
KEY_TEXT_SPAN = "text_spans"
KEY_TEXT_REFERENCE = "text_reference"
KEY_TURNS, KEY_UTTERANCE = "turns", "utterance"
KEY_ROLE, KEY_DA = "role", "da"
KEY_DIAL_ID, KEY_TURN_ID = "dial_id", "turn_id"
KEY_TS_START, KEY_TS_END, KEY_TS_TEXT = "start", "end", "text"
KEY_REF_START, KEY_REF_END = "start", "end"
KEY_TS_ID, KEY_REF_TS_ID = "ts_id", "ts_id"
AGENT_ROLE, USER_ROLE = "agent", "user"
DIAL_CONTEXT_LAST2, DIAL_CONTEXT_ALL = "last2", "all"


def read_dials(dial_path):
    """
    Read the dialogues (in doc2dial format) into dict with keys as document title associated with the dialogue.
    """
    all_dials = defaultdict(list)
    with open(dial_path, "r", encoding="utf-8") as dial_json:
        data = json.load(dial_json)
        max_num_turns = 0
        for dial in data[KEY_DIAL_DATA]:
            all_dials[dial["title"]].append(dial)
            num_turns = len(dial[KEY_TURNS])
            if num_turns > max_num_turns:
                max_num_turns = num_turns
        logger.info("Read {} dialogues for {} docs.".format(len(data[KEY_DIAL_DATA]), len(all_dials)))
    return dict(all_dials), max_num_turns


def read_docs(doc_path):
    """
    Read document (in doc2dial format) into dict with keys as doc_id
    """
    all_docs = {}
    with open(doc_path) as doc_json:
        data = json.load(doc_json)
        for doc in data["doc_data"]:
            text_spans = {}
            for text_span in doc[KEY_TEXT_SPAN]:
                text_spans[text_span[KEY_TS_ID]] = text_span
            doc[KEY_TEXT_SPAN] = text_spans
            all_docs[doc["doc_id"]] = doc
    # print("Read {} docs.".format(len(all_docs)))
    return all_docs


def delete_pretrailing_whitespace(all_docs):
    """
    Delete the pre-trailing '\n's in the beginning of the documents in Doc2Dial dataset, these caused problem with QA.
    """
    deleted_pretrailing_spaces = {}
    for doc_id, doc in all_docs.items():
        del_positions = [m.start() for m in re.finditer('\n', doc[KEY_DOC_TEXT])]
        n_del = 0
        last = -1
        # Delete the pretrailing \n chars
        for i in del_positions:
            if i == last + 1:
                n_del += 1
                last += 1
            else:
                break
        deleted_pretrailing_spaces[doc_id] = n_del
        if n_del == 0:
            continue
        all_docs[doc_id][KEY_DOC_TEXT] = doc[KEY_DOC_TEXT][n_del:]
        # Also change text-span data accordingly to reflect the deleted whitespaces
        for ts_id in doc[KEY_TEXT_SPAN].keys():
            # change start position if necessary
            if doc[KEY_TEXT_SPAN][ts_id][KEY_TS_START] < n_del:
                all_docs[doc_id][KEY_TEXT_SPAN][ts_id][KEY_TS_START] = 0
            else:
                all_docs[doc_id][KEY_TEXT_SPAN][ts_id][KEY_TS_START] -= n_del
            # change end position if necessary
            if doc[KEY_TEXT_SPAN][ts_id][KEY_TS_END] < n_del:
                all_docs[doc_id][KEY_TEXT_SPAN][ts_id][KEY_TS_END] = 0
            else:
                all_docs[doc_id][KEY_TEXT_SPAN][ts_id][KEY_TS_END] -= n_del
            # change doc_text
            start_position = all_docs[doc_id][KEY_TEXT_SPAN][ts_id][KEY_TS_START]
            end_position = all_docs[doc_id][KEY_TEXT_SPAN][ts_id][KEY_TS_END]
            all_docs[doc_id][KEY_TEXT_SPAN][ts_id][KEY_TS_TEXT] = \
                all_docs[doc_id][KEY_DOC_TEXT][start_position:end_position]
    return all_docs, deleted_pretrailing_spaces


def deleted_pretrailing_whitespace_refs(all_dials, deleted_pretrailing):
    for doc_title, dials in all_dials.items():
        n_del = deleted_pretrailing[doc_title]
        if n_del > 0:
            for dial in dials:
                for turn in dial[KEY_TURNS]:
                    for ref in turn[KEY_TEXT_REFERENCE]:
                        if ref[KEY_REF_START] >= n_del:
                            ref[KEY_REF_START] -= n_del
                        else:
                            ref[KEY_REF_START] = 0
                        ref[KEY_REF_END] -= n_del
    return all_dials


def get_start_end_candidates(tss):
    start_candidates, end_candidates = [], []
    for ts_id, ts in tss.items():
        start_candidates.append(ts[KEY_TS_START])
        end_candidates.append(ts[KEY_TS_END])
    return start_candidates, end_candidates


def create_answers_merging_text_ref(refs, tss):
    output = []
    refs = sorted(refs, key=lambda i: int(i[KEY_REF_TS_ID]))
    all_consecutive_tss = []
    consecutive_tss = []
    for idx, ref in enumerate(refs):
        if not consecutive_tss or int(ref[KEY_REF_TS_ID]) == int(consecutive_tss[-1]) + 1:
            consecutive_tss.append(ref[KEY_REF_TS_ID])
        else:
            all_consecutive_tss.append(consecutive_tss)
            consecutive_tss = [ref[KEY_REF_TS_ID]]
    all_consecutive_tss.append(consecutive_tss)
    if len(all_consecutive_tss) > 1:
        all_consecutive_tss.reverse()
    for con_tss in all_consecutive_tss:
        answer = {
            "answer_start": tss[con_tss[0]][KEY_TS_START],
            "answer_end": tss[con_tss[-1]][KEY_TS_END],
            "ts_id": con_tss,
        }
        output.append(answer)
    return output


def construct_qa(question, q_id, turn, doc):
    inc_multiple_ts, inc_multi_consec_tss, inc_long_answer = 0, 0, 0
    qa = {"question": question,
          "id": q_id,
          }
    if KEY_TEXT_REFERENCE not in turn.keys() or len(turn[KEY_TEXT_REFERENCE]) == 0:
        qa["is_impossible"] = True
        logger.info("is_impossible")
    else:
        qa["is_impossible"] = False
        if len(turn[KEY_TEXT_REFERENCE]) > 1:
            inc_multiple_ts += 1
        qa["answers"] = create_answers_merging_text_ref(turn[KEY_TEXT_REFERENCE], doc[KEY_TEXT_SPAN])
        if len(qa["answers"]) < 1:
            assert (len((qa["answers"])) >= 1)
        if len(qa["answers"]) > 1:
            inc_multi_consec_tss += 1
        for answer in qa["answers"]:
            answer["text"] = doc[KEY_DOC_TEXT][answer["answer_start"]:answer["answer_end"]]
            if len(answer["text"].split()) > 60:
                inc_long_answer += 1
    return qa, inc_multiple_ts, inc_multi_consec_tss, inc_long_answer


def get_squad_input_data_agent(all_docs, all_dials, reverse=True):
    """
    From dict of dialogues and dict of docs (both with keys as document titles),
    create squad style QA dataset to identify knowledge for agent turn with the last two utterances
    """
    all_data = []
    n_dial, n_question = 0, 0
    n_multiple_ts, n_multi_consec_tss, n_long_answer = 0, 0, 0
    n_second_user, n_second_agent, n_ignored = 0, 0, 0
    for title, dials in all_dials.items():
        qas = []
        doc = all_docs[title]
        start_pos_char_candidates, end_pos_char_candidates = get_start_end_candidates(doc[KEY_TEXT_SPAN])
        for dial in dials:
            last_agent_utterance = ""
            this_user_utterance = ""
            next_agent_turn = {}
            for idx, turn in enumerate(dial[KEY_TURNS]):
                if turn[KEY_UTTERANCE] in IGNORED_UTTERANCES:
                    n_ignored += 1
                    continue
                if turn[KEY_ROLE] == AGENT_ROLE:
                    if last_agent_utterance:
                        n_second_agent += 1
                    last_agent_utterance += turn[KEY_UTTERANCE] + " "
                    continue
                if this_user_utterance:
                    n_second_user += 1
                this_user_utterance += turn[KEY_UTTERANCE] + " "
                if idx + 1 < len(dial[KEY_TURNS]):
                    if dial[KEY_TURNS][idx+1][KEY_ROLE] == AGENT_ROLE:
                        next_agent_turn = dial[KEY_TURNS][idx+1]
                    else:
                        continue
                if reverse:
                    question = this_user_utterance + " " + last_agent_utterance
                else:
                    question = last_agent_utterance + " " + this_user_utterance
                this_user_utterance = ""
                last_agent_utterance = ""
                q_id = dial[KEY_DIAL_ID] + "_" + str(turn[KEY_TURN_ID])
                qa, inc_multiple_ts, inc_multi_consec_tss, inc_long_answer = \
                    construct_qa(question, q_id, next_agent_turn, doc)
                n_multiple_ts += inc_multiple_ts
                n_multi_consec_tss += inc_multi_consec_tss
                n_long_answer += inc_long_answer

                qas.append(qa)
                n_question += 1
            n_dial += 1

        paragraph = {
            "qas": qas,
            "context": doc[KEY_DOC_TEXT],
            "start_candidates": start_pos_char_candidates,
            "end_candidates": end_pos_char_candidates,
        }
        data = {"title": title,
                "paragraphs": [paragraph],
                }
        all_data.append(data)
    output = all_data
    return output


def get_squad_input_data(all_docs, all_dials, reverse=True):
    """
    From dict of dialogues and dict of docs (both with keys as document titles),
    create squad style QA dataset to identify knowledge for user turn with the last two utterances
    """
    all_data = []
    n_dial, n_question = 0, 0
    n_multiple_ts, n_multi_consec_tss, n_long_answer = 0, 0, 0
    n_second_agent, n_ignored = 0, 0
    for title, dials in all_dials.items():
        qas = []
        doc = all_docs[title]
        start_pos_char_candidates, end_pos_char_candidates = get_start_end_candidates(doc[KEY_TEXT_SPAN])
        for dial in dials:
            # print("dialogue: {}".format(dial[KEY_DIAL_ID]))
            last_turn_role = ""
            last_agent_utterance = ""
            for turn in dial[KEY_TURNS]:
                if turn[KEY_UTTERANCE] in IGNORED_UTTERANCES:
                    n_ignored += 1
                    continue
                if turn[KEY_ROLE] == AGENT_ROLE:
                    if last_turn_role == AGENT_ROLE:
                        n_second_agent += 1
                    else:
                        last_agent_utterance = ""
                    last_turn_role = turn[KEY_ROLE]
                    last_agent_utterance += turn[KEY_UTTERANCE] + ""
                    continue
                if reverse:
                    question = turn[KEY_UTTERANCE] + " " + last_agent_utterance
                else:
                    question = last_agent_utterance + " " + turn[KEY_UTTERANCE]
                last_turn_role = turn[KEY_ROLE]
                last_agent_utterance = ""
                q_id = dial[KEY_DIAL_ID] + "_" + str(turn[KEY_TURN_ID])
                qa, inc_multiple_ts, inc_multi_consec_tss, inc_long_answer = \
                    construct_qa(question, q_id, turn, doc)
                n_multiple_ts += inc_multiple_ts
                n_multi_consec_tss += inc_multi_consec_tss
                n_long_answer += inc_long_answer

                qas.append(qa)
                n_question += 1
            n_dial += 1

        paragraph = {
            "qas": qas,
            "context": doc[KEY_DOC_TEXT],
            "start_candidates": start_pos_char_candidates,
            "end_candidates": end_pos_char_candidates,
        }
        data = {"title": title,
                "paragraphs": [paragraph],
                }
        all_data.append(data)
    output = all_data
    return output


def get_squad_input_data_all_utterances(all_docs,
                                        all_dials,
                                        role_to_predict=USER_ROLE,
                                        include_da=False,
                                        reverse=True,
                                        qa_ids_to_include=None,
                                        ):
    """
    From dict of dialogues and dict of docs (both with keys as document titles),
    create squad style QA dataset with all utterances.
    """
    all_data = []
    n_dial, n_question = 0, 0
    n_multiple_ts, n_multi_consec_tss, n_long_answer = 0, 0, 0
    n_ignored = 0
    logger.info("new preprocessing all method being used.")
    for title, dials in all_dials.items():
        qas = []
        doc = all_docs[title]
        start_pos_char_candidates, end_pos_char_candidates = get_start_end_candidates(doc[KEY_TEXT_SPAN])
        for dial in dials:
            all_prev_utterances = []
            all_prev_da = []
            turn_to_predict = {}
            for idx, turn in enumerate(dial[KEY_TURNS]):
                if turn[KEY_UTTERANCE] in IGNORED_UTTERANCES:
                    n_ignored += 1
                    continue
                if include_da:
                    # This is a hack working for certain tokenizers.
                    # To be deprecated since it only affects include_da cases.
                    sep_tokens = '[SEP]'
                    all_prev_utterances.append(' '.join[turn[KEY_UTTERANCE], sep_tokens, turn[KEY_DA]])
                else:
                    all_prev_utterances.append(turn[KEY_UTTERANCE])
                all_prev_da.append(turn[KEY_DA])
                if turn[KEY_ROLE] == AGENT_ROLE:
                    continue
                if role_to_predict == USER_ROLE:
                    turn_to_predict = turn
                elif role_to_predict == AGENT_ROLE:
                    if idx + 1 < len(dial[KEY_TURNS]):
                        if dial[KEY_TURNS][idx+1][KEY_ROLE] == AGENT_ROLE:
                            turn_to_predict = dial[KEY_TURNS][idx+1]
                        else:
                            continue
                else:
                    assert(role_to_predict in [USER_ROLE, AGENT_ROLE])

                if reverse:
                    question = " ".join(list(reversed(all_prev_utterances)))
                else:
                    question = " ".join(all_prev_utterances)
                # if there is a white-list of qa ids, use it to filter data
                q_id = dial[KEY_DIAL_ID] + "_" + str(turn[KEY_TURN_ID])
                if qa_ids_to_include and q_id not in qa_ids_to_include:
                    continue

                qa, inc_multiple_ts, inc_multi_consec_tss, inc_long_answer = \
                    construct_qa(question, q_id, turn_to_predict, doc)
                n_multiple_ts += inc_multiple_ts
                n_multi_consec_tss += inc_multi_consec_tss
                n_long_answer += inc_long_answer

                qas.append(qa)
                n_question += 1
            n_dial += 1

        paragraph = {
            "qas": qas,
            "context": doc[KEY_DOC_TEXT],
            "start_candidates": start_pos_char_candidates,
            "end_candidates": end_pos_char_candidates,
        }
        data = {"title": title,
                "paragraphs": [paragraph],
                }
        all_data.append(data)
    output = all_data
    return output


class Doc2DialProcessor(DataProcessor):
    train_dial_file = None
    dev_dial_file = None
    doc_file = None

    def _get_example_from_tensor_dict(self, tensor_dict, evaluate=False):
        # This function is modified from Huggingface SquadProcessor._get_example_from_tensor_dict
        if not evaluate:
            answer = tensor_dict["answers"]["text"][0].numpy().decode("utf-8")
            answer_start = tensor_dict["answers"]["answer_start"][0].numpy()
            answers = []
        else:
            answers = [
                {"answer_start": start.numpy(), "text": text.numpy().decode("utf-8")}
                for start, text in zip(tensor_dict["answers"]["answer_start"], tensor_dict["answers"]["text"])
            ]

            answer = None
            answer_start = None

        return Doc2DialExample(
            qas_id=tensor_dict["id"].numpy().decode("utf-8"),
            question_text=tensor_dict["question"].numpy().decode("utf-8"),
            context_text=tensor_dict["context"].numpy().decode("utf-8"),
            answer_text=answer,
            start_position_character=answer_start,
            title=tensor_dict["title"].numpy().decode("utf-8"),
            answers=answers,
        )

    def get_examples_from_dataset(self, dataset, evaluate=False):
        # This function is modified from Huggingface SquadProcessor.get_examples_from_dataset
        if evaluate:
            dataset = dataset["validation"]
        else:
            dataset = dataset["train"]
        examples = []
        for tensor_dict in tqdm(dataset):
            examples.append(self._get_example_from_tensor_dict(tensor_dict, evaluate=evaluate))
        return examples

    def get_train_examples(self, data_dir, dial_filename=None, doc_filename=None,
                           reverse=True,
                           get_utterances=DIAL_CONTEXT_LAST2,
                           predict_agent=False,
                           include_da=False,
                           dial_qa_ids_filename=None,
                           ):
        if data_dir is None:
            data_dir = ""

        if self.train_dial_file is None:
            raise ValueError("Doc2DialProcessor should be instantiated via Doc2Dialv1Processor")

        dial_path = os.path.join(data_dir, self.train_dial_file if dial_filename is None else dial_filename)
        doc_path = os.path.join(data_dir, self.doc_file if doc_filename is None else doc_filename)
        all_docs = read_docs(doc_path)
        all_docs, deleted_pretrailing = delete_pretrailing_whitespace(all_docs)
        all_dials, max_num_turns = read_dials(dial_path)
        all_dials = deleted_pretrailing_whitespace_refs(all_dials, deleted_pretrailing)

        qa_ids_to_include = None
        if dial_qa_ids_filename:
            dial_qa_ids_path = os.path.join(data_dir, dial_qa_ids_filename)
            with open(dial_qa_ids_path, 'r') as qa_ids_in:
                qa_ids_to_include = [s.strip() for s in qa_ids_in.readlines()]
        if predict_agent:
            if get_utterances == DIAL_CONTEXT_ALL:
                input_data = get_squad_input_data_all_utterances(all_docs, all_dials,
                                                                 role_to_predict=AGENT_ROLE,
                                                                 reverse=reverse,
                                                                 include_da=include_da,
                                                                 qa_ids_to_include=qa_ids_to_include,
                                                                 )
            else:
                input_data = get_squad_input_data_agent(all_docs, all_dials, reverse=reverse)
        else:
            if get_utterances == DIAL_CONTEXT_ALL:
                input_data = get_squad_input_data_all_utterances(all_docs, all_dials,
                                                                 role_to_predict=USER_ROLE,
                                                                 reverse=reverse,
                                                                 include_da=include_da,
                                                                 qa_ids_to_include=qa_ids_to_include,
                                                                 )
            else:
                input_data = get_squad_input_data(all_docs, all_dials, reverse=reverse)

        return self._create_examples(input_data, "train")

    def get_dev_examples(self, data_dir, dial_filename=None, doc_filename=None,
                         reverse=True,
                         get_utterances=DIAL_CONTEXT_LAST2,
                         predict_agent=False,
                         include_da=False,
                         dial_qa_ids_filename=None,
                         ):
        if data_dir is None:
            data_dir = ""

        if self.dev_dial_file is None:
            raise ValueError("Doc2DialProcessor should be instantiated via Doc2DialV1Processor or Doc2DialV2Processor")

        dial_path = os.path.join(data_dir, self.dev_dial_file if dial_filename is None else dial_filename)
        doc_path = os.path.join(data_dir, self.doc_file if doc_filename is None else doc_filename)
        all_docs = read_docs(doc_path)
        all_docs, deleted_pretrailing = delete_pretrailing_whitespace(all_docs)
        all_dials, max_num_turns = read_dials(dial_path)
        all_dials = deleted_pretrailing_whitespace_refs(all_dials, deleted_pretrailing)
        qa_ids_to_include = None
        if dial_qa_ids_filename:
            dial_qa_ids_path = os.path.join(data_dir, dial_qa_ids_filename)
            with open(dial_qa_ids_path, 'r') as qa_ids_in:
                qa_ids_to_include = [s.strip() for s in qa_ids_in.readlines()]
        if predict_agent:
            if get_utterances == DIAL_CONTEXT_ALL:
                input_data = get_squad_input_data_all_utterances(all_docs, all_dials,
                                                                 role_to_predict=AGENT_ROLE,
                                                                 reverse=reverse,
                                                                 include_da=include_da,
                                                                 qa_ids_to_include=qa_ids_to_include,
                                                                 )
            else:
                input_data = get_squad_input_data_agent(all_docs, all_dials, reverse=reverse)
        else:
            if get_utterances == DIAL_CONTEXT_ALL:
                input_data = get_squad_input_data_all_utterances(all_docs, all_dials,
                                                                 role_to_predict=USER_ROLE,
                                                                 reverse=reverse,
                                                                 include_da=include_da,
                                                                 qa_ids_to_include=qa_ids_to_include,
                                                                 )
            else:
                input_data = get_squad_input_data(all_docs, all_dials, reverse=reverse)

        return self._create_examples(input_data, "dev")

    def _create_examples(self, input_data, set_type):
        # This function is Modified from Huggingface SquadProcessor._create_examples
        is_training = set_type == "train"
        examples = []
        for entry in tqdm(input_data):
            title = entry["title"]
            for paragraph in entry["paragraphs"]:
                context_text = paragraph["context"]
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position_character = None
                    answer_text = None
                    answers = []

                    is_impossible = qa.get("is_impossible", False)
                    if not is_impossible:
                        if is_training:  # even for dev/test set, only consider the first answer
                            answer = qa["answers"][0]
                            answer_text = answer["text"]
                            start_position_character = answer["answer_start"]
                        else:
                            answers = qa["answers"]

                    example = Doc2DialExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=answer_text,
                        start_position_character=start_position_character,
                        title=title,
                        is_impossible=is_impossible,
                        answers=answers,
                    )
                    examples.append(example)
        return examples


class Doc2DialV1Processor(Doc2DialProcessor):
    doc_file = "doc2dial_document_data.json"
    train_dial_file = "doc2dial_dial_data_train.json"
    dev_dial_file = "doc2dial_dial_data_dev.json"


class Doc2DialExample(object):
    """
    Added word_to_char_offset on top of SquadExample in Huggingface Transformers.
    """

    def __init__(
            self,
            qas_id,
            question_text,
            context_text,
            answer_text,
            start_position_character,
            title,
            answers=None,
            is_impossible=False,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text
        self.title = title
        self.is_impossible = is_impossible
        self.answers = answers

        self.start_position, self.end_position = 0, 0

        doc_tokens = []
        char_to_word_offset = []
        word_to_char_offset = []
        prev_is_whitespace = True

        # Split on whitespace so that different tokens may be attributed to their original position.
        for idx, c in enumerate(self.context_text):
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                    word_to_char_offset.append(idx)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        word_to_char_offset.append(len(self.context_text))
        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset
        self.word_to_char_offset = word_to_char_offset

        # Start and end positions only has a value during training.
        if start_position_character is not None and not is_impossible:
            self.start_position = char_to_word_offset[start_position_character]
            self.end_position = char_to_word_offset[
                min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
            ]
            self.start_char_pos = start_position_character
            self.end_char_pos = min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)


def _is_whitespace(char):
    return char in [" ", "\t", "\r", "\n"] or ord(char) == 0x202F
