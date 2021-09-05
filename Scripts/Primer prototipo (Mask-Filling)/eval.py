from dataset import get_dataloader
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm
import torch
import logging
from torch.nn import functional as F
from torch.cuda.amp import autocast
import os
import argparse
import statistics

def do_eval_function(precision_1, precision_10, precision_100, median_and_variance, k_words, predictions, golds):
    for i, gold in enumerate(golds):
        predictions_i = predictions[i].split(" ")[1:]
        if gold in predictions_i:
            index = predictions_i.index(gold)
            median_and_variance.append(index)
            if index <= 0:
                precision_1 += 1
            if index <= 9:
                precision_10 += 1
            if index <= 99:
                precision_100 += 1
    return precision_1, precision_10, precision_100, median_and_variance


def do_predict_function(predictions, tokenizer, output_path, golds):
    with open(output_path, "a+", encoding="utf8") as output_file:
        predictions = tokenizer.batch_decode(predictions)
        golds = [gold + " ||| " for gold in golds]
        print("\n".join(gold + prediction for gold,prediction in zip(golds, predictions)), file=output_file)


def eval(
    model_name_or_path: str,
    dataset_path: str,
    k_words: int,
    batch_size: int,
    do_eval,
    do_predict,
    fp16: bool,
    output_path: str = None,
):
    assert do_eval or do_predict
    assert (not do_predict) or output_path
    assert batch_size > 0

    # assert (
    #    not fp16
    # ) or torch.cuda.is_available(), "FP16 is only compatible with Cuda devices"

    if torch.cuda.is_available():
        device: torch.device = torch.device("cuda:0")
    else:
        device: torch.device = torch.device("cpu")
        logging.warning(
            "GPU not found, using CPU, training will be very slow. CPU NOT COMPATIBLE WITH FP16"
        )

    if do_predict and os.path.exists(output_path):
        os.remove(output_path)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path
    )
    print("Loading model...")

    model = AutoModelForMaskedLM.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path, return_dict=True
    ).to(device=device)

    print("Loading dataset...")
    data_loader = get_dataloader(
        filename=dataset_path,
        tokenizer=tokenizer,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available(),
    )

    precision_1 = 0
    precision_10 = 0
    precision_100 = 0
    median_and_variance = []
    total_examples = 0

    for sentence_tokens, golds_ids, positions, gold_words in tqdm(
        data_loader, desc="Running inference"
    ):

        with autocast(enabled=fp16), torch.no_grad():
            for key, value in sentence_tokens.items():
                sentence_tokens[key] = sentence_tokens[key].to(device)

            output = model(**sentence_tokens)
            output = output.logits
            output = output[range(output.size()[0]), positions, :]
            softmax = F.softmax(output, dim=-1)  # It should not be necessary
            top_k = torch.topk(softmax, k_words, dim=1)[1].cpu()

        if do_eval:
            #correct_predictions += do_eval_function(predictions=top_k, golds=golds_ids)
            #correct_predictions += do_eval_function(predictions=tokenizer.batch_decode(top_k), golds=gold_words, k_words)
            precision_1, precision_10, precision_100, median_and_variance = do_eval_function(precision_1=precision_1, precision_10=precision_10, precision_100=precision_100, median_and_variance=median_and_variance, k_words=k_words, predictions=tokenizer.batch_decode(top_k), golds=gold_words)
            total_examples += len(top_k)
        if do_predict:
            do_predict_function(
                predictions=top_k, tokenizer=tokenizer, output_path=output_path, golds=gold_words
            )

    if do_eval:
        mediana = statistics.median(median_and_variance)
        prec1 = precision_1 / total_examples
        prec10 = precision_10 / total_examples
        prec100 = precision_100 / total_examples
        
        varianza = statistics.variance(median_and_variance)
        
        print("=========================")
        print(total_examples, precision_1, precision_10, precision_100)
        print("Mediana: " + str(mediana))
        print("Palabras acertadas en precision P@1: {:.2%}".format(prec1))
        print("Palabras acertadas en precision P@10: {:.2%}".format(prec10))
        print("Palabras acertadas en precision P@100: {:.2%}".format(prec100))
        print("Palabras sin acertar: {}".format(total_examples - precision_100))
        print("Varianza: " + str(varianza))
        print("=========================")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Hugging face model name or path",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Dataset path",
    )

    parser.add_argument(
        "--k_words",
        type=int,
        default=10,
        help="Number of most probable words to predict. "
        "For calculating the accuracy we consider that the "
        "prediction is correct if the gold word is one of the top-k words.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size",
    )

    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="Evaluate the model accuracy",
    )

    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="Predict the most k_words probable words",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use fp16",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        required=False,
        help="Output path for predictions",
    )

    args = parser.parse_args()

    eval(
        model_name_or_path=args.model_name_or_path,
        dataset_path=args.dataset_path,
        k_words=args.k_words,
        batch_size=args.batch_size,
        do_eval=args.do_eval,
        do_predict=args.do_predict,
        fp16=args.fp16,
        output_path=args.output_path,
    )
