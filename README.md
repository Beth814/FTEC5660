# FTEC5660 Homework 1: Receipt Chain

Build a LangChain pipeline that reads every supermarket receipt in a folder
with the vision-capable DeepSeek Flash model and answers these two questions:

1. How much money did I spend in total for these bills?
2. How much would I have had to pay without the discount?

For this homework, **amount spent** means the final payment after the receipt's
rounding line. **Without the discount** means the sum of the original positive
item prices: add back every promotion, coupon, member, app, packaging-damage,
and percentage discount, but do not add back rounding.

## Student task

Only edit the two functions in `hw1.py` that contain `### YOUR CODE HERE`:

- `build_chain()` creates your LangChain chain.
- `answer_queries()` runs the chain on the receipt images and returns one final
  response for each question.

You may use prompt chaining, routing, parallel calls, reflection, or a
combination. Your final responses should each contain one HKD amount. Do not
hard-code filenames or public answers; grading uses unseen receipt folders.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put your DeepSeek key after `DEEPSEEK_API_KEY=` in `.env`, then run:

```bash
python3 hw1.py --image-folder public_test
```

The program creates `results.csv` in the current directory. Its columns are
`query`, `model_response`, and `correctness`. The public answers are in
`public_test/ground_truth.json`. The starter intentionally returns the dummy
response `please design your chain to answer these two queries.` so it runs
before you add any API code.

The required model is `deepseek-v4-flash-vision-exp`, the vision-capable
DeepSeek Flash model. JPEG, PNG, GIF, and WebP inputs are accepted by the
homework runner.


## Homework 1 solution

My solution processes the receipts one at a time using a LangChain chain with deepseek-v4-flash-vision-exp. The prompt asks the model to extract the final payment after rounding, the subtotal before rounding, and all discount amounts as JSON. It also distinguishes purchase amounts from cash tendered, change and card balances, and avoids counting repeated savings summaries twice. The chain parses the JSON, checks the required fields and converts the amounts to Decimal values. Python then sums the final payments for the first question and the subtotals plus discounts for the second question, without adding rounding to the discount amounts. Each response contains one HKD amount with two decimal places. In one run on the seven public receipts, both aggregate answers were marked correct: HK$1974.30 and HK$2348.20.

```mermaid
flowchart TD
    A["Receipt images"] --> B["Process each receipt"]
    B --> C["LangChain prompt and DeepSeek"]
    C --> D["Parse JSON and validate amounts"]
    D --> E["Final payment after rounding"]
    D --> F["Subtotal plus discounts"]
    E --> G["Sum across receipts using Decimal"]
    F --> G
    G --> H["Return two HKD amounts"]
    H --> I["Provided runner writes results.csv"]
```

